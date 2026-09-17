#!/usr/bin/env python3
"""Train the released DeepCA model on paired ImageCAS NPZ archives."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from deepca.checkpoint import (
    atomic_torch_save,
    critic_state_dict,
    generator_state_dict,
    make_training_checkpoint,
    restore_rng_state,
    safe_torch_load,
)
from deepca.config import (
    assert_resume_config_compatible,
    config_fingerprint,
    load_config,
    save_resolved_config,
)
from deepca.data import (
    ImageCASDataset,
    build_projection_index,
    resolve_case_pairs,
    worker_seed,
)
from deepca.engine import (
    build_optimizer,
    build_scheduler,
    make_grad_scaler,
    seed_everything,
    train_one_epoch,
    validate,
)
from deepca.modeling import architecture_metadata, build_models
from deepca.splits import load_resolved_splits


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training YAML configuration.")
    parser.add_argument("--resume", help="Rich last/best checkpoint to resume.")
    parser.add_argument("--device", help="Override configured device, e.g. cuda:0 or cpu.")
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {value!r} was requested but torch.cuda.is_available() is False."
        )
    return device


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")


def _pair_manifest(pairs_by_split: Mapping[str, Any]) -> dict[str, Any]:
    return {
        split: [
            {
                "case_id": pair.case_id,
                "vessel_type": pair.vessel_type,
                "case_number": pair.case_number,
                "projection_path": str(pair.projection_path),
                "ground_truth_path": str(pair.ground_truth_path),
            }
            for pair in pairs
        ]
        for split, pairs in pairs_by_split.items()
    }


def _make_loader(
    dataset: ImageCASDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "drop_last": bool(drop_last),
        "pin_memory": bool(pin_memory),
        "worker_init_fn": worker_seed,
        "generator": generator,
        "persistent_workers": False,
    }
    if workers > 0:
        options["prefetch_factor"] = 1
    return DataLoader(**options)


def _validate_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "data", "model", "training"):
        if not isinstance(config.get(section), Mapping):
            raise KeyError(f"Configuration requires a {section!r} mapping.")
    data = config["data"]
    for key in ("vessel_type", "projection_root", "ground_truth_root", "split_json"):
        if key not in data:
            raise KeyError(f"data.{key} is required.")
    vessel = str(data["vessel_type"]).lower()
    if vessel not in {"lca", "rca"}:
        raise ValueError("data.vessel_type must be 'lca' or 'rca'.")
    model_size = int(config["model"].get("volume_size", 128))
    data_size = int(data["preprocessing"].get("volume_size", 128))
    if model_size != data_size:
        raise ValueError(
            f"model.volume_size ({model_size}) must equal "
            f"data.preprocessing.volume_size ({data_size})."
        )


def _validate_resume_config(
    checkpoint: Mapping[str, Any], current_config: Mapping[str, Any]
) -> None:
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError(
            "Resume checkpoint does not contain a valid resolved 'config' mapping; "
            "refusing an unsafe resume."
        )
    assert_resume_config_compatible(current_config, saved_config)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    _validate_config(config)
    experiment = config["experiment"]
    data_config = config["data"]
    training = config["training"]
    seed = int(experiment.get("seed", 1))
    seed_everything(seed, deterministic=bool(training.get("deterministic", True)))

    output_dir = Path(experiment["output_dir"]).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    vessel = str(data_config["vessel_type"]).lower()
    resolved_splits = load_resolved_splits(data_config["split_json"], vessel)
    split_manifest = resolved_splits.as_manifest()
    pairs_by_split: dict[str, Any] = {}
    projection_index = build_projection_index(
        data_config["projection_root"], expected_vessel=vessel
    )
    for split_name, case_ids in resolved_splits.split_ids.items():
        pairs, failures = resolve_case_pairs(
            case_ids,
            projection_root=data_config["projection_root"],
            ground_truth_root=data_config["ground_truth_root"],
            vessel_type=vessel,
            projection_index=projection_index,
            strict=True,
        )
        if failures:
            raise RuntimeError(f"Unexpected strict pairing failures: {failures}")
        pairs_by_split[split_name] = pairs
    split_manifest["resolved_pairs"] = _pair_manifest(pairs_by_split)
    _atomic_json(output_dir / "resolved_cases.json", split_manifest)

    train_dataset = ImageCASDataset(pairs_by_split["train"], config, training=True)
    val_dataset = ImageCASDataset(pairs_by_split["val"], config, training=False)

    configured_device = str(training.get("device", "cuda:0"))
    device = _resolve_device(args.device or configured_device)
    generator, critic = build_models(config, device)
    generator_optimizer = build_optimizer(
        generator.parameters(), training.get("generator_optimizer", training.get("optimizer", {}))
    )
    critic_optimizer = build_optimizer(
        critic.parameters(), training.get("critic_optimizer", training.get("optimizer", {}))
    )
    scheduler = build_scheduler(generator_optimizer, training.get("scheduler"))
    amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    metadata = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "architecture": architecture_metadata(generator, critic),
        "config_fingerprint": config_fingerprint(config),
        "scientific_conventions": {
            "projection_input": "per-view binary cone support backprojection",
            "view_aggregation": data_config["preprocessing"].get("combine", "sum"),
            "source_axis_order": "XYZ",
            "network_axis_order": "ZYX",
            "target_binarization": "vol > 0",
            "generator_output": "raw released-model regression",
            "default_evaluation_threshold": 0.5,
        },
    }
    _atomic_json(output_dir / "run_metadata.json", metadata)

    start_epoch = 0
    global_step = 0
    best_validation_l1 = math.inf
    early_stop_count = 0
    resume_path = args.resume or training.get("resume")
    if resume_path:
        checkpoint = safe_torch_load(resume_path, device)
        if int(checkpoint.get("schema_version", 0)) < 2:
            raise ValueError(
                "Legacy upstream checkpoints contain weights but not enough state for safe "
                "resume. Use them for evaluation or warm-start explicitly, not --resume."
            )
        _validate_resume_config(checkpoint, config)
        saved_architecture = checkpoint.get("architecture")
        current_architecture = architecture_metadata(generator, critic)
        if saved_architecture is None:
            raise ValueError("Resume checkpoint does not record architecture metadata.")
        if saved_architecture != current_architecture:
            raise ValueError(
                "Checkpoint/model architecture mismatch:\n"
                f"saved={saved_architecture}\ncurrent={current_architecture}"
            )
        if checkpoint.get("resolved_splits") != split_manifest:
            raise ValueError(
                "Checkpoint split/case manifest differs from the current resolved dataset."
            )
        generator.load_state_dict(generator_state_dict(checkpoint), strict=True)
        critic.load_state_dict(critic_state_dict(checkpoint), strict=True)
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if scheduler is not None:
            if checkpoint.get("scheduler") is None:
                raise ValueError("Configured scheduler has no state in the checkpoint.")
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_l1 = float(checkpoint["best_validation_l1"])
        early_stop_count = int(checkpoint.get("early_stop_count", 0))
        restore_rng_state(checkpoint["rng_state"])

    batch_size = int(training.get("batch_size", 3))
    workers = int(training.get("workers", 4))
    epochs = int(training.get("epochs", 200))
    validation_interval = int(training.get("validation_interval", 1))
    if validation_interval <= 0:
        raise ValueError("training.validation_interval must be positive.")
    patience = int(training.get("early_stopping_patience", 20))
    log_path = output_dir / "epochs.jsonl"

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        train_dataset.set_epoch(epoch)
        train_loader = _make_loader(
            train_dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=True,
            drop_last=bool(training.get("drop_last", True)),
            pin_memory=device.type == "cuda",
            seed=seed + epoch,
        )
        train_metrics, global_step = train_one_epoch(
            loader=train_loader,
            generator=generator,
            critic=critic,
            generator_optimizer=generator_optimizer,
            critic_optimizer=critic_optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            critic_steps_per_generator=int(training.get("critic_steps_per_generator", 2)),
            l1_weight=float(training.get("l1_weight", 100.0)),
            gradient_penalty_weight=float(training.get("gradient_penalty_weight", 10.0)),
            gradient_clip_norm=(
                None
                if training.get("gradient_clip_norm") is None
                else float(training["gradient_clip_norm"])
            ),
            global_step=global_step,
        )

        val_metrics: Optional[dict[str, float]] = None
        improved = False
        if (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs:
            val_loader = _make_loader(
                val_dataset,
                batch_size=batch_size,
                workers=workers,
                shuffle=False,
                drop_last=False,
                pin_memory=device.type == "cuda",
                seed=seed,
            )
            val_metrics = validate(
                loader=val_loader,
                generator=generator,
                critic=critic,
                device=device,
                amp_enabled=amp_enabled,
                l1_weight=float(training.get("l1_weight", 100.0)),
            )
            validation_l1 = float(val_metrics["l1_loss"])
            improved = validation_l1 < best_validation_l1
            if improved:
                best_validation_l1 = validation_l1
                early_stop_count = 0
            else:
                early_stop_count += 1
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(validation_l1)
                else:
                    scheduler.step()

        checkpoint = make_training_checkpoint(
            epoch=epoch,
            global_step=global_step,
            generator=generator,
            critic=critic,
            generator_optimizer=generator_optimizer,
            critic_optimizer=critic_optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_validation_l1=best_validation_l1,
            config=config,
            config_fingerprint=config_fingerprint(config),
            resolved_splits=split_manifest,
        )
        checkpoint["architecture"] = architecture_metadata(generator, critic)
        checkpoint["early_stop_count"] = early_stop_count
        atomic_torch_save(checkpoint, checkpoints_dir / "last.pt")
        if improved:
            atomic_torch_save(checkpoint, checkpoints_dir / "best.pt")

        record = {
            "epoch": epoch,
            "epoch_number": epoch + 1,
            "global_step": global_step,
            "elapsed_seconds": time.perf_counter() - epoch_start,
            "train": train_metrics,
            "validation": val_metrics,
            "best_validation_l1": best_validation_l1,
            "improved": improved,
            "early_stop_count": early_stop_count,
            "generator_learning_rate": generator_optimizer.param_groups[0]["lr"],
            "critic_learning_rate": critic_optimizer.param_groups[0]["lr"],
        }
        _append_jsonl(log_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_metrics is not None and patience > 0 and early_stop_count >= patience:
            print(f"Early stopping after {epoch + 1} epochs.", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
