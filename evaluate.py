#!/usr/bin/env python3
"""Evaluate a trained DeepCA checkpoint on an ImageCAS split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from deepca.checkpoint import generator_state_dict, safe_torch_load
from deepca.config import config_fingerprint, load_config, save_resolved_config
from deepca.data import (
    ImageCASDataset,
    build_projection_index,
    resolve_case_pairs,
)
from deepca.metrics import (
    PhysicalGrid,
    aggregate_statistics,
    binary_cldice,
    binary_dice,
    skeletonization_metadata,
)
from deepca.modeling import architecture_metadata, build_generator
from deepca.splits import load_resolved_splits


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "valid": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Evaluation YAML configuration.")
    parser.add_argument("--split", help="train, val/validation/dev, or test.")
    parser.add_argument("--checkpoint", help="Override evaluation.checkpoint.")
    parser.add_argument("--device", help="Override evaluation.device.")
    parser.add_argument("--threshold", type=float, help="Raw-output binary threshold.")
    parser.add_argument(
        "--view-counts",
        help="Comma-separated view counts, overriding evaluation.view_counts.",
    )
    parser.add_argument(
        "--no-save-predictions", action="store_true", help="Do not save binary NPZ predictions."
    )
    parser.add_argument(
        "--allow-checkpoint-config-mismatch",
        action="store_true",
        help=(
            "Explicitly allow a legacy checkpoint with no saved config, or a rich "
            "checkpoint whose data/model contract differs from this evaluation config."
        ),
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_view_counts(value: object) -> list[int]:
    if isinstance(value, str):
        raw: Sequence[object] = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        raw = value
    else:
        raise TypeError("view_counts must be a list or comma-separated string.")
    counts = [int(item) for item in raw]
    if not counts or any(item <= 0 for item in counts) or len(set(counts)) != len(counts):
        raise ValueError("view_counts must contain unique positive integers.")
    return counts


def _grid_from_item(item: Mapping[str, Any], shape: Sequence[int]) -> PhysicalGrid:
    return PhysicalGrid(
        shape_zyx=tuple(int(value) for value in shape),
        spacing_xyz_mm=tuple(
            float(value) for value in item["grid_spacing_xyz_mm"].cpu().numpy()
        ),
        origin_xyz_mm=tuple(
            float(value) for value in item["grid_origin_xyz_mm"].cpu().numpy()
        ),
    )


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    stats = aggregate_statistics(values).as_dict()
    stats["standard_deviation"] = stats["sample_std"]
    return stats


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "dice": None, "cldice": None}
    return {
        "n": len(rows),
        "dice": _stats([float(row["dice"]) for row in rows]),
        "cldice": _stats([float(row["cldice"]) for row in rows]),
    }


def _summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_vessel: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_views: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_both: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        vessel = str(row["vessel_type"])
        views = int(row["num_views"])
        by_vessel[vessel].append(row)
        by_views[views].append(row)
        by_both[(vessel, views)].append(row)
    return {
        "overall": _aggregate(rows),
        "by_vessel_type": {
            key: _aggregate(value) for key, value in sorted(by_vessel.items())
        },
        "by_num_views": {
            str(key): _aggregate(value) for key, value in sorted(by_views.items())
        },
        "by_vessel_type_and_num_views": {
            f"{vessel}:{views}": _aggregate(value)
            for (vessel, views), value in sorted(by_both.items())
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "case_id",
        "vessel_type",
        "num_views",
        "view_indices",
        "dice",
        "cldice",
        "threshold",
        "sid_m",
        "sid_source",
        "source_to_isocentre_m",
        "detector_pixel_spacing_mm",
        "detector_pixel_spacing_source",
        "grid_spacing_xyz_mm",
        "grid_origin_xyz_mm",
        "prediction_path",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key)
                    for key in fields
                }
            )


def _save_prediction(
    path: Path,
    prediction_zyx: np.ndarray,
    grid: PhysicalGrid,
    *,
    case_id: str,
    view_indices: Sequence[int],
    threshold: float,
    checkpoint_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        vol=prediction_zyx.astype(np.uint8, copy=False),
        spacing=np.asarray(grid.spacing_xyz_mm, dtype=np.float32),
        origin=np.asarray(grid.origin_xyz_mm, dtype=np.float32),
        axis_order=np.asarray("ZYX"),
        case_id=np.asarray(case_id),
        view_indices=np.asarray(view_indices, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.float32),
        checkpoint=np.asarray(str(checkpoint_path)),
    )
    os.replace(temporary, path)


def _canonical_split(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SPLIT_ALIASES:
        raise ValueError(f"Unsupported split {value!r}.")
    return SPLIT_ALIASES[normalized]


def _evaluation_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that determine checkpoint/cohort scientific compatibility."""

    data = config.get("data")
    model = config.get("model")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise KeyError("Configuration requires 'data' and 'model' mappings.")
    return {
        "model": dict(model),
        "data": {
            key: data.get(key)
            for key in (
                "vessel_type",
                "calibration",
                "ground_truth",
                "preprocessing",
            )
        },
    }


def _split_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Checkpoint split manifest has no 'splits' mapping.")
    case_ids: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        section = splits.get(name)
        if not isinstance(section, Mapping) or not isinstance(
            section.get("case_ids"), list
        ):
            raise ValueError(
                f"Checkpoint split manifest has no case list for {name!r}."
            )
        case_ids[name] = [str(value) for value in section["case_ids"]]
    return {
        "source_sha256": manifest.get("source_sha256"),
        "expected_vessel": manifest.get("expected_vessel"),
        "case_ids": case_ids,
    }


def _write_evaluation_artifacts(
    output_dir: Path,
    *,
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_json(output_dir / "per_case.json", {"context": context, "results": rows})
    _write_csv(output_dir / "per_case.csv", rows)
    _atomic_json(
        output_dir / "summary.json",
        {"context": context, "summary": _summaries(rows)},
    )
    _atomic_json(
        output_dir / "failures.json",
        {"context": context, "failures": failures},
    )


def _terminal_failure(
    output_dir: Path,
    *,
    stage: str,
    error: Exception,
    vessel: str,
    split: str,
    config: Mapping[str, Any],
    case_ids: Sequence[str] = (),
) -> int:
    affected = list(case_ids) or [None]
    failures = [
        {
            "case_id": case_id,
            "vessel_type": vessel,
            "stage": stage,
            "error_type": type(error).__name__,
            "reason": str(error),
        }
        for case_id in affected
    ]
    context = {
        "schema_version": 1,
        "status": "setup_failed",
        "failure_stage": stage,
        "split": split,
        "vessel_type": vessel,
        "config_fingerprint": config_fingerprint(config),
        "successful_results": 0,
        "failed_results": len(failures),
    }
    _write_evaluation_artifacts(
        output_dir, context=context, rows=[], failures=failures
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), **context, "reason": str(error)},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if not isinstance(config.get("evaluation"), Mapping):
        raise KeyError("Configuration requires an 'evaluation' mapping.")
    evaluation = config["evaluation"]
    data_config = config["data"]
    vessel = str(data_config["vessel_type"]).lower()
    split = _canonical_split(args.split or evaluation.get("split", "test"))
    threshold = float(
        evaluation.get("threshold", 0.5) if args.threshold is None else args.threshold
    )
    if not math.isfinite(threshold):
        raise ValueError("Prediction threshold must be finite.")
    view_counts = _parse_view_counts(
        args.view_counts if args.view_counts is not None else evaluation.get("view_counts", [2])
    )
    output_dir = Path(evaluation["output_dir"]).expanduser().resolve() / split
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    try:
        resolved_splits = load_resolved_splits(data_config["split_json"], vessel)
        case_ids = resolved_splits.split_ids[split]
    except Exception as error:
        return _terminal_failure(
            output_dir,
            stage="split_resolution",
            error=error,
            vessel=vessel,
            split=split,
            config=config,
        )

    checkpoint_path = Path(
        args.checkpoint or evaluation.get("checkpoint") or ""
    ).expanduser().resolve()
    try:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Evaluation checkpoint does not exist: {checkpoint_path}"
            )
        device = torch.device(args.device or evaluation.get("device", "cuda:0"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA evaluation requested but torch.cuda.is_available() is False."
            )
        checkpoint = safe_torch_load(checkpoint_path, device)
        checkpoint_config = checkpoint.get("config")
        if isinstance(checkpoint_config, Mapping):
            current_contract = _evaluation_contract(config)
            saved_contract = _evaluation_contract(checkpoint_config)
            if (
                current_contract != saved_contract
                and not args.allow_checkpoint_config_mismatch
            ):
                raise ValueError(
                    "Checkpoint data/model contract differs from the evaluation "
                    "configuration. This can mix cohorts, calibration, preprocessing, "
                    "or model semantics. Use --allow-checkpoint-config-mismatch only "
                    "for a deliberate, documented cross-contract experiment."
                )
        elif not args.allow_checkpoint_config_mismatch:
            raise ValueError(
                "Checkpoint has no saved configuration and cannot be associated safely "
                "with this evaluation cohort. Legacy checkpoints require the explicit "
                "--allow-checkpoint-config-mismatch override."
            )

        saved_split_manifest = checkpoint.get("resolved_splits")
        if isinstance(saved_split_manifest, Mapping):
            if (
                _split_contract(saved_split_manifest)
                != _split_contract(resolved_splits.as_manifest())
                and not args.allow_checkpoint_config_mismatch
            ):
                raise ValueError(
                    "Checkpoint split manifest differs from the configured split JSON. "
                    "Use --allow-checkpoint-config-mismatch only if this is intentional."
                )
        elif (
            isinstance(checkpoint_config, Mapping)
            and not args.allow_checkpoint_config_mismatch
        ):
            raise ValueError("Rich checkpoint does not contain a resolved split manifest.")

        generator = build_generator(config, device).eval()
        generator.load_state_dict(generator_state_dict(checkpoint), strict=True)
        current_architecture = architecture_metadata(generator)
        saved_architecture = checkpoint.get("architecture")
        if isinstance(saved_architecture, Mapping):
            mismatches = {
                key: (saved_architecture.get(key), value)
                for key, value in current_architecture.items()
                if saved_architecture.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"Checkpoint/evaluation architecture mismatch: {mismatches}"
                )
    except Exception as error:
        return _terminal_failure(
            output_dir,
            stage="checkpoint_or_model_setup",
            error=error,
            vessel=vessel,
            split=split,
            config=config,
            case_ids=case_ids,
        )

    failures: list[dict[str, Any]] = []
    try:
        projection_index = build_projection_index(
            data_config["projection_root"], expected_vessel=vessel
        )
        pairs, pairing_failures = resolve_case_pairs(
            case_ids,
            projection_root=data_config["projection_root"],
            ground_truth_root=data_config["ground_truth_root"],
            vessel_type=vessel,
            projection_index=projection_index,
            strict=False,
        )
        failures.extend(pairing_failures)
    except Exception as error:
        failures.extend(
            {
                "case_id": case_id,
                "vessel_type": vessel,
                "stage": "projection_index",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
            for case_id in case_ids
        )
        pairs = []

    rows: list[dict[str, Any]] = []
    save_predictions = bool(evaluation.get("save_predictions", True)) and not args.no_save_predictions
    for num_views in view_counts:
        if not pairs:
            break
        try:
            dataset = ImageCASDataset(
                pairs, config, training=False, num_views_override=num_views
            )
        except Exception as error:
            failures.extend(
                {
                    "case_id": pair.case_id,
                    "vessel_type": pair.vessel_type,
                    "num_views": num_views,
                    "stage": "dataset_setup",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
                for pair in pairs
            )
            continue
        for index, pair in enumerate(pairs):
            started = time.perf_counter()
            try:
                item = dataset[index]
                condition = item["input"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                with torch.inference_mode():
                    raw_prediction = generator(condition)[0, 0].float().cpu().numpy()
                prediction = (raw_prediction >= threshold).astype(np.uint8)
                target = item["target"][0].cpu().numpy().astype(np.uint8)
                grid = _grid_from_item(item, prediction.shape)
                dice = binary_dice(
                    prediction,
                    target,
                    prediction_grid=grid,
                    target_grid=grid,
                )
                cldice = binary_cldice(
                    prediction,
                    target,
                    prediction_grid=grid,
                    target_grid=grid,
                )
                metadata = json.loads(item["metadata_json"])
                prediction_path: Optional[Path] = None
                if save_predictions:
                    prediction_path = (
                        output_dir
                        / "predictions"
                        / f"views_{num_views}"
                        / f"{pair.case_id}.npz"
                    )
                    _save_prediction(
                        prediction_path,
                        prediction,
                        grid,
                        case_id=pair.case_id,
                        view_indices=metadata["view_indices"],
                        threshold=threshold,
                        checkpoint_path=checkpoint_path,
                    )
                rows.append(
                    {
                        "case_id": pair.case_id,
                        "vessel_type": pair.vessel_type,
                        "num_views": num_views,
                        "view_indices": metadata["view_indices"],
                        "clinical_views": metadata["clinical_views"],
                        "theta_deg": metadata["theta_deg"],
                        "phi_deg": metadata["phi_deg"],
                        "dice": dice,
                        "cldice": cldice,
                        "threshold": threshold,
                        "sid_m": metadata["sid_m"],
                        "sid_source": metadata["sid_source"],
                        "source_to_isocentre_m": metadata["source_to_isocentre_m"],
                        "detector_pixel_spacing_mm": metadata["detector_pixel_spacing_mm"],
                        "detector_pixel_spacing_source": metadata[
                            "detector_pixel_spacing_source"
                        ],
                        "grid_spacing_xyz_mm": list(grid.spacing_xyz_mm),
                        "grid_origin_xyz_mm": list(grid.origin_xyz_mm),
                        "projection_path": str(pair.projection_path),
                        "ground_truth_path": str(pair.ground_truth_path),
                        "prediction_path": (
                            None if prediction_path is None else str(prediction_path)
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "case_id": pair.case_id,
                        "vessel_type": pair.vessel_type,
                        "num_views": num_views,
                        "stage": "preprocess_or_inference_or_metric",
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                )

    try:
        skeletonization = skeletonization_metadata()
    except Exception as skeleton_error:
        skeletonization = {
            "error_type": type(skeleton_error).__name__,
            "reason": str(skeleton_error),
        }

    context = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split": split,
        "split_manifest": resolved_splits.as_manifest(),
        "vessel_type": vessel,
        "view_counts": view_counts,
        "view_selection": data_config["views"].get("eval_selection", "first"),
        "threshold": threshold,
        "preprocessing": data_config["preprocessing"],
        "calibration_config": data_config["calibration"],
        "config_fingerprint": config_fingerprint(config),
        "skeletonization": skeletonization,
        "prediction_semantics": "raw generator output >= threshold",
        "metric_grid": "per-case model grid; GT resampled nearest-neighbour before inference",
        "successful_results": len(rows),
        "failed_results": len(failures),
        "checkpoint_config_mismatch_override": bool(
            args.allow_checkpoint_config_mismatch
        ),
    }
    _write_evaluation_artifacts(
        output_dir, context=context, rows=rows, failures=failures
    )
    print(json.dumps({"output_dir": str(output_dir), **context}, sort_keys=True))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
