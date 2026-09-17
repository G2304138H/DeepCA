"""Versioned, atomic DeepCA checkpoint helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.as_tensor(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state

def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    keys = numpy_state["keys"]
    if torch.is_tensor(keys):
        keys = keys.cpu().numpy()
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(keys, dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_torch_load(path: str | Path, device: torch.device | str) -> Mapping[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "This project requires torch.load(..., weights_only=True). "
            "Install the pinned PyTorch version or newer."
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a mapping.")
    return payload


def make_training_checkpoint(
    *,
    epoch: int,
    global_step: int,
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Any,
    best_validation_l1: float,
    config: Mapping[str, Any],
    config_fingerprint: str,
    resolved_splits: Mapping[str, Any],
) -> dict[str, Any]:
    clean_config = {key: value for key, value in config.items() if key != "_config_path"}
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_validation_l1": float(best_validation_l1),
        "config": clean_config,
        "config_fingerprint": str(config_fingerprint),
        "resolved_splits": dict(resolved_splits),
        "rng_state": capture_rng_state(),
    }


def generator_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept rich checkpoints and the upstream ``network`` key."""

    state = checkpoint.get("generator", checkpoint.get("network"))
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint contains neither 'generator' nor legacy 'network' weights.")
    return state


def critic_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state = checkpoint.get("critic", checkpoint.get("discriminator"))
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint contains neither 'critic' nor legacy 'discriminator' weights.")
    return state
