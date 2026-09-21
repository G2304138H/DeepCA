"""Shared model-inference helpers for evaluation."""

from __future__ import annotations

import time

import torch


def timed_inference(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Run one forward pass and return CUDA-synchronized wall-clock seconds.

    The caller moves the input to ``device`` before this function, so the
    measurement covers only the model forward pass. CPU transfer,
    post-processing, metric computation, and output serialization are excluded.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(model_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Expected model inference to return a tensor, got {type(output).__name__}."
        )
    return output, float(elapsed_seconds)


__all__ = ["timed_inference"]
