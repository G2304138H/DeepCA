#!/usr/bin/env python3
"""Run one ImageCAS data-loader batch and one lightweight DeepCA forward pass."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deepca.config import load_config
from deepca.data import CasePair, ImageCASDataset, format_case_id, inspect_projection_identity
from deepca.modeling import build_generator


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--projection", required=True, help="One Stage-2 projection NPZ.")
    parser.add_argument("--ground-truth", required=True, help="Its matching GT NPZ.")
    parser.add_argument("--volume-size", type=int, default=32)
    parser.add_argument("--base-filters", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = copy.deepcopy(load_config(args.config))
    vessel, number = inspect_projection_identity(
        args.projection, expected_vessel=config["data"]["vessel_type"]
    )
    case_id = format_case_id(vessel, number)
    pair = CasePair(
        case_id=case_id,
        vessel_type=vessel,
        case_number=number,
        projection_path=Path(args.projection).expanduser().resolve(),
        ground_truth_path=Path(args.ground_truth).expanduser().resolve(),
    )
    config["model"]["volume_size"] = args.volume_size
    config["model"]["generator"]["base_filters"] = args.base_filters
    config["data"]["preprocessing"]["volume_size"] = args.volume_size
    config["data"]["cache"]["enabled"] = False
    with tempfile.TemporaryDirectory(prefix="deepca-smoke-") as directory:
        config["data"]["cache"]["directory"] = directory
        dataset = ImageCASDataset([pair], config, training=False)
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke test requested but CUDA is unavailable.")
        model = build_generator(config, device).eval()
        with torch.inference_mode():
            output = model(batch["input"].to(device=device, dtype=torch.float32))
    result = {
        "case_id": case_id,
        "input_shape": list(batch["input"].shape),
        "target_shape": list(batch["target"].shape),
        "output_shape": list(output.shape),
        "input_range": [float(batch["input"].min()), float(batch["input"].max())],
        "target_foreground_voxels": int(torch.count_nonzero(batch["target"])),
        "output_finite": bool(torch.isfinite(output).all()),
        "view_indices": batch["view_indices"][0].tolist(),
        "device": str(device),
    }
    if result["output_shape"] != result["target_shape"] or not result["output_finite"]:
        raise RuntimeError(f"Smoke test failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
