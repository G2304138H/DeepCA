#!/usr/bin/env python3
"""Create an AutoCar-style visualization bundle from a saved DeepCA volume."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deepca.data import load_ground_truth_npz
from deepca.geometry import GridSpec, resample_binary_xyz_to_grid
from deepca.visualization import load_prediction_volume, save_prediction_visualization


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction", help="Saved DeepCA prediction NPZ.")
    source.add_argument(
        "--evaluation-dir",
        help="Evaluation split directory containing per_case.json.",
    )
    parser.add_argument(
        "--case-id",
        help="Case to select from --evaluation-dir (for example, lca_0001).",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        help="View count used to disambiguate an evaluation row.",
    )
    parser.add_argument(
        "--condition-id",
        help="Condition used to disambiguate a fixed-translation evaluation row.",
    )
    parser.add_argument("--ground-truth", help="Optional ground-truth volume NPZ override.")
    parser.add_argument("--projection", help="Optional Stage-2 projection NPZ override.")
    parser.add_argument("--output-dir", help="Destination directory for the bundle.")
    parser.add_argument("--threshold", type=float, help="Override the saved threshold.")
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--max-elements", type=int, default=20_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory only after the new bundle succeeds.",
    )
    return parser.parse_args(argv)


def _read_evaluation_row(
    evaluation_dir: Path,
    *,
    case_id: Optional[str],
    num_views: Optional[int],
    condition_id: Optional[str],
) -> dict[str, Any]:
    if not case_id:
        raise ValueError("--case-id is required with --evaluation-dir.")
    manifest_path = evaluation_dir / "per_case.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"{manifest_path} must contain a top-level results list.")
    matches: list[dict[str, Any]] = []
    for candidate in payload["results"]:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("case_id", "")).lower() != case_id.lower():
            continue
        if num_views is not None and candidate.get("num_views") != num_views:
            continue
        if condition_id is not None and candidate.get("condition_id") != condition_id:
            continue
        matches.append(candidate)
    if len(matches) != 1:
        qualifiers = [
            {
                "num_views": row.get("num_views"),
                "condition_id": row.get("condition_id"),
                "prediction_path": row.get("prediction_path"),
            }
            for row in matches
        ]
        raise ValueError(
            f"Expected one evaluation row for {case_id!r}, found {len(matches)}. "
            f"Use --num-views or --condition-id to disambiguate. Matches: {qualifiers}"
        )
    row = matches[0]
    if not row.get("prediction_path"):
        raise ValueError(
            "The selected evaluation row has no saved prediction. Re-run evaluate.py "
            "without --no-save-predictions."
        )
    return row


def _path_from_row(value: Any, evaluation_dir: Path) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (evaluation_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _align_ground_truth(path: Path, prediction: Any) -> np.ndarray:
    ground_truth = load_ground_truth_npz(path)
    spacing = np.asarray(prediction.grid.spacing_xyz_mm, dtype=np.float64)
    origin = np.asarray(prediction.grid.origin_xyz_mm, dtype=np.float64)
    shape_xyz = np.asarray(prediction.grid.shape_zyx[::-1], dtype=np.int64)
    center = origin + 0.5 * (shape_xyz - 1) * spacing
    grid = GridSpec(
        shape_zyx=prediction.grid.shape_zyx,
        spacing_xyz_mm=prediction.grid.spacing_xyz_mm,
        center_xyz_mm=tuple(float(value) for value in center),
        origin_xyz_mm=prediction.grid.origin_xyz_mm,
    )
    return resample_binary_xyz_to_grid(
        ground_truth.volume_xyz,
        ground_truth.spacing_xyz_mm,
        grid,
    ).astype(bool)


def _load_projection_views(path: Path, view_indices: tuple[int, ...]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "images" not in archive.files:
            raise KeyError(f"{path} is missing the required 'images' array.")
        images = np.asarray(archive["images"])
    if images.dtype.hasobject or images.ndim != 3 or not len(images):
        raise ValueError(f"{path}:images must be a non-empty numeric [V,H,W] array.")
    if not np.issubdtype(images.dtype, np.number) or not np.isfinite(images).all():
        raise ValueError(f"{path}:images must contain finite numeric values.")
    indices = view_indices or tuple(range(len(images)))
    if any(index < 0 or index >= len(images) for index in indices):
        raise IndexError(
            f"Saved view indices {list(indices)} exceed {path}'s {len(images)} views."
        )
    return np.ascontiguousarray(images[np.asarray(indices, dtype=np.int64)])


def _remove_backup(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Pass --overwrite to replace it."
        )
    backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.backup")
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    _remove_backup(backup)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    row: dict[str, Any] = {}
    evaluation_dir: Optional[Path] = None
    if args.evaluation_dir:
        evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
        row = _read_evaluation_row(
            evaluation_dir,
            case_id=args.case_id,
            num_views=args.num_views,
            condition_id=args.condition_id,
        )
        prediction_path = _path_from_row(row["prediction_path"], evaluation_dir)
        assert prediction_path is not None
    else:
        prediction_path = Path(args.prediction).expanduser().resolve()
        if args.case_id or args.num_views is not None or args.condition_id:
            raise ValueError(
                "--case-id, --num-views, and --condition-id apply only to "
                "--evaluation-dir."
            )

    prediction = load_prediction_volume(prediction_path, threshold=args.threshold)
    ground_truth_path = (
        Path(args.ground_truth).expanduser().resolve()
        if args.ground_truth
        else _path_from_row(row.get("ground_truth_path"), evaluation_dir)
        if evaluation_dir is not None
        else None
    )
    projection_path = (
        Path(args.projection).expanduser().resolve()
        if args.projection
        else _path_from_row(row.get("projection_path"), evaluation_dir)
        if evaluation_dir is not None
        else None
    )
    target = (
        _align_ground_truth(ground_truth_path, prediction)
        if ground_truth_path is not None
        else None
    )
    projections = (
        _load_projection_views(projection_path, prediction.view_indices)
        if projection_path is not None
        else None
    )
    if args.output_dir:
        destination = Path(args.output_dir).expanduser().resolve()
    elif evaluation_dir is not None:
        qualifier = (
            str(row.get("condition_id"))
            if row.get("condition_id") is not None
            else f"views_{row.get('num_views', len(prediction.view_indices))}"
        )
        destination = evaluation_dir / "visualizations" / qualifier / prediction.case_id
    else:
        destination = prediction_path.parent / "visualizations" / prediction.case_id
    if destination.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Pass --overwrite to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
    try:
        manifest = save_prediction_visualization(
            staging,
            prediction,
            target_zyx=target,
            projection_images=projections,
            ground_truth_path=ground_truth_path,
            projection_path=projection_path,
            gif_frames=args.gif_frames,
            gif_fps=args.gif_fps,
            maximum_elements=args.max_elements,
        )
        _publish(staging, destination, overwrite=args.overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result = {
        "output_dir": str(destination),
        "case_id": prediction.case_id,
        "foreground_voxels": manifest["foreground_voxels"],
        "graph_nodes": manifest["graph_nodes"],
        "graph_edges": manifest["graph_edges"],
        "surface_vertices": manifest["surface_vertices"],
        "surface_faces": manifest["surface_faces"],
        "manifest": str(destination / "manifest.json"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
