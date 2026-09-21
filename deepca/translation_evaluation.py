"""Fixed two-view translational calibration robustness evaluation.

The ordinary DeepCA evaluator consumes stored projection masks.  This module
keeps input view 1 untouched, replaces only input view 2 with a deterministic
re-render of the projection-centred Stage-2 artery, and rebuilds the normal
DeepCA cone-support backprojection for one accurate control and nine fixed
positive-direction translations.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from .data import (
    CasePair,
    ImageCASDataset,
    load_projection_npz,
    select_view_indices,
)
from .inference import timed_inference
from .metrics import PhysicalGrid, aggregate_statistics, binary_cldice, binary_dice
from .translation import (
    load_render_source,
    render_translation,
    resolve_translation_plan,
)


CONTROL_CONDITION_ID = "accurate_control"
MODEL_STAGE_NOTE = (
    "The released DeepCA repository exposes one generator and no separate "
    "refiner. Its generator is recorded as the coarse/model result; refined "
    "metrics and refined-minus-coarse effects are explicitly unavailable."
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _condition_value(condition: Any, *names: str) -> Any:
    for name in names:
        if hasattr(condition, name):
            return getattr(condition, name)
        if isinstance(condition, Mapping) and name in condition:
            return condition[name]
    raise AttributeError(f"Translation condition has none of the fields {names!r}.")


def _condition_id(condition: Any) -> str:
    return str(_condition_value(condition, "condition_id", "name", "label"))


def _condition_vector(condition: Any) -> tuple[float, float, float]:
    raw = _condition_value(
        condition, "translation_xyz_mm", "translation_mm", "vector_xyz_mm"
    )
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(
            f"Condition {_condition_id(condition)!r} has an invalid XYZ translation."
        )
    return tuple(float(value) for value in array)


def _condition_pattern(condition: Any) -> str:
    return str(_condition_value(condition, "pattern", "family"))


def _condition_magnitude(condition: Any) -> float:
    value = float(_condition_value(condition, "magnitude_mm", "total_magnitude_mm"))
    if not math.isfinite(value):
        raise ValueError("Translation magnitude must be finite.")
    return value


def _plan_value(plan: Any, name: str, default: Any = None) -> Any:
    if hasattr(plan, name):
        return getattr(plan, name)
    if isinstance(plan, Mapping):
        return plan.get(name, default)
    return default


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


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _load_projection(pair: CasePair, config: Mapping[str, Any]):
    data = config["data"]
    calibration = data["calibration"]
    return load_projection_npz(
        pair.projection_path,
        expected_case_id=pair.case_id,
        expected_vessel=pair.vessel_type,
        fallback_sid_m=calibration.get("fallback_sid_m"),
        fallback_detector_pixel_spacing_mm=calibration.get(
            "fallback_detector_pixel_spacing_mm"
        ),
        source_to_isocentre_m=float(
            calibration.get("source_to_isocentre_m", 0.75)
        ),
        expected_detector_pixel_spacing_mm=calibration.get(
            "expected_detector_pixel_spacing_mm"
        ),
        detector_spacing_tolerance_mm=float(
            calibration.get("detector_spacing_tolerance_mm", 1.0e-4)
        ),
        default_projection_center_offset_units=str(
            calibration.get("projection_center_offset_units", "m")
        ),
    )


def _infer(
    generator: torch.nn.Module,
    item: Mapping[str, Any],
    *,
    device: torch.device,
    threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    PhysicalGrid,
    dict[str, float],
    float,
    float,
    float,
    float,
    float,
]:
    started = time.perf_counter()
    metadata = json.loads(str(item["metadata_json"]))
    backprojection_seconds = metadata.get("backprojection_seconds")
    if backprojection_seconds is None:
        raise RuntimeError(
            "Fixed-translation evaluation requires a measured backprojection, "
            "but the dataset returned no backprojection timing."
        )
    backprojection_seconds = float(backprojection_seconds)
    prediction_pipeline_started = time.perf_counter()
    model_input = item["input"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    raw_output, inference_seconds = timed_inference(
        generator, model_input, device=device
    )
    raw_prediction = raw_output[0, 0].float().cpu().numpy()
    prediction = (raw_prediction >= threshold).astype(np.uint8)
    prediction_pipeline_seconds = time.perf_counter() - prediction_pipeline_started
    reconstruction_seconds = backprojection_seconds + prediction_pipeline_seconds
    target = item["target"][0].cpu().numpy().astype(np.uint8)
    grid = _grid_from_item(item, prediction.shape)
    metrics = {
        "dice": binary_dice(
            prediction, target, prediction_grid=grid, target_grid=grid
        ),
        "cldice": binary_cldice(
            prediction, target, prediction_grid=grid, target_grid=grid
        ),
    }
    return (
        prediction,
        target,
        grid,
        metrics,
        backprojection_seconds,
        prediction_pipeline_seconds,
        reconstruction_seconds,
        inference_seconds,
        time.perf_counter() - started,
    )


def _save_prediction(
    path: Path,
    prediction: np.ndarray,
    grid: PhysicalGrid,
    *,
    case_id: str,
    condition_id: str,
    view_indices: Sequence[int],
    translation_xyz_mm: Sequence[float],
    threshold: float,
    checkpoint_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        vol=prediction.astype(np.uint8, copy=False),
        spacing=np.asarray(grid.spacing_xyz_mm, dtype=np.float32),
        origin=np.asarray(grid.origin_xyz_mm, dtype=np.float32),
        axis_order=np.asarray("ZYX"),
        case_id=np.asarray(case_id),
        condition_id=np.asarray(condition_id),
        view_indices=np.asarray(view_indices, dtype=np.int32),
        translation_xyz_mm=np.asarray(translation_xyz_mm, dtype=np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
        checkpoint=np.asarray(str(checkpoint_path)),
    )
    os.replace(temporary, path)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    result = aggregate_statistics(values).as_dict()
    result["standard_deviation"] = result["sample_std"]
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    conditions: dict[str, Any] = {}
    for condition_id, condition_rows in grouped.items():
        conditions[condition_id] = {
            "n": len(condition_rows),
            "translation_xyz_mm": condition_rows[0]["translation_xyz_mm"],
            "total_translation_mm": condition_rows[0]["total_translation_mm"],
            "coarse_model_metrics": {
                key: _stats(
                    [float(row["coarse_model_metrics"][key]) for row in condition_rows]
                )
                for key in ("dice", "cldice")
            },
            "refined_model_metrics": None,
            "refined_minus_coarse": None,
            "reconstruction_seconds": _stats(
                [float(row["reconstruction_seconds"]) for row in condition_rows]
            ),
            "inference_seconds": _stats(
                [float(row["inference_seconds"]) for row in condition_rows]
            ),
            "changes_relative_to_accurate_control": {
                key: _stats(
                    [
                        float(row["changes_relative_to_accurate_control"][key])
                        for row in condition_rows
                    ]
                )
                for key in ("dice", "cldice")
            },
            "rendering_diagnostics": {
                key: _stats(
                    [
                        float(row[key])
                        for row in condition_rows
                        if row.get(key) is not None
                    ]
                )
                if any(row.get(key) is not None for row in condition_rows)
                else None
                for key in (
                    "clean_rerender_dice_vs_stored",
                    "visible_centerline_fraction",
                    "visible_vessel_surface_fraction",
                    "stored_foreground_pixel_ratio",
                    "translated_foreground_pixel_ratio",
                    "translated_to_stored_foreground_ratio",
                )
            },
            "cases_below_visibility_warning_threshold": sum(
                bool(row.get("below_visibility_warning_threshold"))
                for row in condition_rows
            ),
        }
    return {
        "case_count": len({str(row["case_id"]) for row in rows}),
        "condition_count": len(grouped),
        "result_count": len(rows),
        "overall_reconstruction_seconds": _stats(
            [float(row["reconstruction_seconds"]) for row in rows]
        ),
        "overall_inference_seconds": _stats(
            [float(row["inference_seconds"]) for row in rows]
        ),
        "conditions": conditions,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "case_id",
        "vessel_type",
        "condition_id",
        "condition_kind",
        "translation_pattern",
        "translation_xyz_mm",
        "equivalent_system_translation_xyz_mm",
        "total_translation_mm",
        "delta_theta_deg",
        "delta_phi_deg",
        "view_indices",
        "replaced_view_indices",
        "selected_images_sha256",
        "replacement_images_sha256",
        "source_view_index",
        "clinical_views",
        "theta_deg",
        "phi_deg",
        "projected_branch_indices",
        "projected_branch_indices_source",
        "projection_center_reference_branch_indices",
        "projection_center_reference_branch_indices_source",
        "stored_projection_center_offset_xyz_mm",
        "centering_offset_used_xyz_mm",
        "recomputed_projection_center_offset_xyz_mm",
        "projection_center_offset_difference_mm",
        "render_mode",
        "render_mode_source",
        "renderer_num_circle_points",
        "renderer_num_circle_points_source",
        "sid_m",
        "sid_source",
        "source_to_isocentre_m",
        "source_to_isocentre_source",
        "detector_pixel_spacing_mm",
        "detector_pixel_spacing_source",
        "clean_rerender_dice_vs_stored",
        "clean_rerender_foreground_pixels",
        "visible_centerline_fraction",
        "visible_vessel_surface_fraction",
        "stored_foreground_pixels",
        "translated_foreground_pixels",
        "stored_foreground_pixel_ratio",
        "translated_foreground_pixel_ratio",
        "translated_to_stored_foreground_ratio",
        "coarse_dice",
        "coarse_cldice",
        "refined_dice",
        "refined_cldice",
        "refined_minus_coarse_dice",
        "refined_minus_coarse_cldice",
        "dice_change_from_accurate",
        "cldice_change_from_accurate",
        "dice_relative_change_percent",
        "cldice_relative_change_percent",
        "prediction_path",
        "backprojection_seconds",
        "prediction_pipeline_seconds",
        "reconstruction_seconds",
        "inference_seconds",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            coarse = row["coarse_model_metrics"]
            refined = row.get("refined_model_metrics") or {}
            effect = row.get("refined_minus_coarse") or {}
            change = row["changes_relative_to_accurate_control"]
            relative_change = row["relative_changes_to_accurate_control_percent"]
            flat = {
                key: (
                    json.dumps(row[key])
                    if isinstance(row.get(key), (list, tuple, dict))
                    else row.get(key)
                )
                for key in fields
            }
            flat.update(
                {
                    "coarse_dice": coarse["dice"],
                    "coarse_cldice": coarse["cldice"],
                    "refined_dice": refined.get("dice"),
                    "refined_cldice": refined.get("cldice"),
                    "refined_minus_coarse_dice": effect.get("dice"),
                    "refined_minus_coarse_cldice": effect.get("cldice"),
                    "dice_change_from_accurate": change["dice"],
                    "cldice_change_from_accurate": change["cldice"],
                    "dice_relative_change_percent": relative_change["dice"],
                    "cldice_relative_change_percent": relative_change["cldice"],
                }
            )
            writer.writerow(flat)


def _normalise_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    stored_foreground_pixels: int,
) -> dict[str, Any]:
    result = dict(_jsonable(diagnostics))
    aliases = {
        "visible_centerline_fraction": (
            "visible_centerline_fraction",
            "visible_centerline_point_fraction",
        ),
        "visible_vessel_surface_fraction": (
            "visible_vessel_surface_fraction",
            "visible_surface_point_fraction",
        ),
        "translated_foreground_pixels": (
            "translated_foreground_pixels",
            "perturbed_foreground_pixels",
        ),
        "translated_to_stored_foreground_ratio": (
            "translated_to_stored_foreground_ratio",
            "perturbed_to_stored_foreground_ratio",
        ),
    }
    for destination, candidates in aliases.items():
        if destination in result:
            continue
        for candidate in candidates:
            if candidate in result:
                result[destination] = result[candidate]
                break
    result.setdefault("stored_foreground_pixels", stored_foreground_pixels)
    translated = result.get("translated_foreground_pixels")
    if result.get("translated_to_stored_foreground_ratio") is None:
        result["translated_to_stored_foreground_ratio"] = (
            None
            if stored_foreground_pixels == 0 or translated is None
            else float(translated) / stored_foreground_pixels
        )
    return result


def run_fixed_translation_evaluation(
    *,
    config: Mapping[str, Any],
    pairs: Sequence[CasePair],
    generator: torch.nn.Module,
    device: torch.device,
    threshold: float,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_config_mismatch_override: bool,
    config_fingerprint_value: str,
    split: str,
    split_manifest: Mapping[str, Any],
    output_dir: Path,
    save_predictions: bool,
) -> dict[str, Any]:
    """Run the accurate control and all nine fixed translation conditions."""

    if not pairs:
        raise ValueError("Fixed translation evaluation requires at least one case.")
    evaluation = config["evaluation"]
    plan = resolve_translation_plan(evaluation, [2])
    conditions = tuple(_plan_value(plan, "conditions", ()))
    if len(conditions) != 9:
        raise ValueError(
            f"Fixed translation evaluation requires nine conditions, got {len(conditions)}."
        )

    runtime_config = copy.deepcopy(dict(config))
    runtime_data = runtime_config.setdefault("data", {})
    runtime_cache = runtime_data.setdefault("cache", {})
    configured_cache_enabled = bool(runtime_cache.get("enabled", True))
    runtime_cache["enabled"] = False

    control_metadata: dict[str, dict[str, Any]] = {}
    diagnostics_by_condition: dict[str, dict[str, dict[str, Any]]] = {
        CONTROL_CONDITION_ID: {}
    }
    replacements_by_condition: dict[
        str, dict[str, dict[int, np.ndarray]]
    ] = {_condition_id(condition): {} for condition in conditions}

    # Complete every renderer/reproduction check before running any translated
    # model inference. This prevents condition-dependent case sets.
    views_config = runtime_data["views"]
    for pair in pairs:
        projection = _load_projection(pair, runtime_config)
        selected_indices = select_view_indices(
            projection.num_views,
            2,
            strategy=str(
                views_config.get(
                    "eval_selection", views_config.get("selection", "first")
                )
            ),
            explicit_indices=views_config.get("indices"),
            seed=int(runtime_config.get("experiment", {}).get("seed", 1)),
            epoch=0,
            case_id=pair.case_id,
        )
        view_indices = [int(value) for value in selected_indices]
        if len(view_indices) != 2:
            raise ValueError(
                f"{pair.case_id}: fixed translation evaluation requires two views."
            )
        source = load_render_source(
            pair.projection_path,
            projection,
            int(_plan_value(plan, "renderer_num_circle_points", 120)),
            str(_plan_value(plan, "render_mode_fallback", "filled")),
            float(
                _plan_value(plan, "maximum_center_offset_difference_mm", 0.01)
            ),
            str(_plan_value(plan, "missing_projected_branch_policy", "error")),
        )
        second_source_index = view_indices[1]
        stored_second = np.asarray(
            projection.images[second_source_index], dtype=np.float32
        )
        theta = float(projection.theta_deg[second_source_index])
        phi = float(projection.phi_deg[second_source_index])

        def render_condition(
            condition_id: str,
            vector: tuple[float, float, float],
            *,
            clean_rerender: np.ndarray | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            try:
                return render_translation(
                    source,
                    stored_second,
                    theta,
                    phi,
                    vector,
                    float(
                        _plan_value(plan, "minimum_clean_rerender_dice", 0.98)
                    ),
                    float(
                        _plan_value(plan, "visibility_warning_threshold", 0.95)
                    ),
                    bool(
                        _plan_value(
                            plan, "fail_below_visibility_threshold", False
                        )
                    ),
                    clean_rerender=clean_rerender,
                )
            except Exception as error:
                raise RuntimeError(
                    f"{pair.case_id}/{condition_id}: renderer/control failure: "
                    f"{type(error).__name__}: {error}"
                ) from error

        clean_second_rerender, control_diagnostics_raw = render_condition(
            CONTROL_CONDITION_ID,
            (0.0, 0.0, 0.0),
        )
        stored_foreground = int(np.count_nonzero(stored_second > 0.5))
        detector_pixel_count = int(stored_second.size)
        control_diagnostics = _normalise_diagnostics(
            control_diagnostics_raw,
            stored_foreground_pixels=stored_foreground,
        )
        control_diagnostics["source_view_index"] = second_source_index
        control_diagnostics["translated_foreground_pixels"] = stored_foreground
        control_diagnostics["translated_to_stored_foreground_ratio"] = (
            None if stored_foreground == 0 else 1.0
        )
        control_diagnostics["stored_foreground_pixel_ratio"] = float(
            stored_foreground / detector_pixel_count
        )
        control_diagnostics["translated_foreground_pixel_ratio"] = float(
            stored_foreground / detector_pixel_count
        )
        diagnostics_by_condition[CONTROL_CONDITION_ID][pair.case_id] = (
            control_diagnostics
        )

        for condition in conditions:
            condition_id = _condition_id(condition)
            vector = _condition_vector(condition)
            replacement, raw_diagnostics = render_condition(
                condition_id,
                vector,
                clean_rerender=clean_second_rerender,
            )
            replacements_by_condition[condition_id][pair.case_id] = {
                second_source_index: np.asarray(replacement, dtype=np.float32)
            }
            condition_diagnostics = _normalise_diagnostics(
                raw_diagnostics,
                stored_foreground_pixels=stored_foreground,
            )
            condition_diagnostics["source_view_index"] = second_source_index
            condition_diagnostics["stored_foreground_pixel_ratio"] = float(
                stored_foreground / detector_pixel_count
            )
            condition_diagnostics["translated_foreground_pixel_ratio"] = float(
                condition_diagnostics["translated_foreground_pixels"]
                / detector_pixel_count
            )
            diagnostics_by_condition.setdefault(condition_id, {})[pair.case_id] = (
                condition_diagnostics
            )

    control_dataset = ImageCASDataset(
        pairs,
        runtime_config,
        training=False,
        num_views_override=2,
        measure_backprojection_time=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_run_id = f"run-{time.time_ns()}-{os.getpid()}-{checkpoint_sha256[:12]}"
    result_staging_dir = output_dir / f".{result_run_id}.staging"
    result_publish_dir = output_dir / "runs" / result_run_id
    result_staging_dir.mkdir(parents=True, exist_ok=False)

    prediction_run_id: str | None = None
    prediction_staging_dir: Path | None = None
    prediction_publish_dir: Path | None = None
    if save_predictions:
        prediction_run_id = result_run_id
        prediction_staging_dir = result_staging_dir / "predictions"
        prediction_publish_dir = result_publish_dir / "predictions"

    rows: list[dict[str, Any]] = []
    control_metrics: dict[str, dict[str, float]] = {}
    control_target_hashes: dict[str, str] = {}
    active_condition_id: str | None = None
    active_case_id: str | None = None

    def evaluate_dataset(
        *,
        condition_id: str,
        condition_kind: str,
        pattern: str,
        vector: tuple[float, float, float],
        magnitude: float,
        dataset: ImageCASDataset,
    ) -> None:
        nonlocal active_condition_id, active_case_id
        active_condition_id = condition_id
        for pair_index, pair in enumerate(pairs):
            active_case_id = pair.case_id
            item = dataset[pair_index]
            metadata = json.loads(str(item["metadata_json"]))
            expected_replaced_views = (
                []
                if condition_id == CONTROL_CONDITION_ID
                else [int(metadata["view_indices"][1])]
            )
            if metadata["replaced_view_indices"] != expected_replaced_views:
                raise ValueError(
                    f"{pair.case_id}/{condition_id}: expected replacements "
                    f"{expected_replaced_views}, got "
                    f"{metadata['replaced_view_indices']}"
                )
            if condition_id != CONTROL_CONDITION_ID:
                expected = control_metadata[pair.case_id]
                for key in (
                    "view_indices",
                    "theta_deg",
                    "phi_deg",
                    "sid_m",
                    "sid_source",
                    "source_to_isocentre_m",
                    "detector_pixel_spacing_mm",
                    "detector_pixel_spacing_source",
                    "grid",
                ):
                    if metadata[key] != expected[key]:
                        raise ValueError(
                            f"{pair.case_id}/{condition_id}: nominal {key} "
                            "changed relative to the accurate control."
                        )
            (
                prediction,
                target,
                grid,
                metrics,
                backprojection_seconds,
                prediction_pipeline_seconds,
                reconstruction_seconds,
                inference_seconds,
                elapsed,
            ) = _infer(
                generator, item, device=device, threshold=threshold
            )
            target_hash = _array_sha256(target)
            if condition_id == CONTROL_CONDITION_ID:
                control_metrics[pair.case_id] = metrics
                control_target_hashes[pair.case_id] = target_hash
                control_metadata[pair.case_id] = metadata
            elif target_hash != control_target_hashes[pair.case_id]:
                raise ValueError(
                    f"{pair.case_id}/{condition_id}: canonical target changed."
                )
            baseline = control_metrics[pair.case_id]
            change = {
                key: float(metrics[key] - baseline[key])
                for key in ("dice", "cldice")
            }
            relative_change = {
                key: (
                    None
                    if baseline[key] == 0.0
                    else float(100.0 * change[key] / baseline[key])
                )
                for key in ("dice", "cldice")
            }
            prediction_path: Optional[Path] = None
            if save_predictions:
                if prediction_staging_dir is None or prediction_publish_dir is None:
                    raise AssertionError("Prediction staging was not initialized.")
                prediction_path = (
                    prediction_publish_dir
                    / condition_id
                    / f"{pair.case_id}.npz"
                )
                _save_prediction(
                    prediction_staging_dir
                    / condition_id
                    / f"{pair.case_id}.npz",
                    prediction,
                    grid,
                    case_id=pair.case_id,
                    condition_id=condition_id,
                    view_indices=metadata["view_indices"],
                    translation_xyz_mm=vector,
                    threshold=threshold,
                    checkpoint_path=checkpoint_path,
                )
            diagnostics = diagnostics_by_condition[condition_id][pair.case_id]
            row = {
                **diagnostics,
                "case_id": pair.case_id,
                "vessel_type": pair.vessel_type,
                "condition_id": condition_id,
                "condition_kind": condition_kind,
                "translation_pattern": pattern,
                "translation_xyz_mm": list(vector),
                "equivalent_system_translation_xyz_mm": [
                    -float(value) for value in vector
                ],
                "total_translation_mm": float(magnitude),
                "delta_theta_deg": [0.0, 0.0],
                "delta_phi_deg": [0.0, 0.0],
                "view_indices": metadata["view_indices"],
                "replaced_view_indices": metadata["replaced_view_indices"],
                "selected_images_sha256": metadata["selected_images_sha256"],
                "replacement_images_sha256": metadata[
                    "replacement_images_sha256"
                ],
                "clinical_views": metadata["clinical_views"],
                "theta_deg": metadata["theta_deg"],
                "phi_deg": metadata["phi_deg"],
                "input_view_1": "stored_unchanged",
                "input_view_2": (
                    "stored_accurate"
                    if condition_id == CONTROL_CONDITION_ID
                    else "rerendered_translated_artery"
                ),
                "nominal_view_directions_changed": False,
                "canonical_target_changed": False,
                "projection_source_files_changed": False,
                "image_features_recomputed": True,
                "feature_cache_used": False,
                "target_sha256": target_hash,
                "coarse_model_metrics": metrics,
                "refined_model_metrics": None,
                "refined_minus_coarse": None,
                "refiner_status": "unavailable_in_released_deepca",
                "changes_relative_to_accurate_control": change,
                "relative_changes_to_accurate_control_percent": relative_change,
                "threshold": threshold,
                "sid_m": metadata["sid_m"],
                "sid_source": metadata["sid_source"],
                "source_to_isocentre_m": metadata["source_to_isocentre_m"],
                "source_to_isocentre_source": "configuration",
                "detector_pixel_spacing_mm": metadata[
                    "detector_pixel_spacing_mm"
                ],
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
                "backprojection_seconds": backprojection_seconds,
                "prediction_pipeline_seconds": prediction_pipeline_seconds,
                "reconstruction_seconds": reconstruction_seconds,
                "inference_seconds": inference_seconds,
                "elapsed_seconds": elapsed,
            }
            rows.append(_jsonable(row))

    try:
        evaluate_dataset(
            condition_id=CONTROL_CONDITION_ID,
            condition_kind="accurate_two_view_control",
            pattern="control",
            vector=(0.0, 0.0, 0.0),
            magnitude=0.0,
            dataset=control_dataset,
        )
        for condition in conditions:
            condition_id = _condition_id(condition)
            evaluate_dataset(
                condition_id=condition_id,
                condition_kind="translated_second_view",
                pattern=_condition_pattern(condition),
                vector=_condition_vector(condition),
                magnitude=_condition_magnitude(condition),
                dataset=ImageCASDataset(
                    pairs,
                    runtime_config,
                    training=False,
                    num_views_override=2,
                    image_replacements=replacements_by_condition[condition_id],
                    measure_backprojection_time=True,
                ),
            )
    except BaseException as error:
        shutil.rmtree(result_staging_dir, ignore_errors=True)
        location = "/".join(
            value
            for value in (active_case_id, active_condition_id)
            if value is not None
        )
        if location and isinstance(error, Exception):
            raise RuntimeError(
                f"Fixed translation evaluation failed at {location}: "
                f"{type(error).__name__}: {error}"
            ) from error
        raise

    expected_rows = len(pairs) * 10
    if len(rows) != expected_rows:
        shutil.rmtree(result_staging_dir, ignore_errors=True)
        raise RuntimeError(
            f"Incomplete fixed translation matrix: expected {expected_rows} rows, "
            f"got {len(rows)}."
        )
    context = {
        "schema_version": 1,
        "status": "complete",
        "evaluation_mode": "fixed_two_view_translation",
        "description": (
            "fixed two-view translational calibration robustness evaluation"
        ),
        "stress_test_scope": "fixed-positive-direction translational stress test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_config_mismatch_override": bool(
            checkpoint_config_mismatch_override
        ),
        "config_fingerprint": config_fingerprint_value,
        "split": split,
        "split_manifest": _jsonable(split_manifest),
        "case_ids_in_order": [pair.case_id for pair in pairs],
        "case_count": len(pairs),
        "num_input_views": 2,
        "accurate_control_count": 1,
        "perturbed_condition_count": 9,
        "condition_order": [
            CONTROL_CONDITION_ID, *[_condition_id(item) for item in conditions]
        ],
        "plan": _jsonable(plan),
        "angle_perturbation": {"delta_theta_deg": 0.0, "delta_phi_deg": 0.0},
        "coordinate_convention": {
            "+x": "patient left",
            "+y": "patient anterior, away from the table",
            "+z": "patient superior, toward the head",
        },
        "translation_convention": (
            "The recorded vector translates the projection-centred artery; the "
            "equivalent source-detector system or isocentre translation is its "
            "negative."
        ),
        "configured_cache_enabled": configured_cache_enabled,
        "effective_feature_cache_enabled": False,
        "result_run_id": result_run_id,
        "result_directory": str(result_publish_dir),
        "prediction_run_id": prediction_run_id,
        "prediction_directory": (
            None if prediction_publish_dir is None else str(prediction_publish_dir)
        ),
        "feature_recomputation": (
            "DeepCA has no 2D feature encoder; its derived 3D cone-support "
            "backprojection is recomputed from each resulting image pair."
        ),
        "model_stage_note": MODEL_STAGE_NOTE,
        "metrics": ["dice", "cldice"],
        "reconstruction_timing": {
            "field": "reconstruction_seconds",
            "unit": "seconds",
            "scope": (
                "2D projection thresholding/backprojection through final binary "
                "3D prediction"
            ),
            "clock": "time.perf_counter",
            "cuda_synchronized_generator": device.type == "cuda",
            "components": [
                "backprojection_seconds",
                "prediction_pipeline_seconds",
            ],
            "includes": [
                "2D projection thresholding and cone backprojection",
                "host-to-device model-input transfer",
                "generator forward pass",
                "device-to-host model-output transfer",
                "final 3D prediction thresholding",
            ],
            "excludes": [
                "projection NPZ loading and view selection",
                "translation-condition rendering",
                "ground-truth loading and resampling",
                "metric computation",
                "prediction serialization",
                "visualization",
            ],
            "preprocessing_cache_bypassed": True,
        },
        "inference_timing": {
            "field": "inference_seconds",
            "unit": "seconds",
            "scope": "generator forward pass only",
            "clock": "time.perf_counter",
            "cuda_synchronized": device.type == "cuda",
            "excludes": [
                "data loading and preprocessing",
                "host-to-device input transfer",
                "device-to-host output transfer",
                "thresholding and metric computation",
                "prediction serialization",
            ],
        },
        "successful_results": len(rows),
        "failed_results": 0,
    }
    summary = _summary(rows)
    try:
        _atomic_json(
            result_staging_dir / "per_case.json",
            {"context": context, "results": rows},
        )
        _write_csv(result_staging_dir / "per_case.csv", rows)
        _atomic_json(
            result_staging_dir / "summary.json",
            {"context": context, "summary": summary},
        )
        _atomic_json(
            result_staging_dir / "failures.json",
            {"context": context, "failures": []},
        )
        result_publish_dir.parent.mkdir(parents=True, exist_ok=True)
        if result_publish_dir.exists():
            raise FileExistsError(
                f"Refusing to replace completed result run {result_publish_dir}"
            )
        os.replace(result_staging_dir, result_publish_dir)
    except BaseException:
        shutil.rmtree(result_staging_dir, ignore_errors=True)
        raise
    return context


__all__ = [
    "CONTROL_CONDITION_ID",
    "MODEL_STAGE_NOTE",
    "run_fixed_translation_evaluation",
]
