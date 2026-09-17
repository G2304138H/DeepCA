"""Fixed two-view translational calibration robustness support.

This module deliberately owns the complete Stage-2 vessel-surface and camera
renderer needed by the evaluation.  It does not import the sibling data
generator at runtime, so an evaluation remains reproducible from this
repository alone.

The perturbation convention is fixed: input view 1 remains the stored accurate
image and input view 2 is re-rendered after translating the projection-centred
artery.  Nominal camera angles never change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from skimage import draw, filters, morphology

if TYPE_CHECKING:
    from .data import ProjectionCase


FIXED_TRANSLATION_MODE = "fixed_two_view_translation"
FIXED_TRANSLATION_DESCRIPTION = (
    "fixed two-view translational calibration robustness evaluation"
)
TRANSLATION_DIRECTION_DESCRIPTION = (
    "fixed-positive-direction translational stress test"
)
TRANSLATION_SIGN_CONVENTION = (
    "The recorded vector is added to the projection-centred artery. Moving the "
    "artery by +delta_t is geometrically equivalent to moving the "
    "source-detector system or isocentre by -delta_t."
)
MINIMUM_REQUIRED_CLEAN_RERENDER_DICE = 0.98
MAXIMUM_ALLOWED_CENTER_OFFSET_DIFFERENCE_MM = 0.01
PATIENT_COORDINATE_CONVENTION: Mapping[str, str] = {
    "+x": "patient left",
    "+y": "patient anterior, away from the table",
    "+z": "patient superior, toward the head",
}


def _finite_number(raw: Any, *, label: str) -> float:
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number, not boolean")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _unit_interval(raw: Any, *, label: str) -> float:
    value = _finite_number(raw, label=label)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return value


def _integer_at_least(raw: Any, *, label: str, minimum: int) -> int:
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    try:
        value = int(raw)
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer >= {minimum}") from error
    if not math.isfinite(numeric) or numeric != float(value) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _translation_vector(pattern: str, magnitude_mm: float) -> tuple[float, float, float]:
    magnitude = float(magnitude_mm)
    if pattern == "Y":
        return (0.0, magnitude, 0.0)
    if pattern == "XZ":
        component = magnitude / math.sqrt(2.0)
        return (component, 0.0, component)
    if pattern == "XYZ":
        component = magnitude / math.sqrt(3.0)
        return (component, component, component)
    raise ValueError(f"Unsupported fixed translation pattern {pattern!r}")


@dataclass(frozen=True)
class TranslationCondition:
    """One immutable fixed-positive-direction translation condition."""

    condition_id: str
    pattern: str
    magnitude_mm: float
    translation_xyz_mm: tuple[float, float, float]
    delta_theta_deg: float = 0.0
    delta_phi_deg: float = 0.0

    def __post_init__(self) -> None:
        vector = tuple(float(value) for value in self.translation_xyz_mm)
        if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
            raise ValueError("translation_xyz_mm must contain three finite values")
        if not math.isclose(
            math.sqrt(sum(value * value for value in vector)),
            float(self.magnitude_mm),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{self.condition_id}: translation norm does not equal magnitude_mm"
            )
        if float(self.delta_theta_deg) != 0.0 or float(self.delta_phi_deg) != 0.0:
            raise ValueError("Fixed translation conditions must have zero angle deltas")
        object.__setattr__(self, "translation_xyz_mm", vector)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "pattern": self.pattern,
            "magnitude_mm": float(self.magnitude_mm),
            "translation_xyz_mm": list(self.translation_xyz_mm),
            "delta_theta_deg": 0.0,
            "delta_phi_deg": 0.0,
        }


def _make_default_conditions() -> tuple[TranslationCondition, ...]:
    return tuple(
        TranslationCondition(
            condition_id=f"{pattern}-{int(magnitude)}",
            pattern=pattern,
            magnitude_mm=magnitude,
            translation_xyz_mm=_translation_vector(pattern, magnitude),
        )
        for pattern in ("Y", "XZ", "XYZ")
        for magnitude in (5.0, 10.0, 20.0)
    )


DEFAULT_TRANSLATION_CONDITIONS = _make_default_conditions()


@dataclass(frozen=True)
class TranslationPlan:
    """Validated settings for the fixed nine-condition evaluation."""

    conditions: tuple[TranslationCondition, ...]
    perturbed_input_position: int
    renderer_num_circle_points: int
    minimum_clean_rerender_dice: float
    maximum_center_offset_difference_mm: float
    visibility_warning_threshold: float
    fail_below_visibility_threshold: bool
    render_mode_fallback: str
    missing_projected_branch_policy: str

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "perturbed_input_position": int(self.perturbed_input_position),
            "renderer_num_circle_points": int(self.renderer_num_circle_points),
            "minimum_clean_rerender_dice": float(
                self.minimum_clean_rerender_dice
            ),
            "maximum_center_offset_difference_mm": float(
                self.maximum_center_offset_difference_mm
            ),
            "visibility_warning_threshold": float(
                self.visibility_warning_threshold
            ),
            "fail_below_visibility_threshold": bool(
                self.fail_below_visibility_threshold
            ),
            "render_mode_fallback": self.render_mode_fallback,
            "missing_projected_branch_policy": (
                self.missing_projected_branch_policy
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": FIXED_TRANSLATION_MODE,
            "description": FIXED_TRANSLATION_DESCRIPTION,
            "direction_description": TRANSLATION_DIRECTION_DESCRIPTION,
            "settings": self.settings,
            "conditions": [condition.as_dict() for condition in self.conditions],
            "sign_convention": TRANSLATION_SIGN_CONVENTION,
            "coordinate_convention": dict(PATIENT_COORDINATE_CONVENTION),
        }


def is_fixed_translation_mode(raw: Any) -> bool:
    """Return whether *raw* names the fixed two-view translation mode."""

    if raw is None:
        return False
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        FIXED_TRANSLATION_MODE,
        "fixed_two_view_translational_calibration",
        "fixed_positive_direction_translation",
        "second_view_translation_robustness",
    }


def _validate_two_view_counts(view_counts: Any) -> None:
    if isinstance(view_counts, (bool, np.bool_)):
        raise ValueError(
            "Fixed translation evaluation requires view_counts to be exactly [2]"
        )
    if isinstance(view_counts, (int, np.integer)):
        values: Sequence[Any] = [view_counts]
    elif isinstance(view_counts, Sequence) and not isinstance(
        view_counts, (str, bytes)
    ):
        values = view_counts
    else:
        raise ValueError(
            "Fixed translation evaluation requires view_counts to be exactly [2]"
        )
    if len(values) != 1:
        raise ValueError(
            "Fixed translation evaluation requires view_counts to be exactly [2]"
        )
    value = values[0]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "Fixed translation evaluation requires view_counts to be exactly [2]"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Fixed translation evaluation requires view_counts to be exactly [2]"
        ) from error
    if not math.isfinite(numeric) or numeric != 2.0:
        raise ValueError(
            "Fixed translation evaluation requires view_counts to be exactly [2]"
        )


def resolve_translation_plan(
    evaluation: Mapping[str, Any], view_counts: Any
) -> TranslationPlan:
    """Validate and return the immutable fixed translation plan.

    ``evaluation`` is the top-level evaluation mapping.  Scientific condition
    vectors cannot be overridden by configuration; only renderer/control-gate
    settings are configurable.
    """

    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping")
    if "mode" in evaluation and not is_fixed_translation_mode(evaluation["mode"]):
        raise ValueError(
            f"evaluation.mode must be {FIXED_TRANSLATION_MODE!r} for this plan"
        )
    _validate_two_view_counts(view_counts)

    raw = evaluation.get("fixed_two_view_translation", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation.fixed_two_view_translation must be a mapping")
    forbidden = sorted(
        key
        for key in (
            "conditions",
            "magnitudes_mm",
            "patterns",
            "translation_xyz_mm",
            "delta_theta_deg",
            "delta_phi_deg",
        )
        if key in raw
    )
    if forbidden:
        raise ValueError(
            "The fixed nine translation conditions cannot be overridden; remove "
            + ", ".join(forbidden)
        )

    position = _integer_at_least(
        raw.get("perturbed_input_position", 1),
        label="evaluation.fixed_two_view_translation.perturbed_input_position",
        minimum=0,
    )
    if position != 1:
        raise ValueError(
            "evaluation.fixed_two_view_translation.perturbed_input_position must "
            "be 1: input view 1 remains accurate and only input view 2 is perturbed"
        )
    points = _integer_at_least(
        raw.get("renderer_num_circle_points", 120),
        label="evaluation.fixed_two_view_translation.renderer_num_circle_points",
        minimum=3,
    )

    clean_key = "minimum_clean_rerender_dice"
    legacy_clean_key = "minimum_clean_dice"
    if clean_key in raw and legacy_clean_key in raw:
        first = _finite_number(raw[clean_key], label=clean_key)
        second = _finite_number(raw[legacy_clean_key], label=legacy_clean_key)
        if first != second:
            raise ValueError(
                "minimum_clean_rerender_dice and minimum_clean_dice conflict"
            )
    clean_dice = _unit_interval(
        raw.get(clean_key, raw.get(legacy_clean_key, 0.98)),
        label=(
            "evaluation.fixed_two_view_translation."
            "minimum_clean_rerender_dice"
        ),
    )
    if clean_dice < MINIMUM_REQUIRED_CLEAN_RERENDER_DICE:
        raise ValueError(
            "minimum_clean_rerender_dice cannot be below the fixed contractual "
            f"floor of {MINIMUM_REQUIRED_CLEAN_RERENDER_DICE:.2f}"
        )
    maximum_center_difference = _finite_number(
        raw.get("maximum_center_offset_difference_mm", 0.01),
        label=(
            "evaluation.fixed_two_view_translation."
            "maximum_center_offset_difference_mm"
        ),
    )
    if not (
        0.0
        <= maximum_center_difference
        <= MAXIMUM_ALLOWED_CENTER_OFFSET_DIFFERENCE_MM
    ):
        raise ValueError(
            "maximum_center_offset_difference_mm must be in [0, "
            f"{MAXIMUM_ALLOWED_CENTER_OFFSET_DIFFERENCE_MM:g}]"
        )
    visibility = _unit_interval(
        raw.get("visibility_warning_threshold", 0.95),
        label=(
            "evaluation.fixed_two_view_translation."
            "visibility_warning_threshold"
        ),
    )
    fail_visibility = raw.get("fail_below_visibility_threshold", False)
    if not isinstance(fail_visibility, (bool, np.bool_)):
        raise ValueError(
            "evaluation.fixed_two_view_translation."
            "fail_below_visibility_threshold must be boolean"
        )
    fallback = str(raw.get("render_mode_fallback", "filled")).strip().lower()
    if fallback not in {"filled", "point"}:
        raise ValueError("render_mode_fallback must be 'filled' or 'point'")
    missing_branch_policy = str(
        raw.get("missing_projected_branch_policy", "error")
    ).strip().lower()
    if missing_branch_policy not in {"error", "all"}:
        raise ValueError(
            "missing_projected_branch_policy must be 'error' or 'all'"
        )

    return TranslationPlan(
        conditions=DEFAULT_TRANSLATION_CONDITIONS,
        perturbed_input_position=1,
        renderer_num_circle_points=points,
        minimum_clean_rerender_dice=clean_dice,
        maximum_center_offset_difference_mm=maximum_center_difference,
        visibility_warning_threshold=visibility,
        fail_below_visibility_threshold=bool(fail_visibility),
        render_mode_fallback=fallback,
        missing_projected_branch_policy=missing_branch_policy,
    )


def _readonly(array: np.ndarray, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Stage2RenderSource:
    """Projection-centred Stage-2 geometry and per-case calibration."""

    path: Path
    artery_shape: tuple[int, int, int]
    surface_rings: tuple[np.ndarray, ...]
    centered_centerline_xyz_m: np.ndarray
    projected_branch_indices: tuple[int, ...]
    projected_branch_indices_source: str
    projection_center_reference_branch_indices: tuple[int, ...] | None
    projection_center_reference_branch_indices_source: str
    stored_projection_center_offset_xyz_mm: tuple[float, float, float]
    centering_offset_used_xyz_mm: tuple[float, float, float]
    projection_center_offset_source: str
    recomputed_projection_center_offset_xyz_mm: tuple[float, float, float]
    projection_center_offset_difference_mm: float
    maximum_center_offset_difference_mm: float
    image_shape_hw: tuple[int, int]
    sid_m: float
    sid_source: str
    source_to_isocentre_m: float
    source_to_isocentre_source: str
    detector_pixel_spacing_mm: float
    detector_pixel_spacing_source: str
    render_mode: str
    render_mode_source: str
    renderer_num_circle_points: int
    renderer_num_circle_points_source: str
    projection_theta_deg: np.ndarray
    projection_phi_deg: np.ndarray
    projection_images: np.ndarray


def _npz_scalar(archive: np.lib.npyio.NpzFile, key: str) -> Any:
    value = np.asarray(archive[key])
    if value.dtype.hasobject:
        raise TypeError(f"{key!r} uses unsafe object dtype")
    if value.size != 1:
        raise ValueError(f"{key!r} must be scalar, got shape {value.shape}")
    return value.reshape(()).item()


def _integer_indices(
    raw: np.ndarray, *, label: str, upper_bound: int, allow_empty: bool = False
) -> tuple[int, ...]:
    values = np.asarray(raw)
    if values.dtype.hasobject:
        raise TypeError(f"{label} cannot use object dtype")
    values = values.reshape(-1)
    if values.size == 0 and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    try:
        numeric = values.astype(np.float64)
        indices = values.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must contain integer indices") from error
    if (
        not np.isfinite(numeric).all()
        or not np.array_equal(numeric, indices.astype(np.float64))
        or np.any(indices < 0)
        or np.any(indices >= int(upper_bound))
        or len(set(indices.tolist())) != int(indices.size)
    ):
        raise ValueError(
            f"{label} must contain unique indices in [0, {upper_bound}), got "
            f"{values.tolist()}"
        )
    return tuple(int(value) for value in indices.tolist())


def _estimate_derivatives(centerline: np.ndarray) -> np.ndarray:
    derivatives = np.zeros_like(centerline)
    if centerline.shape[0] < 2:
        return derivatives
    derivatives[0] = centerline[1] - centerline[0]
    derivatives[-1] = centerline[-1] - centerline[-2]
    if centerline.shape[0] > 2:
        derivatives[1:-1] = 0.5 * (centerline[2:] - centerline[:-2])
    return derivatives


def _legacy_tube_surface(
    centerline: np.ndarray, radius: np.ndarray, num_circle_points: int
) -> np.ndarray:
    """The Stage-2 generator's ordinary ``get_vessel_surface`` path."""

    curve = np.asarray(centerline)
    derivatives = _estimate_derivatives(curve)
    radii = np.asarray(radius).reshape(-1)
    keep = np.flatnonzero(np.sum(np.abs(derivatives), axis=1) != 0)
    derivatives = derivatives[keep]
    curve = curve[keep]
    if curve.shape[0] < 2:
        raise ValueError("Legacy tube frame needs at least two non-degenerate points")

    normal = np.zeros(3, dtype=curve.dtype)
    normal[int(np.argmin(np.abs(curve[1, :]))) ] = 1
    angles = np.linspace(0.0, 2.0 * np.pi, int(num_circle_points))
    cos_factor = np.tile(np.cos(angles), (3, 1))
    sin_factor = np.tile(np.sin(angles), (3, 1))
    rings: list[np.ndarray] = []
    for index in range(curve.shape[0]):
        conormal = np.cross(normal, derivatives[index, :])
        conormal = conormal / np.linalg.norm(conormal)
        normal = np.cross(derivatives[index, :], conormal)
        normal = normal / np.linalg.norm(normal)
        if index == 0:
            ring_radii = np.linspace(0.0, radii[index], 50)[1:]
        elif index == curve.shape[0] - 1:
            ring_radii = np.flip(np.linspace(0.0, radii[index], 50)[1:])
        else:
            ring_radii = [radii[index]]
        for ring_radius in ring_radii:
            points = np.tile(curve[index, :], (num_circle_points, 1)) + np.multiply(
                cos_factor.T,
                np.tile(ring_radius * normal, (num_circle_points, 1)),
            ) + np.multiply(
                sin_factor.T,
                np.tile(ring_radius * conormal, (num_circle_points, 1)),
            )
            rings.append(points)
    surface = np.asarray(rings)
    if surface.ndim != 3 or surface.shape[-1] != 3 or not np.isfinite(surface).all():
        raise ValueError("Legacy Stage-2 tube surface is non-finite")
    return surface.astype(np.float32)


def _robust_tube_surface(
    centerline: np.ndarray, radius: np.ndarray, num_circle_points: int
) -> np.ndarray:
    """The Stage-2 generator's deterministic fallback moving-frame tube."""

    centerline = np.asarray(centerline, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or centerline.shape[0] < 2:
        raise ValueError(
            f"Tube centerline must have shape [N>=2,3], got {centerline.shape}"
        )
    if radius.shape[0] != centerline.shape[0]:
        raise ValueError("Tube radius length does not match centerline length")
    if not np.isfinite(centerline).all() or not np.isfinite(radius).all():
        raise ValueError("Tube inputs contain NaN or infinity")

    derivatives = _estimate_derivatives(centerline)
    derivative_norm = np.linalg.norm(derivatives, axis=1)
    usable = np.flatnonzero(derivative_norm > 1.0e-12)
    if usable.size == 0:
        raise ValueError("Cannot build a tube with no non-zero tangent")
    for index in np.flatnonzero(derivative_norm <= 1.0e-12):
        nearest = usable[np.argmin(np.abs(usable - index))]
        derivatives[index] = derivatives[nearest]
        derivative_norm[index] = derivative_norm[nearest]
    tangents = derivatives / derivative_norm[:, None]

    angles = np.linspace(0.0, 2.0 * np.pi, int(num_circle_points))
    cos_angles = np.cos(angles)[:, None]
    sin_angles = np.sin(angles)[:, None]
    surfaces: list[np.ndarray] = []
    previous_normal: np.ndarray | None = None
    coordinate_axes = np.eye(3, dtype=np.float64)
    for index, tangent in enumerate(tangents):
        normal = None
        if previous_normal is not None:
            transported = previous_normal - np.dot(previous_normal, tangent) * tangent
            norm = float(np.linalg.norm(transported))
            if norm > 1.0e-12:
                normal = transported / norm
        if normal is None:
            for axis_index in np.argsort(np.abs(coordinate_axes @ tangent)):
                candidate = coordinate_axes[int(axis_index)]
                candidate = candidate - np.dot(candidate, tangent) * tangent
                norm = float(np.linalg.norm(candidate))
                if norm > 1.0e-12:
                    normal = candidate / norm
                    break
        if normal is None:
            raise ValueError(f"Could not construct tube frame at point {index}")
        conormal = np.cross(normal, tangent)
        conormal_norm = float(np.linalg.norm(conormal))
        if conormal_norm <= 1.0e-12:
            raise ValueError(f"Degenerate tube conormal at point {index}")
        conormal = conormal / conormal_norm
        normal = np.cross(tangent, conormal)
        normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
        previous_normal = normal

        if index == 0:
            ring_radii = np.linspace(0.0, radius[index], 50)[1:]
        elif index == centerline.shape[0] - 1:
            ring_radii = np.flip(np.linspace(0.0, radius[index], 50)[1:])
        else:
            ring_radii = np.asarray([radius[index]])
        for ring_radius in ring_radii:
            surfaces.append(
                centerline[index][None, :]
                + float(ring_radius) * cos_angles * normal[None, :]
                + float(ring_radius) * sin_angles * conormal[None, :]
            )
    surface = np.stack(surfaces, axis=0)
    if not np.isfinite(surface).all():
        raise ValueError("Fallback tube surface contains NaN or infinity")
    return surface.astype(np.float32)


def _artery_to_surface_rings(
    artery: np.ndarray,
    *,
    num_circle_points: int,
    center_reference_branch_indices: tuple[int, ...] | None,
    centering_offset_m: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    surfaces: list[np.ndarray] = []
    surface_by_index: dict[int, np.ndarray] = {}
    for branch_index in range(artery.shape[0]):
        branch = artery[branch_index]
        valid = np.logical_and(
            np.any(np.abs(branch[:, :3]) > 0.0, axis=1), branch[:, 3] > 0.0
        )
        if not np.any(valid):
            continue
        centerline = branch[valid, :3]
        if centerline.shape[0] < 2:
            continue
        radius = np.clip(branch[valid, 3], 1.0e-5, None)
        surface = None
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                surface = _legacy_tube_surface(
                    centerline, radius, num_circle_points
                )
        except Exception:
            surface = None
        if surface is None:
            surface = _robust_tube_surface(
                centerline, radius, num_circle_points
            )
        surfaces.append(surface)
        surface_by_index[branch_index] = surface

    if not surfaces:
        raise ValueError("Stage-2 artery has no renderable projected branches")
    if center_reference_branch_indices is None:
        reference_indices = tuple(surface_by_index)[:1]
    else:
        reference_indices = center_reference_branch_indices
        missing = [index for index in reference_indices if index not in surface_by_index]
        if missing:
            raise ValueError(
                "Projection centring reference branches are missing or invalid: "
                f"{missing}"
            )
    reference_points = np.concatenate(
        [surface_by_index[index].reshape(-1, 3) for index in reference_indices],
        axis=0,
    )
    center = reference_points.mean(axis=0).astype(np.float32)
    applied_center = center
    if centering_offset_m is not None:
        applied_center = np.asarray(centering_offset_m, dtype=np.float32)
        if applied_center.shape != (3,) or not np.isfinite(applied_center).all():
            raise ValueError("centering_offset_m must contain three finite values")
    centered = tuple(
        (surface - applied_center.reshape(1, 1, 3)).astype(np.float32)
        for surface in surfaces
    )
    if not np.isfinite(center).all() or any(
        not np.isfinite(surface).all() for surface in centered
    ):
        raise ValueError("Tube generation produced non-finite centred geometry")
    return centered, center


def load_render_source(
    path: str | Path,
    projection: "ProjectionCase",
    num_circle_points: int,
    render_mode_fallback: str,
    maximum_center_offset_difference_mm: float = 0.01,
    missing_projected_branch_policy: str = "error",
) -> Stage2RenderSource:
    """Load and projection-centre the exact Stage-2 artery used for rendering."""

    from .data import ProjectionCase

    if not isinstance(projection, ProjectionCase):
        raise TypeError("projection must be a deepca.data.ProjectionCase")
    source_path = Path(path).expanduser().resolve()
    if source_path != Path(projection.path).expanduser().resolve():
        raise ValueError("Render source path must match ProjectionCase.path")
    points = _integer_at_least(
        num_circle_points, label="num_circle_points", minimum=3
    )
    fallback = str(render_mode_fallback).strip().lower()
    if fallback not in {"filled", "point"}:
        raise ValueError("render_mode_fallback must be 'filled' or 'point'")
    maximum_center_difference = _finite_number(
        maximum_center_offset_difference_mm,
        label="maximum_center_offset_difference_mm",
    )
    if not (
        0.0
        <= maximum_center_difference
        <= MAXIMUM_ALLOWED_CENTER_OFFSET_DIFFERENCE_MM
    ):
        raise ValueError(
            "maximum_center_offset_difference_mm must be in [0, "
            f"{MAXIMUM_ALLOWED_CENTER_OFFSET_DIFFERENCE_MM:g}]"
        )
    branch_policy = str(missing_projected_branch_policy).strip().lower()
    if branch_policy not in {"error", "all"}:
        raise ValueError("missing_projected_branch_policy must be 'error' or 'all'")
    if projection.projection_center_offset_xyz_mm is None:
        raise ValueError(
            f"{source_path}: fixed translation evaluation requires the recorded "
            "projection_center_offset used to produce the accurate projections"
        )
    if projection.projection_center_offset_source != "archive":
        raise ValueError(
            f"{source_path}: projection centring must come from the archive, got "
            f"{projection.projection_center_offset_source!r}"
        )

    with np.load(source_path, allow_pickle=False) as archive:
        if "artery" not in archive.files:
            raise KeyError(
                f"{source_path} is missing 'artery'; translation evaluation must "
                "re-render the original Stage-2 vessel"
            )
        raw_artery = np.asarray(archive["artery"])
        if raw_artery.dtype.hasobject:
            raise TypeError(f"{source_path}: artery cannot use object dtype")
        artery = np.asarray(raw_artery, dtype=np.float32)
        if artery.ndim != 3 or artery.shape[-1] < 4:
            raise ValueError(
                f"{source_path}: artery must have shape [M,N,>=4], got "
                f"{artery.shape}"
            )
        if artery.shape[0] < 1 or artery.shape[1] < 2:
            raise ValueError(f"{source_path}: artery has no renderable branch samples")
        if not np.isfinite(artery).all():
            raise ValueError(f"{source_path}: artery contains NaN or infinity")

        if "projected_branch_indices" in archive.files:
            projected_indices = _integer_indices(
                archive["projected_branch_indices"],
                label="projected_branch_indices",
                upper_bound=artery.shape[0],
            )
            projected_indices_source = "archive"
        else:
            if branch_policy != "all":
                raise KeyError(
                    f"{source_path} is missing projected_branch_indices; set "
                    "missing_projected_branch_policy: all only for a verified "
                    "all-branch Stage-2 cohort"
                )
            projected_indices = tuple(
                index
                for index, branch in enumerate(artery)
                if bool(np.any(np.abs(branch[:, :3]) > 0.0))
                and bool(np.any(branch[:, 3] > 0.0))
            )
            if not projected_indices:
                raise ValueError(
                    f"{source_path}: verified all-branch policy found no valid "
                    "artery branches"
                )
            projected_indices_source = "configured_all_valid_archive_branches"
        if "num_projected_branches" in archive.files:
            recorded_count = _integer_at_least(
                _npz_scalar(archive, "num_projected_branches"),
                label="num_projected_branches",
                minimum=1,
            )
            if recorded_count != len(projected_indices):
                raise ValueError(
                    "num_projected_branches disagrees with projected_branch_indices"
                )

        global_reference: tuple[int, ...] | None = None
        local_reference: tuple[int, ...] | None = None
        if "projection_center_reference_branch_indices" in archive.files:
            center_reference_source = "archive"
            global_reference = _integer_indices(
                archive["projection_center_reference_branch_indices"],
                label="projection_center_reference_branch_indices",
                upper_bound=artery.shape[0],
            )
            local_by_global = {
                global_index: local_index
                for local_index, global_index in enumerate(projected_indices)
            }
            missing = [
                index for index in global_reference if index not in local_by_global
            ]
            if missing:
                raise ValueError(
                    "Projection centring reference branches were excluded from the "
                    f"projected subset: {missing}"
                )
            local_reference = tuple(local_by_global[index] for index in global_reference)
        else:
            center_reference_source = "historical_first_valid_projected_branch"

        render_mode = fallback
        render_mode_source = "configured_fallback"
        if "mask_render_mode" in archive.files:
            render_mode = str(_npz_scalar(archive, "mask_render_mode")).strip().lower()
            if render_mode not in {"filled", "point"}:
                raise ValueError(
                    f"{source_path}: unsupported mask_render_mode {render_mode!r}"
                )
            render_mode_source = "archive"

        renderer_points_source = "configured"
        if "num_circle_points" in archive.files:
            stored_points = _integer_at_least(
                _npz_scalar(archive, "num_circle_points"),
                label="num_circle_points",
                minimum=3,
            )
            if stored_points != points:
                raise ValueError(
                    f"{source_path}: stored num_circle_points={stored_points} but "
                    f"the translation renderer is configured for {points}"
                )
            renderer_points_source = "archive_verified"

        if "image_dim" in archive.files:
            stored_image_dim = _integer_at_least(
                _npz_scalar(archive, "image_dim"),
                label="image_dim",
                minimum=1,
            )
            if projection.images.shape[1:] != (stored_image_dim, stored_image_dim):
                raise ValueError(
                    f"{source_path}: stored image_dim={stored_image_dim} disagrees "
                    f"with projection image shape {projection.images.shape[1:]}"
                )

    projected_artery = artery[np.asarray(projected_indices, dtype=np.int64)]
    stored_center_offset_m = (
        np.asarray(projection.projection_center_offset_xyz_mm, dtype=np.float64)
        * 1.0e-3
    )
    surfaces, recomputed_center_offset_m = _artery_to_surface_rings(
        projected_artery,
        num_circle_points=points,
        center_reference_branch_indices=local_reference,
        centering_offset_m=stored_center_offset_m,
    )
    recomputed_offset_mm = (
        np.asarray(recomputed_center_offset_m, dtype=np.float64) * 1000.0
    )
    stored_offset_mm = np.asarray(
        projection.projection_center_offset_xyz_mm, dtype=np.float64
    )
    difference_mm = float(np.linalg.norm(recomputed_offset_mm - stored_offset_mm))
    if difference_mm > maximum_center_difference:
        raise ValueError(
            f"{source_path}: recomputed projection centre differs from the recorded "
            f"projection_center_offset by {difference_mm:.6g} mm, exceeding "
            f"maximum_center_offset_difference_mm={maximum_center_difference:.6g}; "
            "branch subset, centring reference, or tube renderer settings do not "
            "match the accurate projection provenance"
        )
    # ``surfaces`` were centred directly with the archive offset above. The
    # independently recomputed centroid is retained only for this provenance audit.
    centered_lines: list[np.ndarray] = []
    for branch in projected_artery:
        valid = np.logical_and(
            np.any(np.abs(branch[:, :3]) > 0.0, axis=1), branch[:, 3] > 0.0
        )
        if np.any(valid):
            centered_lines.append(
                branch[valid, :3] - stored_center_offset_m.reshape(1, 3)
            )
    if not centered_lines:
        raise ValueError(f"{source_path}: no valid projected centreline points")

    images = np.asarray(projection.images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1] <= 0 or images.shape[2] <= 0:
        raise ValueError("ProjectionCase.images must have shape [V,H,W]")
    stored_offset_tuple = tuple(float(value) for value in stored_offset_mm)
    readonly_surfaces = tuple(
        _readonly(surface, dtype=np.float32) for surface in surfaces
    )
    return Stage2RenderSource(
        path=source_path,
        artery_shape=tuple(int(value) for value in artery.shape),
        surface_rings=readonly_surfaces,
        centered_centerline_xyz_m=_readonly(
            np.concatenate(centered_lines, axis=0), dtype=np.float32
        ),
        projected_branch_indices=projected_indices,
        projected_branch_indices_source=projected_indices_source,
        projection_center_reference_branch_indices=global_reference,
        projection_center_reference_branch_indices_source=center_reference_source,
        stored_projection_center_offset_xyz_mm=stored_offset_tuple,
        centering_offset_used_xyz_mm=stored_offset_tuple,
        projection_center_offset_source=projection.projection_center_offset_source,
        recomputed_projection_center_offset_xyz_mm=tuple(
            float(value) for value in recomputed_offset_mm
        ),
        projection_center_offset_difference_mm=difference_mm,
        maximum_center_offset_difference_mm=maximum_center_difference,
        image_shape_hw=(int(images.shape[1]), int(images.shape[2])),
        sid_m=float(projection.sid_m),
        sid_source=projection.sid_source,
        source_to_isocentre_m=float(projection.source_to_isocentre_m),
        source_to_isocentre_source="configuration",
        detector_pixel_spacing_mm=float(projection.detector_pixel_spacing_mm),
        detector_pixel_spacing_source=projection.detector_pixel_spacing_source,
        render_mode=render_mode,
        render_mode_source=render_mode_source,
        renderer_num_circle_points=points,
        renderer_num_circle_points_source=renderer_points_source,
        projection_theta_deg=_readonly(projection.theta_deg, dtype=np.float64),
        projection_phi_deg=_readonly(projection.phi_deg, dtype=np.float64),
        projection_images=_readonly(images, dtype=np.float32),
    )


@dataclass(frozen=True)
class _Camera:
    detector_center_xyz_m: np.ndarray
    source_xyz_m: np.ndarray
    detector_x_axis: np.ndarray
    detector_y_axis: np.ndarray


def _camera(
    theta_deg: float,
    phi_deg: float,
    *,
    sid_m: float,
    source_to_isocentre_m: float,
) -> _Camera:
    detector_to_isocentre_m = float(sid_m) - float(source_to_isocentre_m)
    if (
        not math.isfinite(detector_to_isocentre_m)
        or not math.isfinite(float(source_to_isocentre_m))
        or detector_to_isocentre_m <= 0.0
        or float(source_to_isocentre_m) <= 0.0
    ):
        raise ValueError("Projection calibration must satisfy SID > SOD > 0")
    theta = math.radians(float(theta_deg))
    phi = math.radians(float(phi_deg))
    coordinate_change = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    coordinate_inverse = np.linalg.inv(coordinate_change)
    rotation_theta = coordinate_change @ np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ) @ coordinate_inverse
    rotation_phi = coordinate_change @ np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(phi), math.sin(phi)],
            [0.0, -math.sin(phi), math.cos(phi)],
        ]
    ) @ coordinate_inverse
    rotation = rotation_theta @ rotation_phi
    detector = rotation @ np.asarray([0.0, 0.0, detector_to_isocentre_m])
    source = (
        -detector / detector_to_isocentre_m * float(source_to_isocentre_m)
    )
    local_x = rotation @ np.asarray([0.0, 1.0, 0.0])
    local_y = rotation @ np.asarray([-1.0, 0.0, 0.0])
    return _Camera(detector, source, local_x, local_y)


def _ray_image_intersection(
    points_xyz_m: np.ndarray, camera: _Camera
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz_m, dtype=np.float64).reshape(-1, 3)
    normal = np.cross(camera.detector_x_axis, camera.detector_y_axis).reshape(1, 3)
    source_vectors = camera.source_xyz_m.reshape(1, 3) - points
    projected = np.einsum("ij,ij->i", normal, source_vectors)
    valid = np.logical_and(np.abs(projected) > 1.0e-6, np.isfinite(projected))
    valid_points = points[valid]
    valid_vectors = source_vectors[valid]
    if valid_points.size == 0:
        return np.empty((0, 3), dtype=np.float64), valid
    w = valid_points - camera.detector_center_xyz_m.reshape(1, 3)
    scale = -np.einsum("ij,ij->i", normal, w) / projected[valid]
    intersections = valid_points + scale[:, None] * valid_vectors
    finite = np.isfinite(intersections).all(axis=1)
    if not np.all(finite):
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices[~finite]] = False
        intersections = intersections[finite]
    return intersections, valid


def _convert_to_pixels(
    plane_points: np.ndarray,
    camera: _Camera,
    *,
    image_shape_hw: tuple[int, int],
    detector_pixel_spacing_mm: float,
) -> np.ndarray:
    height, width = image_shape_hw
    width_m = float(detector_pixel_spacing_mm) * width / 1000.0
    height_m = float(detector_pixel_spacing_mm) * height / 1000.0
    x_axis = camera.detector_x_axis.reshape(1, 3)
    y_axis = camera.detector_y_axis.reshape(1, 3)
    detector = camera.detector_center_xyz_m
    local_origin = detector + (
        width_m * ((1.0 - width / 2.0) / width) * x_axis[0]
        + height_m * ((1.0 - height / 2.0) / height) * y_axis[0]
    )
    local_xmax = detector + (
        width_m * ((width - width / 2.0) / width) * x_axis[0]
        + height_m * ((1.0 - height / 2.0) / height) * y_axis[0]
    )
    local_ymax = detector + (
        width_m * ((1.0 - width / 2.0) / width) * x_axis[0]
        + height_m * ((height - height / 2.0) / height) * y_axis[0]
    )
    vectors = np.asarray(plane_points, dtype=np.float64) - local_origin
    x_projected = (
        np.einsum("ij,ij->i", vectors, x_axis)
        / np.einsum("ij,ij->i", x_axis, x_axis)
    )[:, None] * x_axis
    y_projected = (
        np.einsum("ij,ij->i", vectors, y_axis)
        / np.einsum("ij,ij->i", y_axis, y_axis)
    )[:, None] * y_axis
    x_sign = np.sign(
        np.einsum("ij,ij->i", (local_xmax - local_origin).reshape(1, 3), x_projected)
    )
    y_sign = np.sign(
        np.einsum("ij,ij->i", (local_ymax - local_origin).reshape(1, 3), y_projected)
    )
    x = (
        x_sign
        * width
        * np.linalg.norm(x_projected, axis=1)
        / np.linalg.norm(local_xmax - local_origin)
    )
    y = (
        y_sign
        * height
        * np.linalg.norm(y_projected, axis=1)
        / np.linalg.norm(local_ymax - local_origin)
    )
    return np.stack((x, height - y), axis=1)


def _in_bounds(pixels_xy: np.ndarray, image_shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_shape_hw
    return np.logical_and.reduce(
        (
            np.isfinite(pixels_xy).all(axis=1),
            pixels_xy[:, 0] > 0.0,
            pixels_xy[:, 0] < width,
            pixels_xy[:, 1] > 0.0,
            pixels_xy[:, 1] < height,
        )
    )


def _render_points_to_mask(
    points_xyz_m: np.ndarray,
    camera: _Camera,
    *,
    image_shape_hw: tuple[int, int],
    detector_pixel_spacing_mm: float,
) -> np.ndarray:
    height, width = image_shape_hw
    plane_points, _ = _ray_image_intersection(points_xyz_m, camera)
    pixels = np.round(
        _convert_to_pixels(
            plane_points,
            camera,
            image_shape_hw=image_shape_hw,
            detector_pixel_spacing_mm=detector_pixel_spacing_mm,
        )
    )
    valid = _in_bounds(pixels, image_shape_hw)
    mask = np.zeros((height, width), dtype=np.bool_)
    if not np.any(valid):
        return mask
    pixels = pixels[valid].astype(np.int64)
    rows = height - pixels[:, 1]
    columns = pixels[:, 0]
    mask[rows, columns] = True
    return mask


def _render_filled_rings_to_mask(
    surface_rings: Sequence[np.ndarray],
    camera: _Camera,
    *,
    image_shape_hw: tuple[int, int],
    detector_pixel_spacing_mm: float,
) -> np.ndarray:
    height, width = image_shape_hw
    mask = np.zeros((height, width), dtype=np.bool_)
    bridge_stride = 2
    for branch_surface in surface_rings:
        rings = branch_surface.reshape(-1, branch_surface.shape[-2], 3)
        previous_rows: np.ndarray | None = None
        previous_columns: np.ndarray | None = None
        for ring in rings:
            plane_points, _ = _ray_image_intersection(ring, camera)
            if plane_points.shape[0] < 3:
                continue
            pixels = np.round(
                _convert_to_pixels(
                    plane_points,
                    camera,
                    image_shape_hw=image_shape_hw,
                    detector_pixel_spacing_mm=detector_pixel_spacing_mm,
                )
            )
            pixels = pixels[_in_bounds(pixels, image_shape_hw)]
            if pixels.shape[0] < 3:
                continue
            rows = np.clip(
                (height - pixels[:, 1]).astype(np.int32), 0, height - 1
            )
            columns = np.clip(pixels[:, 0].astype(np.int32), 0, width - 1)
            polygon_rows, polygon_columns = draw.polygon(
                rows, columns, shape=mask.shape
            )
            mask[polygon_rows, polygon_columns] = True
            if previous_rows is not None and previous_columns is not None:
                ring_length = min(len(rows), len(previous_rows))
                for point_index in range(0, ring_length, bridge_stride):
                    line_rows, line_columns = draw.line(
                        int(previous_rows[point_index]),
                        int(previous_columns[point_index]),
                        int(rows[point_index]),
                        int(columns[point_index]),
                    )
                    valid = np.logical_and.reduce(
                        (
                            line_rows >= 0,
                            line_rows < height,
                            line_columns >= 0,
                            line_columns < width,
                        )
                    )
                    mask[line_rows[valid], line_columns[valid]] = True
            previous_rows = rows
            previous_columns = columns
    return mask


def _render_mask(
    surface_rings: Sequence[np.ndarray],
    *,
    theta_deg: float,
    phi_deg: float,
    image_shape_hw: tuple[int, int],
    sid_m: float,
    source_to_isocentre_m: float,
    detector_pixel_spacing_mm: float,
    render_mode: str,
) -> np.ndarray:
    camera = _camera(
        theta_deg,
        phi_deg,
        sid_m=sid_m,
        source_to_isocentre_m=source_to_isocentre_m,
    )
    if render_mode == "filled":
        binary = _render_filled_rings_to_mask(
            surface_rings,
            camera,
            image_shape_hw=image_shape_hw,
            detector_pixel_spacing_mm=detector_pixel_spacing_mm,
        )
    elif render_mode == "point":
        all_points = np.concatenate(
            [surface.reshape(-1, 3) for surface in surface_rings], axis=0
        )
        binary = _render_points_to_mask(
            all_points,
            camera,
            image_shape_hw=image_shape_hw,
            detector_pixel_spacing_mm=detector_pixel_spacing_mm,
        )
    else:
        raise ValueError(f"Unsupported Stage-2 render mode {render_mode!r}")
    closed = morphology.closing(binary, morphology.disk(2))
    blurred = filters.gaussian(closed, sigma=0.5) > 0.25
    return blurred.astype(np.float32)


def _binary_dice_2d(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = np.asarray(first) > 0.5
    second_mask = np.asarray(second) > 0.5
    denominator = int(first_mask.sum()) + int(second_mask.sum())
    if denominator == 0:
        return 1.0
    return float(
        2.0 * np.logical_and(first_mask, second_mask).sum() / denominator
    )


def _visible_fraction(
    points_xyz_m: np.ndarray,
    *,
    theta_deg: float,
    phi_deg: float,
    source: Stage2RenderSource,
) -> float:
    points = np.asarray(points_xyz_m, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return 0.0
    camera = _camera(
        theta_deg,
        phi_deg,
        sid_m=source.sid_m,
        source_to_isocentre_m=source.source_to_isocentre_m,
    )
    plane_points, projection_valid = _ray_image_intersection(points, camera)
    visible = np.zeros(points.shape[0], dtype=np.bool_)
    if plane_points.shape[0] > 0:
        pixels = _convert_to_pixels(
            plane_points,
            camera,
            image_shape_hw=source.image_shape_hw,
            detector_pixel_spacing_mm=source.detector_pixel_spacing_mm,
        )
        visible[np.flatnonzero(projection_valid)] = _in_bounds(
            pixels, source.image_shape_hw
        )
    return float(np.mean(visible))


def _source_view_index(
    source: Stage2RenderSource,
    stored_image: np.ndarray,
    theta_deg: float,
    phi_deg: float,
) -> int | None:
    candidates = np.flatnonzero(
        np.logical_and(
            np.isclose(source.projection_theta_deg, theta_deg, atol=1.0e-6),
            np.isclose(source.projection_phi_deg, phi_deg, atol=1.0e-6),
        )
    )
    if candidates.size == 0:
        return None
    if candidates.size == 1:
        return int(candidates[0])
    scores = [
        _binary_dice_2d(source.projection_images[index], stored_image)
        for index in candidates
    ]
    return int(candidates[int(np.argmax(scores))])


def _matching_condition_id(vector_mm: np.ndarray) -> str | None:
    for condition in DEFAULT_TRANSLATION_CONDITIONS:
        if np.allclose(
            vector_mm,
            np.asarray(condition.translation_xyz_mm),
            rtol=0.0,
            atol=1.0e-9,
        ):
            return condition.condition_id
    return None


def render_translation(
    source: Stage2RenderSource,
    stored_image: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    translation_xyz_mm: Sequence[float],
    minimum_clean_dice: float,
    visibility_threshold: float,
    fail_below_visibility: bool,
    *,
    clean_rerender: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate a clean re-render and return a translated view-2 replacement.

    The returned diagnostics contain only JSON-native scalar/list/dict values.
    The caller must retain input view 1 and replace only input position 1 with
    the returned mask. ``clean_rerender`` may reuse the exact zero-translation
    mask already validated for the same case, selected view, and camera.
    """

    if not isinstance(source, Stage2RenderSource):
        raise TypeError("source must be a Stage2RenderSource")
    image = np.asarray(stored_image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or tuple(image.shape) != source.image_shape_hw:
        raise ValueError(
            f"stored_image must have shape {source.image_shape_hw}, got {image.shape}"
        )
    if not np.isfinite(image).all() or np.any(image < 0.0):
        raise ValueError("stored_image must contain finite non-negative values")
    theta = _finite_number(theta_deg, label="theta_deg")
    phi = _finite_number(phi_deg, label="phi_deg")
    clean_threshold = _unit_interval(
        minimum_clean_dice, label="minimum_clean_dice"
    )
    if clean_threshold < MINIMUM_REQUIRED_CLEAN_RERENDER_DICE:
        raise ValueError(
            "minimum_clean_dice cannot be below the fixed contractual floor "
            f"of {MINIMUM_REQUIRED_CLEAN_RERENDER_DICE:.2f}"
        )
    visibility_gate = _unit_interval(
        visibility_threshold, label="visibility_threshold"
    )
    if not isinstance(fail_below_visibility, (bool, np.bool_)):
        raise ValueError("fail_below_visibility must be boolean")
    translation_mm = np.asarray(translation_xyz_mm, dtype=np.float64)
    if translation_mm.shape != (3,) or not np.isfinite(translation_mm).all():
        raise ValueError("translation_xyz_mm must contain three finite values")

    stored_foreground = int(np.count_nonzero(image > 0.5))
    if stored_foreground == 0:
        raise ValueError(
            f"{source.path} stored view 2 has no foreground vessel pixels; an "
            "empty/empty Dice score cannot validate renderer reproduction"
        )

    if clean_rerender is None:
        clean = _render_mask(
            source.surface_rings,
            theta_deg=theta,
            phi_deg=phi,
            image_shape_hw=source.image_shape_hw,
            sid_m=source.sid_m,
            source_to_isocentre_m=source.source_to_isocentre_m,
            detector_pixel_spacing_mm=source.detector_pixel_spacing_mm,
            render_mode=source.render_mode,
        )
    else:
        clean = np.asarray(clean_rerender, dtype=np.float32)
        if clean.shape != source.image_shape_hw or not np.isfinite(clean).all():
            raise ValueError(
                "clean_rerender must be the finite zero-translation mask for "
                f"shape {source.image_shape_hw}, got {clean.shape}"
            )
        clean = np.ascontiguousarray(clean)
    clean_foreground = int(np.count_nonzero(clean > 0.5))
    if clean_foreground == 0:
        raise ValueError(
            f"{source.path} clean view-2 re-render has no foreground vessel "
            "pixels and cannot validate renderer reproduction"
        )
    clean_dice = _binary_dice_2d(clean, image)
    if clean_dice < clean_threshold:
        raise ValueError(
            f"{source.path} clean view-2 re-render Dice={clean_dice:.6f} is "
            f"below minimum_clean_dice={clean_threshold:.6f}; renderer, branch "
            "subset, or centring metadata does not reproduce the stored evidence"
        )

    translation_m = (translation_mm * 1.0e-3).astype(np.float32)
    translated_surfaces = tuple(
        np.asarray(surface, dtype=np.float32)
        + translation_m.reshape(1, 1, 3)
        for surface in source.surface_rings
    )
    if np.all(translation_mm == 0.0):
        replacement = clean.copy()
    else:
        replacement = _render_mask(
            translated_surfaces,
            theta_deg=theta,
            phi_deg=phi,
            image_shape_hw=source.image_shape_hw,
            sid_m=source.sid_m,
            source_to_isocentre_m=source.source_to_isocentre_m,
            detector_pixel_spacing_mm=source.detector_pixel_spacing_mm,
            render_mode=source.render_mode,
        )
    translated_centerline = (
        np.asarray(source.centered_centerline_xyz_m, dtype=np.float32)
        + translation_m.reshape(1, 3)
    )
    translated_surface_points = np.concatenate(
        [surface.reshape(-1, 3) for surface in translated_surfaces], axis=0
    )
    centerline_visibility = _visible_fraction(
        translated_centerline,
        theta_deg=theta,
        phi_deg=phi,
        source=source,
    )
    surface_visibility = _visible_fraction(
        translated_surface_points,
        theta_deg=theta,
        phi_deg=phi,
        source=source,
    )
    below_visibility = centerline_visibility < visibility_gate
    if below_visibility and bool(fail_below_visibility):
        raise ValueError(
            f"{source.path} translated view-2 centreline visibility="
            f"{centerline_visibility:.6f} is below visibility_threshold="
            f"{visibility_gate:.6f}"
        )

    translated_foreground = int(np.count_nonzero(replacement > 0.5))
    pixel_count = int(image.size)
    equivalent_system_mm = -translation_mm
    diagnostics: dict[str, Any] = {
        "evaluation_description": FIXED_TRANSLATION_DESCRIPTION,
        "direction_description": TRANSLATION_DIRECTION_DESCRIPTION,
        "condition_id": _matching_condition_id(translation_mm),
        "perturbed_input_position": 1,
        "source_view_index": _source_view_index(
            source, image, theta, phi
        ),
        "theta_deg": float(theta),
        "phi_deg": float(phi),
        "delta_theta_deg": 0.0,
        "delta_phi_deg": 0.0,
        "artery_translation_xyz_mm": [float(value) for value in translation_mm],
        "artery_translation_xyz_m": [float(value) for value in translation_m],
        "equivalent_system_translation_xyz_mm": [
            float(value) for value in equivalent_system_mm
        ],
        "translation_magnitude_mm": float(np.linalg.norm(translation_mm)),
        "clean_rerender_dice_vs_stored": float(clean_dice),
        "minimum_clean_rerender_dice": float(clean_threshold),
        "visible_centerline_fraction": float(centerline_visibility),
        "visible_vessel_surface_fraction": float(surface_visibility),
        "visibility_warning_threshold": float(visibility_gate),
        "below_visibility_warning_threshold": bool(below_visibility),
        "stored_foreground_pixels": stored_foreground,
        "clean_rerender_foreground_pixels": clean_foreground,
        "translated_foreground_pixels": translated_foreground,
        "stored_foreground_pixel_ratio": float(stored_foreground / pixel_count),
        "translated_foreground_pixel_ratio": float(
            translated_foreground / pixel_count
        ),
        "translated_to_stored_foreground_ratio": (
            None
            if stored_foreground == 0
            else float(translated_foreground / stored_foreground)
        ),
        "projected_branch_indices": list(source.projected_branch_indices),
        "projected_branch_indices_source": (
            source.projected_branch_indices_source
        ),
        "projection_center_reference_branch_indices": (
            None
            if source.projection_center_reference_branch_indices is None
            else list(source.projection_center_reference_branch_indices)
        ),
        "projection_center_reference_branch_indices_source": (
            source.projection_center_reference_branch_indices_source
        ),
        "stored_projection_center_offset_xyz_mm": list(
            source.stored_projection_center_offset_xyz_mm
        ),
        "centering_offset_used_xyz_mm": list(
            source.centering_offset_used_xyz_mm
        ),
        "projection_center_offset_source": source.projection_center_offset_source,
        "recomputed_projection_center_offset_xyz_mm": list(
            source.recomputed_projection_center_offset_xyz_mm
        ),
        "projection_center_offset_difference_mm": float(
            source.projection_center_offset_difference_mm
        ),
        "maximum_center_offset_difference_mm": float(
            source.maximum_center_offset_difference_mm
        ),
        "image_shape_hw": list(source.image_shape_hw),
        "sid_m": float(source.sid_m),
        "sid_source": source.sid_source,
        "source_to_isocentre_m": float(source.source_to_isocentre_m),
        "source_to_isocentre_source": source.source_to_isocentre_source,
        "detector_pixel_spacing_mm": float(source.detector_pixel_spacing_mm),
        "detector_pixel_spacing_source": (
            source.detector_pixel_spacing_source
        ),
        "render_mode": source.render_mode,
        "render_mode_source": source.render_mode_source,
        "renderer_num_circle_points": source.renderer_num_circle_points,
        "renderer_num_circle_points_source": (
            source.renderer_num_circle_points_source
        ),
        "coordinate_convention": dict(PATIENT_COORDINATE_CONVENTION),
        "translation_sign_convention": TRANSLATION_SIGN_CONVENTION,
        "implementation_convention": "translate_projection_centred_artery",
    }
    return np.ascontiguousarray(replacement, dtype=np.float32), diagnostics


__all__ = [
    "DEFAULT_TRANSLATION_CONDITIONS",
    "FIXED_TRANSLATION_DESCRIPTION",
    "FIXED_TRANSLATION_MODE",
    "MAXIMUM_ALLOWED_CENTER_OFFSET_DIFFERENCE_MM",
    "MINIMUM_REQUIRED_CLEAN_RERENDER_DICE",
    "PATIENT_COORDINATE_CONVENTION",
    "Stage2RenderSource",
    "TRANSLATION_DIRECTION_DESCRIPTION",
    "TRANSLATION_SIGN_CONVENTION",
    "TranslationCondition",
    "TranslationPlan",
    "is_fixed_translation_mode",
    "load_render_source",
    "render_translation",
    "resolve_translation_plan",
]
