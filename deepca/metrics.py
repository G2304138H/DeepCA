"""Strict, physical-grid-aware metrics for binary 3D vessel volumes.

The public metric functions deliberately require physical grid metadata.  A
matching array shape alone is not sufficient evidence that two medical image
volumes are spatially aligned.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import skimage
    from skimage.morphology import skeletonize as _skimage_skeletonize

    try:
        from skimage.morphology import skeletonize_3d as _skimage_skeletonize_3d
    except ImportError:  # Removed after being folded into skeletonize(method="lee").
        _skimage_skeletonize_3d = None
except ImportError:  # Dice and aggregation remain usable without scikit-image.
    skimage = None
    _skimage_skeletonize = None
    _skimage_skeletonize_3d = None


SKELETONIZATION_LIBRARY = "scikit-image"
SKELETONIZATION_VERSION = (
    str(skimage.__version__) if skimage is not None else None
)


def _normalise_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(shape)
    if len(values) != 3:
        raise ValueError(f"Grid shape must contain three ZYX dimensions, got {values!r}.")
    normalised: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Grid dimensions must be positive integers, not booleans.")
        integer = int(value)
        if integer != value or integer <= 0:
            raise ValueError(f"Grid dimensions must be positive integers, got {values!r}.")
        normalised.append(integer)
    return tuple(normalised)  # type: ignore[return-value]


def _normalise_vector(
    values: Sequence[float], *, name: str, positive: bool
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite XYZ values.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain three positive XYZ values.")
    return tuple(float(value) for value in array)


@dataclass(frozen=True)
class PhysicalGrid:
    """Identity-oriented grid metadata for a volume stored in ZYX array order."""

    shape_zyx: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_zyx", _normalise_shape(self.shape_zyx))
        object.__setattr__(
            self,
            "spacing_xyz_mm",
            _normalise_vector(
                self.spacing_xyz_mm,
                name="spacing_xyz_mm",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "origin_xyz_mm",
            _normalise_vector(
                self.origin_xyz_mm,
                name="origin_xyz_mm",
                positive=False,
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "shape_zyx": list(self.shape_zyx),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "origin_xyz_mm": list(self.origin_xyz_mm),
            "axis_order": "ZYX",
        }


def _coerce_grid(grid: Any, *, name: str) -> PhysicalGrid:
    if isinstance(grid, PhysicalGrid):
        return grid
    required = ("shape_zyx", "spacing_xyz_mm", "origin_xyz_mm")
    missing = [attribute for attribute in required if not hasattr(grid, attribute)]
    if missing:
        raise TypeError(
            f"{name} must be PhysicalGrid-like and provide {required}; "
            f"missing {missing}."
        )
    return PhysicalGrid(
        shape_zyx=getattr(grid, "shape_zyx"),
        spacing_xyz_mm=getattr(grid, "spacing_xyz_mm"),
        origin_xyz_mm=getattr(grid, "origin_xyz_mm"),
    )


def _validate_volume_shape(volume: Any, *, name: str) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D ZYX volume, got shape {array.shape}.")
    if any(dimension <= 0 for dimension in array.shape):
        raise ValueError(f"{name} dimensions must be non-empty, got shape {array.shape}.")
    return array


def validate_physical_grid_alignment(
    prediction: Any,
    target: Any,
    *,
    prediction_grid: Any,
    target_grid: Any,
    atol_mm: float = 1.0e-6,
) -> None:
    """Raise when two ZYX volumes do not occupy the same physical XYZ grid."""

    predicted = _validate_volume_shape(prediction, name="prediction")
    reference = _validate_volume_shape(target, name="target")
    predicted_grid = _coerce_grid(prediction_grid, name="prediction_grid")
    reference_grid = _coerce_grid(target_grid, name="target_grid")
    tolerance = float(atol_mm)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol_mm must be finite and non-negative.")

    if tuple(predicted.shape) != predicted_grid.shape_zyx:
        raise ValueError(
            "Prediction array/grid shape mismatch: "
            f"array {predicted.shape}, grid {predicted_grid.shape_zyx}."
        )
    if tuple(reference.shape) != reference_grid.shape_zyx:
        raise ValueError(
            "Target array/grid shape mismatch: "
            f"array {reference.shape}, grid {reference_grid.shape_zyx}."
        )
    if predicted_grid.shape_zyx != reference_grid.shape_zyx:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{predicted_grid.shape_zyx} != {reference_grid.shape_zyx}."
        )
    if not np.allclose(
        predicted_grid.spacing_xyz_mm,
        reference_grid.spacing_xyz_mm,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Prediction/target spacing mismatch (XYZ mm): "
            f"{predicted_grid.spacing_xyz_mm} != {reference_grid.spacing_xyz_mm}."
        )
    if not np.allclose(
        predicted_grid.origin_xyz_mm,
        reference_grid.origin_xyz_mm,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Prediction/target origin mismatch (XYZ mm): "
            f"{predicted_grid.origin_xyz_mm} != {reference_grid.origin_xyz_mm}."
        )


def _validate_binary_mask(mask: Any, *, name: str) -> np.ndarray:
    array = _validate_volume_shape(mask, name=name)
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must contain numeric or boolean binary values.")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"{name} must be strictly binary (only 0 and 1); threshold it explicitly first."
        )
    return array.astype(bool, copy=False)


def _prepare_binary_pair(
    prediction: Any,
    target: Any,
    *,
    prediction_grid: Any,
    target_grid: Any,
    atol_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    validate_physical_grid_alignment(
        prediction,
        target,
        prediction_grid=prediction_grid,
        target_grid=target_grid,
        atol_mm=atol_mm,
    )
    return (
        _validate_binary_mask(prediction, name="prediction"),
        _validate_binary_mask(target, name="target"),
    )


def binary_dice(
    prediction: Any,
    target: Any,
    *,
    prediction_grid: Any,
    target_grid: Any,
    atol_mm: float = 1.0e-6,
) -> float:
    """Return hard 3D Dice after strict binary and physical-grid validation.

    Both masks empty is defined as 1.0.  Exactly one mask empty is defined as
    0.0.
    """

    predicted, reference = _prepare_binary_pair(
        prediction,
        target,
        prediction_grid=prediction_grid,
        target_grid=target_grid,
        atol_mm=atol_mm,
    )
    predicted_count = int(np.count_nonzero(predicted))
    reference_count = int(np.count_nonzero(reference))
    if predicted_count == 0 and reference_count == 0:
        return 1.0
    if predicted_count == 0 or reference_count == 0:
        return 0.0
    intersection = int(np.count_nonzero(predicted & reference))
    return float(2.0 * intersection / (predicted_count + reference_count))


def _skeletonizer_backend() -> tuple[str, str]:
    if _skimage_skeletonize is None:
        raise ImportError(
            "binary_cldice requires scikit-image; install the project's pinned "
            "scikit-image dependency before evaluation."
        )
    try:
        has_method_argument = "method" in inspect.signature(
            _skimage_skeletonize
        ).parameters
    except (TypeError, ValueError):
        has_method_argument = False
    if has_method_argument:
        return "skimage.morphology.skeletonize", "lee"
    if _skimage_skeletonize_3d is not None:
        return "skimage.morphology.skeletonize_3d", "lee"
    # Historical scikit-image versions select Lee automatically for 3D input.
    return "skimage.morphology.skeletonize", "lee (3D default)"


def skeletonization_metadata() -> dict[str, object]:
    """Return reproducibility metadata for the hard-clDice skeletonizer."""

    function, method = _skeletonizer_backend()
    return {
        "library": SKELETONIZATION_LIBRARY,
        "version": SKELETONIZATION_VERSION,
        "function": function,
        "method": method,
    }


def _skeletonize_3d(mask: np.ndarray) -> np.ndarray:
    function, method = _skeletonizer_backend()
    if function.endswith("skeletonize_3d"):
        skeleton = _skimage_skeletonize_3d(mask)  # type: ignore[misc]
    elif method == "lee":
        skeleton = _skimage_skeletonize(mask, method="lee")  # type: ignore[misc]
    else:
        skeleton = _skimage_skeletonize(mask)  # type: ignore[misc]
    skeleton_array = np.asarray(skeleton, dtype=bool)
    if skeleton_array.shape != mask.shape:
        raise RuntimeError(
            "scikit-image skeletonization changed the mask shape: "
            f"{mask.shape} -> {skeleton_array.shape}."
        )
    return skeleton_array


def binary_cldice(
    prediction: Any,
    target: Any,
    *,
    prediction_grid: Any,
    target_grid: Any,
    atol_mm: float = 1.0e-6,
) -> float:
    """Return standard hard 3D centerline Dice (clDice).

    Skeletons are generated deterministically with scikit-image's Lee method.
    Both masks empty is defined as 1.0; exactly one empty is 0.0.  If an
    unexpected skeletonizer result is empty for a non-empty mask, the score is
    conservatively defined as 0.0.
    """

    predicted, reference = _prepare_binary_pair(
        prediction,
        target,
        prediction_grid=prediction_grid,
        target_grid=target_grid,
        atol_mm=atol_mm,
    )
    predicted_count = int(np.count_nonzero(predicted))
    reference_count = int(np.count_nonzero(reference))
    if predicted_count == 0 and reference_count == 0:
        return 1.0
    if predicted_count == 0 or reference_count == 0:
        return 0.0

    predicted_skeleton = _skeletonize_3d(predicted)
    reference_skeleton = _skeletonize_3d(reference)
    predicted_skeleton_count = int(np.count_nonzero(predicted_skeleton))
    reference_skeleton_count = int(np.count_nonzero(reference_skeleton))
    if predicted_skeleton_count == 0 or reference_skeleton_count == 0:
        return 0.0

    topology_precision = float(
        np.count_nonzero(predicted_skeleton & reference)
        / predicted_skeleton_count
    )
    topology_sensitivity = float(
        np.count_nonzero(reference_skeleton & predicted)
        / reference_skeleton_count
    )
    denominator = topology_precision + topology_sensitivity
    if denominator == 0.0:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / denominator)


@dataclass(frozen=True)
class AggregateStats:
    """Serializable aggregate statistics over per-case metric values."""

    mean: float
    sample_std: float
    median: float
    standard_error: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "sample_std": self.sample_std,
            "median": self.median,
            "standard_error": self.standard_error,
            "n": self.n,
        }


def aggregate_statistics(values: Iterable[float]) -> AggregateStats:
    """Aggregate finite case-level values using sample standard deviation.

    For a single case, sample standard deviation and standard error are both
    defined as 0.0.  An empty sequence is an error rather than a silent NaN.
    """

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    else:
        array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot aggregate an empty metric sequence.")
    if not np.isfinite(array).all():
        raise ValueError("Metric values must all be finite.")
    count = int(array.size)
    sample_std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    standard_error = sample_std / sqrt(count) if count > 1 else 0.0
    return AggregateStats(
        mean=float(np.mean(array)),
        sample_std=sample_std,
        median=float(np.median(array)),
        standard_error=float(standard_error),
        n=count,
    )


__all__ = [
    "AggregateStats",
    "PhysicalGrid",
    "SKELETONIZATION_LIBRARY",
    "SKELETONIZATION_VERSION",
    "aggregate_statistics",
    "binary_cldice",
    "binary_dice",
    "skeletonization_metadata",
    "validate_physical_grid_alignment",
]
