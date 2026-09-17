"""ImageCAS Stage-2 NPZ loading, pairing, preprocessing, and caching."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .geometry import (
    GridSpec,
    binary_cone_backproject,
    detector_limited_fov_mm,
    make_cubic_grid,
    resample_binary_xyz_to_grid,
)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Data validation remains importable in preprocessing-only environments.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


CASE_PATTERN = re.compile(r"(?i)^(lca|rca)[_-]?0*([0-9]+)$")


@dataclass(frozen=True)
class ProjectionCase:
    path: Path
    case_id: str
    vessel_type: str
    case_number: int
    images: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    sid_m: float
    sid_source: str
    detector_pixel_spacing_mm: float
    detector_pixel_spacing_source: str
    source_to_isocentre_m: float
    projection_center_offset_xyz_mm: Optional[np.ndarray]
    projection_center_offset_source: str
    clinical_views: tuple[str, ...]

    @property
    def num_views(self) -> int:
        return int(self.images.shape[0])


@dataclass(frozen=True)
class GroundTruthCase:
    path: Path
    volume_xyz: np.ndarray
    spacing_xyz_mm: np.ndarray


@dataclass(frozen=True)
class CasePair:
    case_id: str
    vessel_type: str
    case_number: int
    projection_path: Path
    ground_truth_path: Path


def parse_typed_case_id(value: object) -> tuple[str, int]:
    text = str(value).strip()
    match = CASE_PATTERN.fullmatch(Path(text).stem)
    if match is None:
        raise ValueError(f"Expected an LCA/RCA case identifier, got {value!r}.")
    vessel, number_text = match.groups()
    number = int(number_text)
    if number <= 0:
        raise ValueError(f"Case number must be positive, got {number}.")
    return vessel.lower(), number


def format_case_id(vessel_type: str, case_number: int) -> str:
    vessel = str(vessel_type).lower()
    number = int(case_number)
    if vessel not in {"lca", "rca"} or number <= 0:
        raise ValueError(f"Invalid ImageCAS identity: vessel={vessel!r}, case={number}.")
    return f"{vessel}_{number:04d}"


def _scalar(npz: np.lib.npyio.NpzFile, key: str) -> object:
    value = np.asarray(npz[key])
    if value.dtype.hasobject:
        raise TypeError(f"{key!r} uses unsafe object dtype; pickle loading is disabled.")
    if value.size != 1:
        raise ValueError(f"{key!r} must be scalar, got shape {value.shape}.")
    return value.reshape(()).item()


def _unit_scale_to_mm(units: str, *, field: str) -> float:
    normalized = str(units).strip().lower()
    if normalized in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return 1.0
    if normalized in {"m", "meter", "meters", "metre", "metres"}:
        return 1000.0
    raise ValueError(f"Unsupported {field} units {units!r}; expected metres or millimetres.")


def _unit_scale_to_m(units: str, *, field: str) -> float:
    return _unit_scale_to_mm(units, field=field) * 1.0e-3


def _identity_candidates(npz: np.lib.npyio.NpzFile, path: Path) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []
    for source, value in (("filename", path.stem),):
        match = CASE_PATTERN.fullmatch(str(value).strip())
        if match:
            candidates.append((match.group(1).lower(), int(match.group(2)), source))
    for key in ("sample_name", "case_name"):
        if key in npz.files:
            try:
                value = str(_scalar(npz, key)).strip()
            except TypeError:
                continue
            match = CASE_PATTERN.fullmatch(Path(value).stem)
            if match:
                candidates.append((match.group(1).lower(), int(match.group(2)), key))
    if "source_relpath" in npz.files:
        value = str(_scalar(npz, "source_relpath")).replace("\\", "/")
        parts = [part for part in value.split("/") if part]
        for index, part in enumerate(parts[:-1]):
            if part.lower() in {"lca", "rca"} and parts[index + 1].isdigit():
                candidates.append(
                    (part.lower(), int(parts[index + 1]), "source_relpath")
                )
                break
    if "vessel_type" in npz.files and "case_id" in npz.files:
        try:
            vessel = str(_scalar(npz, "vessel_type")).strip().lower()
            case_value = _scalar(npz, "case_id")
        except TypeError:
            pass
        else:
            if vessel in {"lca", "rca"} and str(case_value).strip().isdigit():
                candidates.append((vessel, int(case_value), "vessel_type/case_id"))
    return candidates


def inspect_projection_identity(
    path: str | Path, *, expected_vessel: Optional[str] = None
) -> tuple[str, int]:
    npz_path = Path(path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as archive:
        if "images" not in archive.files:
            raise KeyError(f"{npz_path} is not a projection archive: missing 'images'.")
        candidates = _identity_candidates(archive, npz_path)
    identities = {(vessel, number) for vessel, number, _ in candidates}
    if not identities:
        raise ValueError(f"Could not determine a typed case identity for {npz_path}.")
    if len(identities) != 1:
        raise ValueError(f"Conflicting case identities in {npz_path}: {candidates}.")
    vessel, number = identities.pop()
    if expected_vessel is not None and vessel != str(expected_vessel).lower():
        raise ValueError(
            f"Projection {npz_path} is {vessel.upper()}, expected {expected_vessel.upper()}."
        )
    return vessel, number


def load_projection_npz(
    path: str | Path,
    *,
    expected_case_id: Optional[str] = None,
    expected_vessel: Optional[str] = None,
    fallback_sid_m: Optional[float],
    fallback_detector_pixel_spacing_mm: Optional[float],
    source_to_isocentre_m: float = 0.75,
    expected_detector_pixel_spacing_mm: Optional[float] = None,
    detector_spacing_tolerance_mm: float = 1.0e-4,
    default_projection_center_offset_units: str = "m",
    max_views: int = 7,
) -> ProjectionCase:
    """Load required projection fields without ever enabling pickle."""

    npz_path = Path(path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"images", "theta_deg", "phi_deg"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{npz_path} is missing required keys: {missing}.")
        candidates = _identity_candidates(archive, npz_path)
        identities = {(vessel, number) for vessel, number, _ in candidates}
        if not identities:
            if expected_case_id is None:
                raise ValueError(f"Could not determine a case identity for {npz_path}.")
            identities = {parse_typed_case_id(expected_case_id)}
        if len(identities) != 1:
            raise ValueError(f"Conflicting case identities in {npz_path}: {candidates}.")
        vessel, number = identities.pop()
        case_id = format_case_id(vessel, number)
        if expected_case_id is not None and case_id != expected_case_id.lower():
            raise ValueError(
                f"Projection identity {case_id!r} does not match requested {expected_case_id!r}."
            )
        if expected_vessel is not None and vessel != str(expected_vessel).lower():
            raise ValueError(
                f"Projection {case_id} is {vessel.upper()}, expected {expected_vessel.upper()}."
            )

        images = np.asarray(archive["images"], dtype=np.float32)
        theta = np.asarray(archive["theta_deg"], dtype=np.float32).reshape(-1)
        phi = np.asarray(archive["phi_deg"], dtype=np.float32).reshape(-1)
        if images.ndim != 3:
            raise ValueError(f"{npz_path}: images must have shape [V,H,W], got {images.shape}.")
        if images.shape[0] < 1 or images.shape[0] > int(max_views):
            raise ValueError(f"{npz_path}: expected 1..{max_views} views, got {images.shape[0]}.")
        if images.shape[1] <= 0 or images.shape[2] <= 0:
            raise ValueError(f"{npz_path}: detector dimensions must be positive.")
        if theta.shape != phi.shape or theta.shape != (images.shape[0],):
            raise ValueError(
                f"{npz_path}: images/theta_deg/phi_deg view counts disagree: "
                f"{images.shape[0]}, {theta.shape}, {phi.shape}."
            )
        if not np.isfinite(images).all() or not np.isfinite(theta).all() or not np.isfinite(phi).all():
            raise ValueError(f"{npz_path}: projection arrays contain NaN or infinity.")
        if np.any(images < 0.0):
            raise ValueError(f"{npz_path}: projection masks must be non-negative.")

        if "sid" in archive.files:
            sid_units = str(_scalar(archive, "sid_units")) if "sid_units" in archive.files else "m"
            sid_m = float(_scalar(archive, "sid")) * _unit_scale_to_m(
                sid_units, field="sid"
            )
            sid_source = "archive"
        elif fallback_sid_m is not None:
            sid_m = float(fallback_sid_m)
            sid_source = "fallback"
        else:
            raise KeyError(f"{npz_path} has no sid and no fallback_sid_m was configured.")
        sod_m = float(source_to_isocentre_m)
        if not np.isfinite(sid_m) or not np.isfinite(sod_m) or sid_m <= sod_m or sod_m <= 0.0:
            raise ValueError(f"{npz_path}: expected SID > SOD > 0, got {sid_m} m, {sod_m} m.")

        if "imager_pixel_spacing" in archive.files:
            units = (
                str(_scalar(archive, "imager_pixel_spacing_units"))
                if "imager_pixel_spacing_units" in archive.files
                else "mm"
            )
            pixel_spacing_mm = float(_scalar(archive, "imager_pixel_spacing")) * _unit_scale_to_mm(
                units, field="imager_pixel_spacing"
            )
            spacing_source = "archive"
        elif fallback_detector_pixel_spacing_mm is not None:
            pixel_spacing_mm = float(fallback_detector_pixel_spacing_mm)
            spacing_source = "fallback"
        else:
            raise KeyError(
                f"{npz_path} has no imager_pixel_spacing and no fallback was configured."
            )
        if not np.isfinite(pixel_spacing_mm) or pixel_spacing_mm <= 0.0:
            raise ValueError(f"{npz_path}: detector pixel spacing must be positive.")
        if expected_detector_pixel_spacing_mm is not None and spacing_source == "archive":
            expected_spacing = float(expected_detector_pixel_spacing_mm)
            if abs(pixel_spacing_mm - expected_spacing) > float(detector_spacing_tolerance_mm):
                raise ValueError(
                    f"{npz_path}: stored detector spacing {pixel_spacing_mm:g} mm is "
                    f"inconsistent with expected {expected_spacing:g} mm. The stored value "
                    "is authoritative; correct the cohort/config rather than replacing it."
                )

        if "projection_center_offset" in archive.files:
            center = np.asarray(archive["projection_center_offset"], dtype=np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError(
                    f"{npz_path}: projection_center_offset must be three finite XYZ values."
                )
            units = (
                str(_scalar(archive, "projection_center_offset_units"))
                if "projection_center_offset_units" in archive.files
                else default_projection_center_offset_units
            )
            center_mm = center * _unit_scale_to_mm(units, field="projection_center_offset")
            center_source = "archive"
        else:
            center_mm = None
            center_source = "ground_truth_center_fallback"

        if "clinical_views" in archive.files:
            clinical_array = np.asarray(archive["clinical_views"])
            if clinical_array.dtype.hasobject:
                raise TypeError(f"{npz_path}: clinical_views cannot use object dtype.")
            clinical_views = tuple(str(value) for value in clinical_array.reshape(-1))
            if len(clinical_views) != images.shape[0]:
                raise ValueError(f"{npz_path}: clinical_views must have one label per view.")
        else:
            clinical_views = tuple(f"view_{index}" for index in range(images.shape[0]))

        if "view_features" in archive.files:
            features = np.asarray(archive["view_features"], dtype=np.float64)
            expected_features = np.stack(
                (
                    np.sin(np.deg2rad(theta)),
                    np.cos(np.deg2rad(theta)),
                    np.sin(np.deg2rad(phi)),
                    np.cos(np.deg2rad(phi)),
                ),
                axis=1,
            )
            if features.shape != expected_features.shape or not np.allclose(
                features, expected_features, atol=1.0e-5
            ):
                raise ValueError(f"{npz_path}: view_features disagree with theta_deg/phi_deg.")

    return ProjectionCase(
        path=npz_path,
        case_id=case_id,
        vessel_type=vessel,
        case_number=number,
        images=np.ascontiguousarray(images),
        theta_deg=theta,
        phi_deg=phi,
        sid_m=sid_m,
        sid_source=sid_source,
        detector_pixel_spacing_mm=pixel_spacing_mm,
        detector_pixel_spacing_source=spacing_source,
        source_to_isocentre_m=sod_m,
        projection_center_offset_xyz_mm=(
            None if center_mm is None else np.asarray(center_mm, dtype=np.float64)
        ),
        projection_center_offset_source=center_source,
        clinical_views=clinical_views,
    )


def load_ground_truth_npz(
    path: str | Path,
    *,
    volume_key: str = "vol",
    spacing_key: str = "spacing",
    max_voxels: int = 200_000_000,
) -> GroundTruthCase:
    npz_path = Path(path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = [key for key in (volume_key, spacing_key) if key not in archive.files]
        if missing:
            raise KeyError(f"{npz_path} is missing required keys: {missing}.")
        volume = np.asarray(archive[volume_key])
        spacing = np.asarray(archive[spacing_key], dtype=np.float64)
        if volume.dtype.hasobject:
            raise TypeError(f"{npz_path}:{volume_key} cannot use object dtype.")
        if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
            raise ValueError(f"{npz_path}:{volume_key} must be non-empty 3D XYZ.")
        if volume.size > int(max_voxels):
            raise ValueError(
                f"{npz_path}:{volume_key} has {volume.size:,} voxels, above the "
                f"configured safety limit {int(max_voxels):,}."
            )
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
            raise ValueError(f"{npz_path}:{spacing_key} must be three positive XYZ mm values.")
        if np.issubdtype(volume.dtype, np.floating):
            if not np.isfinite(volume).all() or not np.allclose(volume, np.rint(volume), atol=1.0e-6):
                raise ValueError(f"{npz_path}:{volume_key} must contain finite integer-like labels.")
        if np.any(volume < 0):
            raise ValueError(f"{npz_path}:{volume_key} contains negative labels.")
    return GroundTruthCase(
        path=npz_path,
        volume_xyz=np.ascontiguousarray(volume),
        spacing_xyz_mm=spacing,
    )


def build_projection_index(
    projection_root: str | Path, *, expected_vessel: str
) -> dict[str, Path]:
    root = Path(projection_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Projection root does not exist: {root}")
    index: dict[str, Path] = {}
    errors: list[str] = []
    for path in sorted(root.rglob("*.npz")):
        try:
            vessel, number = inspect_projection_identity(path, expected_vessel=expected_vessel)
        except KeyError:
            continue
        except ValueError as error:
            errors.append(str(error))
            continue
        case_id = format_case_id(vessel, number)
        if case_id in index:
            raise ValueError(
                f"Ambiguous projection case {case_id}: {index[case_id]} and {path.resolve()}."
            )
        index[case_id] = path.resolve()
    if errors and not index:
        raise ValueError("No valid projection archives found. " + " | ".join(errors[:5]))
    if not index:
        raise FileNotFoundError(f"No projection NPZ files found under {root}.")
    return index


def resolve_case_pairs(
    case_ids: Sequence[str],
    *,
    projection_root: str | Path,
    ground_truth_root: str | Path,
    vessel_type: str,
    projection_index: Optional[Mapping[str, Path]] = None,
    strict: bool = True,
) -> tuple[list[CasePair], list[dict[str, str]]]:
    vessel = str(vessel_type).lower()
    index = (
        dict(projection_index)
        if projection_index is not None
        else build_projection_index(projection_root, expected_vessel=vessel)
    )
    gt_root = Path(ground_truth_root).expanduser().resolve()
    pairs: list[CasePair] = []
    failures: list[dict[str, str]] = []
    for case_id in case_ids:
        try:
            case_vessel, number = parse_typed_case_id(case_id)
            if case_vessel != vessel:
                raise ValueError(f"Case {case_id} is not part of the configured {vessel.upper()} cohort.")
            projection_path = index.get(case_id)
            if projection_path is None:
                raise FileNotFoundError(
                    f"No projection archive matched {case_id} under {Path(projection_root).resolve()}."
                )
            gt_path = (gt_root / vessel / f"{number}.npz").resolve()
            if not gt_path.is_file():
                raise FileNotFoundError(
                    f"Ground truth for {case_id} must exist at {gt_path}."
                )
            pairs.append(
                CasePair(case_id, vessel, number, projection_path.resolve(), gt_path)
            )
        except Exception as error:
            failure = {
                "case_id": str(case_id),
                "stage": "pairing",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
            failures.append(failure)
            if strict:
                raise RuntimeError(
                    f"Failed to pair requested case {case_id}: {type(error).__name__}: {error}"
                ) from error
    return pairs, failures


def select_view_indices(
    num_available: int,
    num_views: int,
    *,
    strategy: str,
    explicit_indices: Optional[Sequence[int]],
    seed: int,
    epoch: int,
    case_id: str,
) -> np.ndarray:
    available = int(num_available)
    count = int(num_views)
    if count <= 0 or count > available:
        raise ValueError(f"Requested {count} views but case {case_id} has {available}.")
    if explicit_indices is not None:
        indices = np.asarray(tuple(int(value) for value in explicit_indices), dtype=np.int64)
        if indices.size != count:
            raise ValueError("Explicit view_indices length must equal num_views.")
    elif strategy == "first":
        indices = np.arange(count, dtype=np.int64)
    elif strategy == "evenly_spaced":
        indices = np.rint(np.linspace(0, available - 1, count)).astype(np.int64)
    elif strategy == "random":
        digest = hashlib.sha256(f"{seed}:{epoch}:{case_id}".encode("utf-8")).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        indices = np.sort(generator.choice(available, size=count, replace=False))
    else:
        raise ValueError("view selection must be 'first', 'evenly_spaced', or 'random'.")
    if len(set(indices.tolist())) != count or np.any(indices < 0) or np.any(indices >= available):
        raise ValueError(f"Invalid view indices for {case_id}: {indices.tolist()}.")
    return indices


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _float32_array_sha256(array: np.ndarray) -> str:
    """Hash an array's exact canonical float32 shape and values."""

    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _normalise_image_replacements(
    replacements: Optional[Mapping[str, Mapping[int, np.ndarray]]],
    *,
    case_ids: Sequence[str],
) -> dict[str, dict[int, np.ndarray]]:
    """Validate and defensively copy per-case source-view replacement images."""

    if replacements is None:
        return {}
    if not isinstance(replacements, Mapping):
        raise TypeError(
            "image_replacements must map case IDs to source-view image mappings."
        )

    known_case_ids = {str(case_id).strip().lower() for case_id in case_ids}
    normalised: dict[str, dict[int, np.ndarray]] = {}
    for raw_case_id, raw_case_replacements in replacements.items():
        if not isinstance(raw_case_id, str) or not raw_case_id.strip():
            raise TypeError("image_replacements case IDs must be non-empty strings.")
        case_id = raw_case_id.strip().lower()
        if case_id not in known_case_ids:
            raise ValueError(
                f"image_replacements contains unknown case ID {raw_case_id!r}."
            )
        if case_id in normalised:
            raise ValueError(
                f"image_replacements contains duplicate normalized case ID {case_id!r}."
            )
        if not isinstance(raw_case_replacements, Mapping):
            raise TypeError(
                f"image_replacements[{raw_case_id!r}] must map source-view indices "
                "to 2D images."
            )

        case_replacements: dict[int, np.ndarray] = {}
        for raw_view_index, raw_image in raw_case_replacements.items():
            if isinstance(raw_view_index, (bool, np.bool_)) or not isinstance(
                raw_view_index, (int, np.integer)
            ):
                raise TypeError(
                    f"image_replacements[{case_id!r}] source-view indices must be "
                    f"integers, got {raw_view_index!r}."
                )
            view_index = int(raw_view_index)
            if view_index < 0:
                raise ValueError(
                    f"image_replacements[{case_id!r}] source-view index must be "
                    f"non-negative, got {view_index}."
                )

            image = np.asarray(raw_image)
            if image.ndim != 2:
                raise ValueError(
                    f"image_replacements[{case_id!r}][{view_index}] must be a 2D "
                    f"detector image, got shape {image.shape}."
                )
            if not (
                np.issubdtype(image.dtype, np.bool_)
                or np.issubdtype(image.dtype, np.integer)
                or np.issubdtype(image.dtype, np.floating)
            ):
                raise TypeError(
                    f"image_replacements[{case_id!r}][{view_index}] must contain "
                    "real numeric values."
                )
            if not np.isfinite(image).all():
                raise ValueError(
                    f"image_replacements[{case_id!r}][{view_index}] contains NaN "
                    "or infinity."
                )
            if np.any(image < 0):
                raise ValueError(
                    f"image_replacements[{case_id!r}][{view_index}] contains "
                    "negative values."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                float_image = np.asarray(image, dtype=np.float32)
            if not np.isfinite(float_image).all():
                raise ValueError(
                    f"image_replacements[{case_id!r}][{view_index}] cannot be "
                    "represented as finite float32 values."
                )
            case_replacements[view_index] = np.ascontiguousarray(float_image).copy()
        normalised[case_id] = case_replacements
    return normalised


class ImageCASDataset(Dataset):
    """Generate the released DeepCA 3D input directly from paired NPZ files."""

    def __init__(
        self,
        pairs: Sequence[CasePair],
        config: Mapping[str, Any],
        *,
        training: bool,
        num_views_override: Optional[int] = None,
        image_replacements: Optional[
            Mapping[str, Mapping[int, np.ndarray]]
        ] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to construct ImageCASDataset.")
        if not pairs:
            raise ValueError("ImageCASDataset requires at least one case pair.")
        self.pairs = tuple(pairs)
        self.config = config
        self.training = bool(training)
        self.epoch = 0
        self.image_replacements = _normalise_image_replacements(
            image_replacements,
            case_ids=[pair.case_id for pair in self.pairs],
        )
        data_config = config["data"]
        preprocessing = data_config["preprocessing"]
        target_config = data_config.get("ground_truth", {})
        views = data_config["views"]
        cache = data_config.get("cache", {})
        self.source_axis_order = str(
            target_config.get("source_axis_order", "XYZ")
        ).strip().upper()
        if self.source_axis_order != "XYZ":
            raise ValueError(
                "data.ground_truth.source_axis_order must be 'XYZ'. Other source "
                "orders require an explicit, tested conversion and cannot be ignored."
            )
        self.num_views = int(
            views["count"] if num_views_override is None else num_views_override
        )
        self.view_strategy = str(
            views.get("train_selection" if training else "eval_selection", views.get("selection", "first"))
        )
        self.explicit_view_indices = views.get("indices")
        self.seed = int(config.get("experiment", {}).get("seed", 1))
        self.volume_size = int(preprocessing.get("volume_size", 128))
        self.fov_mode = str(preprocessing.get("fov_mode", "detector")).lower()
        self.fov_mm = preprocessing.get("fov_mm")
        self.projection_threshold = float(preprocessing.get("projection_threshold", 0.0))
        self.backprojection_interpolation = str(
            preprocessing.get("backprojection_interpolation", "nearest")
        )
        self.combine = str(preprocessing.get("combine", "sum"))
        self.chunk_depth = int(preprocessing.get("chunk_depth", 8))
        self.cache_enabled = bool(cache.get("enabled", True))
        self.cache_dir = Path(
            cache.get("directory", config.get("experiment", {}).get("output_dir", "outputs") + "/cache")
        ).expanduser()
        if not self.cache_dir.is_absolute():
            self.cache_dir = Path.cwd() / self.cache_dir

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_case(self, pair: CasePair) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        data_config = self.config["data"]
        calibration = data_config["calibration"]
        target_config = data_config.get("ground_truth", {})
        projection = load_projection_npz(
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
        gt = load_ground_truth_npz(
            pair.ground_truth_path,
            volume_key=str(target_config.get("volume_key", "vol")),
            spacing_key=str(target_config.get("spacing_key", "spacing")),
            max_voxels=int(target_config.get("max_voxels", 200_000_000)),
        )
        indices = select_view_indices(
            projection.num_views,
            self.num_views,
            strategy=self.view_strategy,
            explicit_indices=self.explicit_view_indices,
            seed=self.seed,
            epoch=self.epoch if self.training else 0,
            case_id=pair.case_id,
        )
        case_replacements = self.image_replacements.get(pair.case_id, {})
        invalid_indices = sorted(
            index for index in case_replacements if index >= projection.num_views
        )
        if invalid_indices:
            raise ValueError(
                f"{pair.case_id}: replacement source-view indices {invalid_indices} "
                f"are outside the available range 0..{projection.num_views - 1}."
            )
        detector_shape = tuple(int(value) for value in projection.images.shape[1:])
        for source_view_index, replacement in case_replacements.items():
            if replacement.shape != detector_shape:
                raise ValueError(
                    f"{pair.case_id}: replacement for source view {source_view_index} "
                    f"has detector shape {replacement.shape}, expected {detector_shape}."
                )
        selected_index_set = set(int(value) for value in indices)
        unselected_indices = sorted(set(case_replacements).difference(selected_index_set))
        if unselected_indices:
            raise ValueError(
                f"{pair.case_id}: replacement source-view indices {unselected_indices} "
                f"are not in the ordered selected views {indices.tolist()}."
            )

        selected = np.ascontiguousarray(projection.images[indices], dtype=np.float32)
        selected = selected.copy()
        replaced_view_indices: list[int] = []
        replacement_sha256_by_view: dict[str, str] = {}
        for selected_position, source_view_index_value in enumerate(indices):
            source_view_index = int(source_view_index_value)
            replacement = case_replacements.get(source_view_index)
            if replacement is None:
                continue
            selected[selected_position] = replacement
            replaced_view_indices.append(source_view_index)
            replacement_sha256_by_view[str(source_view_index)] = (
                _float32_array_sha256(replacement)
            )
        selected_images_sha256 = _float32_array_sha256(selected)

        if projection.projection_center_offset_xyz_mm is None:
            center_mm = (np.asarray(gt.volume_xyz.shape) - 1) * gt.spacing_xyz_mm / 2.0
            center_source = "ground_truth_physical_center"
        else:
            center_mm = projection.projection_center_offset_xyz_mm
            center_source = "projection_archive"
        if self.fov_mode == "detector":
            fov_mm = detector_limited_fov_mm(
                projection.images.shape[1:],
                projection.detector_pixel_spacing_mm,
                projection.source_to_isocentre_m,
                projection.sid_m,
            )
        elif self.fov_mode == "fixed":
            if self.fov_mm is None:
                raise ValueError("preprocessing.fov_mm is required when fov_mode='fixed'.")
            fov_mm = float(self.fov_mm)
        else:
            raise ValueError("preprocessing.fov_mode must be 'detector' or 'fixed'.")
        grid = make_cubic_grid(self.volume_size, fov_mm, center_mm)

        cache_payload = {
            "schema_version": 3,
            "algorithm": "stage2_binary_cone_support_backprojection_v1",
            "projection": _file_signature(pair.projection_path),
            "ground_truth": _file_signature(pair.ground_truth_path),
            "case_id": pair.case_id,
            "view_indices": indices.tolist(),
            "selected_images_sha256": selected_images_sha256,
            "image_replacements": {
                "source_view_indices": replaced_view_indices,
                "sha256_by_source_view": replacement_sha256_by_view,
            },
            "sid_m": projection.sid_m,
            "sod_m": projection.source_to_isocentre_m,
            "pixel_spacing_mm": projection.detector_pixel_spacing_mm,
            "grid": grid.as_dict(),
            "projection_threshold": self.projection_threshold,
            "backprojection_interpolation": self.backprojection_interpolation,
            "combine": self.combine,
            "chunk_depth": self.chunk_depth,
            "fov_mode": self.fov_mode,
            "configured_fov_mm": (
                None if self.fov_mm is None else float(self.fov_mm)
            ),
            "ground_truth_interpretation": {
                "volume_key": str(target_config.get("volume_key", "vol")),
                "spacing_key": str(target_config.get("spacing_key", "spacing")),
                "source_axis_order": self.source_axis_order,
                "binarization": "labels > 0",
                "max_voxels": int(
                    target_config.get("max_voxels", 200_000_000)
                ),
            },
        }
        digest = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / pair.vessel_type / f"{pair.case_id}_{digest}.npz"
        if self.cache_enabled and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                input_volume = np.asarray(cached["input"], dtype=np.float32)
                target_volume = np.asarray(cached["target"], dtype=np.float32)
            if input_volume.shape != grid.shape_zyx or target_volume.shape != grid.shape_zyx:
                raise ValueError(f"Corrupt cache shape in {cache_path}.")
        else:
            input_volume = binary_cone_backproject(
                selected,
                projection.theta_deg[indices],
                projection.phi_deg[indices],
                grid=grid,
                sid_m=projection.sid_m,
                source_to_isocentre_m=projection.source_to_isocentre_m,
                detector_pixel_spacing_mm=projection.detector_pixel_spacing_mm,
                projection_threshold=self.projection_threshold,
                interpolation=self.backprojection_interpolation,
                combine=self.combine,
                chunk_depth=self.chunk_depth,
            )
            target_volume = resample_binary_xyz_to_grid(
                gt.volume_xyz, gt.spacing_xyz_mm, grid
            )
            if self.cache_enabled:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(
                    f".{cache_path.name}.{os.getpid()}.tmp.npz"
                )
                np.savez_compressed(
                    temporary,
                    input=input_volume,
                    target=target_volume,
                    metadata=np.asarray(json.dumps(cache_payload, sort_keys=True)),
                )
                os.replace(temporary, cache_path)
        if bool(target_config.get("require_nonempty_target", True)) and not np.any(
            target_volume
        ):
            raise ValueError(
                f"{pair.case_id}: the resampled target is empty on the model grid. "
                "Check projection_center_offset, spacing, FOV, and XYZ/ZYX conventions."
            )
        metadata = {
            "case_id": pair.case_id,
            "vessel_type": pair.vessel_type,
            "projection_path": str(pair.projection_path),
            "ground_truth_path": str(pair.ground_truth_path),
            "view_indices": indices.tolist(),
            "replaced_view_indices": replaced_view_indices,
            "selected_images_sha256": selected_images_sha256,
            "replacement_images_sha256": replacement_sha256_by_view,
            "clinical_views": [projection.clinical_views[index] for index in indices],
            "theta_deg": projection.theta_deg[indices].astype(float).tolist(),
            "phi_deg": projection.phi_deg[indices].astype(float).tolist(),
            "sid_m": projection.sid_m,
            "sid_source": projection.sid_source,
            "source_to_isocentre_m": projection.source_to_isocentre_m,
            "detector_pixel_spacing_mm": projection.detector_pixel_spacing_mm,
            "detector_pixel_spacing_source": projection.detector_pixel_spacing_source,
            "center_source": center_source,
            "grid": grid.as_dict(),
            "source_axis_order": self.source_axis_order,
            "tensor_axis_order": "ZYX",
        }
        return input_volume, target_volume, metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        input_volume, target_volume, metadata = self._load_case(self.pairs[index])
        return {
            "input": torch.from_numpy(input_volume[None].copy()),
            "target": torch.from_numpy(target_volume[None].copy()),
            "case_id": metadata["case_id"],
            "vessel_type": metadata["vessel_type"],
            "num_views": self.num_views,
            "view_indices": torch.as_tensor(metadata["view_indices"], dtype=torch.int64),
            "grid_spacing_xyz_mm": torch.as_tensor(
                metadata["grid"]["spacing_xyz_mm"], dtype=torch.float64
            ),
            "grid_origin_xyz_mm": torch.as_tensor(
                metadata["grid"]["origin_xyz_mm"], dtype=torch.float64
            ),
            "metadata_json": json.dumps(metadata, sort_keys=True),
        }


def worker_seed(worker_id: int) -> None:
    """Seed NumPy deterministically from PyTorch's per-worker seed."""

    if torch is None:
        return
    seed = int(torch.initial_seed() % (2**32))
    np.random.seed(seed)
