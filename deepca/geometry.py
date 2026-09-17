"""Stage-2 camera geometry and deterministic binary cone backprojection.

The Stage-2 NPZ angle convention is reproduced from the ImageCAS projection
pipeline, not inferred from the DeepCA paper. Coordinates are right-handed XYZ
in metres for camera operations. Network tensors are explicitly ZYX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """A cubic model grid whose array order is ZYX."""

    shape_zyx: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    center_xyz_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "shape_zyx": list(self.shape_zyx),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "center_xyz_mm": list(self.center_xyz_mm),
            "origin_xyz_mm": list(self.origin_xyz_mm),
            "axis_order": "ZYX",
        }


def make_cubic_grid(
    volume_size: int,
    fov_mm: float,
    center_xyz_mm: Sequence[float],
) -> GridSpec:
    size = int(volume_size)
    extent = float(fov_mm)
    center = np.asarray(center_xyz_mm, dtype=np.float64).reshape(3)
    if size <= 0:
        raise ValueError("volume_size must be positive.")
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("fov_mm must be finite and positive.")
    if not np.isfinite(center).all():
        raise ValueError("center_xyz_mm must contain three finite values.")
    spacing = np.full(3, extent / size, dtype=np.float64)
    origin = center - (size - 1) * spacing / 2.0
    return GridSpec(
        shape_zyx=(size, size, size),
        spacing_xyz_mm=tuple(float(value) for value in spacing),
        center_xyz_mm=tuple(float(value) for value in center),
        origin_xyz_mm=tuple(float(value) for value in origin),
    )


def detector_limited_fov_mm(
    detector_shape: Sequence[int],
    detector_pixel_spacing_mm: float,
    source_to_isocentre_m: float,
    sid_m: float,
) -> float:
    shape = np.asarray(detector_shape, dtype=np.int64).reshape(2)
    spacing = float(detector_pixel_spacing_mm)
    sod = float(source_to_isocentre_m)
    sid = float(sid_m)
    if np.any(shape <= 0) or not all(
        np.isfinite(value) and value > 0.0 for value in (spacing, sod, sid)
    ):
        raise ValueError("Detector shape, spacing, SOD, and SID must be positive.")
    if sid <= sod:
        raise ValueError(f"SID ({sid} m) must exceed SOD ({sod} m).")
    return float(np.min(shape) * spacing * sod / sid)


def stage2_angles_to_camera_frames(
    theta_deg: Iterable[float],
    phi_deg: Iterable[float],
    *,
    sid_m: float,
    source_to_isocentre_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source, detector centre, detector-column, and detector-row axes.

    This exactly follows ``get_local_params(..., coord_system_change=True)`` in
    the projection code that produced the supplied Stage-2 archives. The first
    detector axis advances stored image columns and the second advances rows
    after the producer's final vertical flip.
    """

    theta = np.deg2rad(np.asarray(tuple(theta_deg), dtype=np.float64).reshape(-1))
    phi = np.deg2rad(np.asarray(tuple(phi_deg), dtype=np.float64).reshape(-1))
    if theta.shape != phi.shape or theta.size == 0:
        raise ValueError("theta_deg and phi_deg must be matching non-empty arrays.")
    if not np.isfinite(theta).all() or not np.isfinite(phi).all():
        raise ValueError("Projection angles must be finite.")
    sid = float(sid_m)
    sod = float(source_to_isocentre_m)
    detector_distance = sid - sod
    if not np.isfinite(sid) or not np.isfinite(sod) or sod <= 0.0 or detector_distance <= 0.0:
        raise ValueError(f"Expected SID > SOD > 0, got SID={sid}, SOD={sod}.")

    change = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    change_inv = change.T
    sources: list[np.ndarray] = []
    detectors: list[np.ndarray] = []
    u_axes: list[np.ndarray] = []
    v_axes: list[np.ndarray] = []
    for theta_value, phi_value in zip(theta, phi):
        r_theta_native = np.asarray(
            [
                [np.cos(theta_value), -np.sin(theta_value), 0.0],
                [np.sin(theta_value), np.cos(theta_value), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        r_phi_native = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(phi_value), np.sin(phi_value)],
                [0.0, -np.sin(phi_value), np.cos(phi_value)],
            ]
        )
        rotation = (
            change @ r_theta_native @ change_inv @ change @ r_phi_native @ change_inv
        )
        direction = rotation @ np.asarray([0.0, 0.0, 1.0])
        u_axis = rotation @ np.asarray([0.0, 1.0, 0.0])
        v_axis = rotation @ np.asarray([-1.0, 0.0, 0.0])
        sources.append(-direction * sod)
        detectors.append(direction * detector_distance)
        u_axes.append(u_axis / np.linalg.norm(u_axis))
        v_axes.append(v_axis / np.linalg.norm(v_axis))
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (sources, detectors, u_axes, v_axes)
    )


def _sample_detector(
    image: np.ndarray,
    row: np.ndarray,
    column: np.ndarray,
    *,
    interpolation: str,
) -> np.ndarray:
    height, width = image.shape
    if interpolation == "nearest":
        rr = np.rint(row).astype(np.int64)
        cc = np.rint(column).astype(np.int64)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        sampled = np.zeros(row.shape, dtype=np.float32)
        sampled[valid] = image[rr[valid], cc[valid]]
        return sampled
    if interpolation != "bilinear":
        raise ValueError("backprojection interpolation must be 'nearest' or 'bilinear'.")
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(column).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1
    valid = (r0 >= 0) & (r1 < height) & (c0 >= 0) & (c1 < width)
    sampled = np.zeros(row.shape, dtype=np.float32)
    if not np.any(valid):
        return sampled
    wr = row[valid] - r0[valid]
    wc = column[valid] - c0[valid]
    sampled[valid] = (
        image[r0[valid], c0[valid]] * (1.0 - wr) * (1.0 - wc)
        + image[r0[valid], c1[valid]] * (1.0 - wr) * wc
        + image[r1[valid], c0[valid]] * wr * (1.0 - wc)
        + image[r1[valid], c1[valid]] * wr * wc
    )
    return sampled


def binary_cone_backproject(
    images: np.ndarray,
    theta_deg: Sequence[float],
    phi_deg: Sequence[float],
    *,
    grid: GridSpec,
    sid_m: float,
    source_to_isocentre_m: float,
    detector_pixel_spacing_mm: float,
    projection_threshold: float = 0.0,
    interpolation: str = "nearest",
    combine: str = "sum",
    chunk_depth: int = 8,
) -> np.ndarray:
    """Voxel-driven cone backprojection of binary projections onto ``grid``.

    Each per-view support is thresholded after backprojection, matching the
    paper's one-iteration-SIRT -> ``>0`` representation. The default sum keeps
    the released two-view input range {0,1,2}.
    """

    projections = np.asarray(images, dtype=np.float32)
    if projections.ndim != 3 or projections.shape[0] == 0:
        raise ValueError(f"images must have shape [V,H,W], got {projections.shape}.")
    if not np.isfinite(projections).all():
        raise ValueError("Projection images contain NaN or infinity.")
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    if theta.shape != phi.shape or theta.size != projections.shape[0]:
        raise ValueError("images, theta_deg, and phi_deg disagree on view count.")
    spacing_m = float(detector_pixel_spacing_mm) * 1.0e-3
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("detector_pixel_spacing_mm must be finite and positive.")
    threshold = float(projection_threshold)
    if not np.isfinite(threshold):
        raise ValueError("projection_threshold must be finite.")
    if combine not in {"sum", "mean", "union"}:
        raise ValueError("combine must be 'sum', 'mean', or 'union'.")
    if int(chunk_depth) <= 0:
        raise ValueError("chunk_depth must be positive.")

    sources, detectors, u_axes, v_axes = stage2_angles_to_camera_frames(
        theta,
        phi,
        sid_m=sid_m,
        source_to_isocentre_m=source_to_isocentre_m,
    )
    depth, height, width = grid.shape_zyx
    spacing_xyz_m = np.asarray(grid.spacing_xyz_mm, dtype=np.float64) * 1.0e-3
    local_x = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * spacing_xyz_m[0]
    local_y = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * spacing_xyz_m[1]
    local_z = (np.arange(depth, dtype=np.float64) - (depth - 1) / 2.0) * spacing_xyz_m[2]
    output = np.zeros((depth, height, width), dtype=np.float32)
    detector_height, detector_width = projections.shape[1:]

    binary_images = projections > threshold
    for start in range(0, depth, int(chunk_depth)):
        stop = min(start + int(chunk_depth), depth)
        zz, yy, xx = np.meshgrid(local_z[start:stop], local_y, local_x, indexing="ij")
        points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
        accumulated = np.zeros(points.shape[0], dtype=np.float32)
        for view_index in range(projections.shape[0]):
            source = sources[view_index]
            detector = detectors[view_index]
            normal = (detector - source) / float(sid_m)
            rays = points - source
            denominator = rays @ normal
            valid = denominator > 0.0
            intersection = np.empty_like(points)
            intersection[:] = np.nan
            scale = np.zeros_like(denominator)
            scale[valid] = float(sid_m) / denominator[valid]
            valid &= scale > 0.0
            intersection[valid] = source + scale[valid, None] * rays[valid]
            relative = intersection - detector
            columns = relative @ u_axes[view_index] / spacing_m + (detector_width - 1) / 2.0
            rows = relative @ v_axes[view_index] / spacing_m + (detector_height - 1) / 2.0
            values = _sample_detector(
                binary_images[view_index].astype(np.float32, copy=False),
                rows,
                columns,
                interpolation=interpolation,
            )
            values[~valid] = 0.0
            accumulated += values > 0.0
        if combine == "mean":
            accumulated /= projections.shape[0]
        elif combine == "union":
            accumulated = (accumulated > 0.0).astype(np.float32)
        output[start:stop] = accumulated.reshape(stop - start, height, width)
    return output

def resample_binary_xyz_to_grid(
    volume_xyz: np.ndarray,
    spacing_xyz_mm: Sequence[float],
    grid: GridSpec,
) -> np.ndarray:
    """Nearest-neighbour sample an identity-oriented XYZ mask onto a ZYX grid."""

    volume = np.asarray(volume_xyz)
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64).reshape(3)
    if volume.ndim != 3:
        raise ValueError(f"Ground-truth volume must be 3D XYZ, got {volume.shape}.")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("Ground-truth spacing must contain three positive values.")
    if np.issubdtype(volume.dtype, np.floating):
        if not np.isfinite(volume).all():
            raise ValueError("Ground-truth volume contains NaN or infinity.")
        if not np.allclose(volume, np.rint(volume), atol=1.0e-6):
            raise ValueError("Floating ground truth must contain integer-like labels.")
    if np.any(volume < 0):
        raise ValueError("Ground-truth labels must be non-negative.")
    mask = volume > 0

    origin = np.asarray(grid.origin_xyz_mm, dtype=np.float64)
    target_spacing = np.asarray(grid.spacing_xyz_mm, dtype=np.float64)
    depth, height, width = grid.shape_zyx
    target_x = origin[0] + np.arange(width) * target_spacing[0]
    target_y = origin[1] + np.arange(height) * target_spacing[1]
    target_z = origin[2] + np.arange(depth) * target_spacing[2]
    source_indices = [
        np.rint(axis / step).astype(np.int64)
        for axis, step in zip((target_x, target_y, target_z), spacing)
    ]
    valid = [
        (indices >= 0) & (indices < size)
        for indices, size in zip(source_indices, volume.shape)
    ]
    output = np.zeros((depth, height, width), dtype=np.float32)
    if not all(np.any(item) for item in valid):
        return output
    x_positions, y_positions, z_positions = (np.flatnonzero(item) for item in valid)
    x_indices, y_indices, z_indices = (
        indices[item] for indices, item in zip(source_indices, valid)
    )
    sampled_xyz = mask[np.ix_(x_indices, y_indices, z_indices)]
    output[np.ix_(z_positions, y_positions, x_positions)] = sampled_xyz.transpose(2, 1, 0)
    return output
