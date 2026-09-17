"""Visualization-only vascular geometry derived from DeepCA prediction volumes."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .metrics import PhysicalGrid, skeletonization_metadata


@dataclass(frozen=True)
class PredictionVolume:
    """One saved prediction and its physical ZYX grid metadata."""

    path: Path
    case_id: str
    volume_zyx: np.ndarray
    grid: PhysicalGrid
    threshold: float
    view_indices: tuple[int, ...]


@dataclass(frozen=True)
class VascularGeometry:
    """Centerline graph, EDT radii, and triangle surface from a binary volume."""

    foreground_voxels: int
    node_index_zyx: np.ndarray
    node_xyz_mm: np.ndarray
    node_radius_mm: np.ndarray
    edge_node_indices: np.ndarray
    node_degree: np.ndarray
    node_kind: np.ndarray
    component_id: np.ndarray
    surface_vertices_xyz_mm: np.ndarray
    surface_faces: np.ndarray
    surface_vertex_radius_mm: np.ndarray

    @property
    def component_count(self) -> int:
        return int(self.component_id.max()) + 1 if self.component_id.size else 0


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> Any:
    value = np.asarray(archive[key])
    if value.dtype.hasobject:
        raise TypeError(f"{key!r} uses unsafe object dtype.")
    if value.size != 1:
        raise ValueError(f"{key!r} must be scalar, got {value.shape}.")
    return value.reshape(()).item()


def _vector(
    archive: np.lib.npyio.NpzFile,
    keys: Sequence[str],
    *,
    label: str,
    positive: bool,
) -> tuple[float, float, float]:
    key = next((candidate for candidate in keys if candidate in archive.files), None)
    if key is None:
        raise KeyError(f"Prediction archive is missing {label}; tried {list(keys)}.")
    array = np.asarray(archive[key], dtype=np.float64)
    if array.size == 1 and label == "spacing_xyz_mm":
        array = np.repeat(array.reshape(1), 3)
    array = array.reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{key!r} must contain three finite XYZ values.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{key!r} must contain positive XYZ values.")
    return tuple(float(value) for value in array)


def load_prediction_volume(
    path: str | Path,
    *,
    threshold: Optional[float] = None,
) -> PredictionVolume:
    """Load a DeepCA prediction NPZ, with limited AutoCar-key compatibility."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        volume_key = next(
            (
                key
                for key in ("vol", "prediction_volume_zyx", "prediction")
                if key in archive.files
            ),
            None,
        )
        if volume_key is None:
            raise KeyError(
                f"{source} has no prediction volume; expected 'vol' or "
                "'prediction_volume_zyx'."
            )
        volume = np.asarray(archive[volume_key])
        if volume.dtype.hasobject or volume.ndim != 3 or min(volume.shape) <= 0:
            raise ValueError(f"{source}:{volume_key} must be a non-empty numeric 3D array.")
        if not (
            np.issubdtype(volume.dtype, np.number)
            or np.issubdtype(volume.dtype, np.bool_)
        ) or not np.isfinite(volume).all():
            raise ValueError(f"{source}:{volume_key} must contain finite numeric values.")

        axis_key = next(
            (key for key in ("axis_order", "volume_axis_order") if key in archive.files),
            None,
        )
        if axis_key is None:
            raise KeyError(f"{source} is missing explicit volume axis-order metadata.")
        axis_order = str(_scalar(archive, axis_key)).strip().upper()
        if axis_order != "ZYX":
            raise ValueError(
                f"{source}:{axis_key} must declare ZYX, got {axis_order!r}."
            )

        spacing = _vector(
            archive,
            ("spacing", "spacing_xyz_mm", "voxel_size_mm"),
            label="spacing_xyz_mm",
            positive=True,
        )
        origin_keys = ("origin", "origin_xyz_mm", "bbox_min_xyz_mm")
        origin_source = next(
            (key for key in origin_keys if key in archive.files),
            None,
        )
        origin = _vector(
            archive,
            origin_keys,
            label="origin_xyz_mm",
            positive=False,
        )
        if origin_source == "bbox_min_xyz_mm":
            origin = tuple(
                lower_boundary + 0.5 * step
                for lower_boundary, step in zip(origin, spacing)
            )
        case_id = (
            str(_scalar(archive, "case_id")).strip()
            if "case_id" in archive.files
            else source.stem
        )
        if not case_id:
            raise ValueError("Prediction case_id must not be empty.")
        if threshold is None:
            threshold_key = next(
                (
                    key
                    for key in ("threshold", "prediction_threshold")
                    if key in archive.files
                ),
                None,
            )
            resolved_threshold = (
                float(_scalar(archive, threshold_key))
                if threshold_key is not None
                else 0.5
            )
        else:
            resolved_threshold = float(threshold)
        if not np.isfinite(resolved_threshold):
            raise ValueError("Prediction threshold must be finite.")
        if "view_indices" in archive.files:
            indices_array = np.asarray(archive["view_indices"])
            if not np.issubdtype(indices_array.dtype, np.integer):
                raise ValueError("view_indices must contain integers.")
            view_indices = tuple(int(value) for value in indices_array.reshape(-1))
        else:
            view_indices = ()

    grid = PhysicalGrid(
        shape_zyx=tuple(int(value) for value in volume.shape),
        spacing_xyz_mm=spacing,
        origin_xyz_mm=origin,
    )
    return PredictionVolume(
        path=source,
        case_id=case_id,
        volume_zyx=np.ascontiguousarray(volume),
        grid=grid,
        threshold=resolved_threshold,
        view_indices=view_indices,
    )


def _tight_padded_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.argwhere(mask)
    minimum = occupied.min(axis=0).astype(np.int64)
    maximum = (occupied.max(axis=0) + 1).astype(np.int64)
    slices = tuple(slice(int(low), int(high)) for low, high in zip(minimum, maximum))
    return np.pad(mask[slices], 1, mode="constant", constant_values=False), minimum


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    try:
        supports_method = "method" in inspect.signature(skeletonize).parameters
    except (TypeError, ValueError):
        supports_method = False
    result = skeletonize(mask, method="lee") if supports_method else skeletonize(mask)
    result = np.asarray(result, dtype=bool)
    if result.shape != mask.shape:
        raise RuntimeError(f"Skeletonization changed shape {mask.shape} to {result.shape}.")
    return result


def _overlap_slices(
    shape: Sequence[int], offset: Sequence[int]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    first: list[slice] = []
    second: list[slice] = []
    for size, delta in zip(shape, offset):
        if delta >= 0:
            first.append(slice(0, size - delta))
            second.append(slice(delta, size))
        else:
            first.append(slice(-delta, size))
            second.append(slice(0, size + delta))
    return tuple(first), tuple(second)


def _graph_edges(skeleton: np.ndarray, node_indices: np.ndarray) -> np.ndarray:
    if node_indices.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    node_ids = np.full(skeleton.shape, -1, dtype=np.int32)
    node_ids[tuple(node_indices.T)] = np.arange(len(node_indices), dtype=np.int32)
    edges: list[np.ndarray] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) == (0, 0, 0):
                    continue
                if not (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0)):
                    continue
                source, target = _overlap_slices(skeleton.shape, (dz, dy, dx))
                connected = skeleton[source] & skeleton[target]
                if np.any(connected):
                    edges.append(
                        np.column_stack(
                            (node_ids[source][connected], node_ids[target][connected])
                        ).astype(np.int32)
                    )
    return np.concatenate(edges, axis=0) if edges else np.empty((0, 2), dtype=np.int32)


def extract_vascular_geometry(
    volume_zyx: np.ndarray,
    *,
    threshold: float,
    grid: PhysicalGrid,
) -> VascularGeometry:
    """Derive a 26-neighbour skeleton/radius graph and marching-cubes surface."""

    volume = np.asarray(volume_zyx)
    if volume.shape != grid.shape_zyx:
        raise ValueError(
            f"Prediction shape {volume.shape} does not match grid {grid.shape_zyx}."
        )
    if not np.isfinite(volume).all():
        raise ValueError("Prediction volume contains NaN or infinity.")
    cutoff = float(threshold)
    if not np.isfinite(cutoff):
        raise ValueError("threshold must be finite.")
    mask = volume >= cutoff
    foreground = int(np.count_nonzero(mask))
    if foreground == 0:
        return VascularGeometry(
            foreground_voxels=0,
            node_index_zyx=np.empty((0, 3), dtype=np.int32),
            node_xyz_mm=np.empty((0, 3), dtype=np.float32),
            node_radius_mm=np.empty((0,), dtype=np.float32),
            edge_node_indices=np.empty((0, 2), dtype=np.int32),
            node_degree=np.empty((0,), dtype=np.int16),
            node_kind=np.empty((0,), dtype=np.int8),
            component_id=np.empty((0,), dtype=np.int32),
            surface_vertices_xyz_mm=np.empty((0, 3), dtype=np.float32),
            surface_faces=np.empty((0, 3), dtype=np.int32),
            surface_vertex_radius_mm=np.empty((0,), dtype=np.float32),
        )

    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes

    padded, crop_min = _tight_padded_mask(mask)
    skeleton = _skeletonize(padded)
    spacing_xyz = np.asarray(grid.spacing_xyz_mm, dtype=np.float64)
    spacing_zyx = spacing_xyz[::-1]
    radii = ndimage.distance_transform_edt(
        padded, sampling=tuple(float(value) for value in spacing_zyx)
    )
    padded_nodes = np.argwhere(skeleton).astype(np.int32)
    if padded_nodes.size == 0:
        fallback = np.asarray(
            [np.unravel_index(int(np.argmax(radii)), radii.shape)], dtype=np.int32
        )
        padded_nodes = fallback
        skeleton[tuple(fallback[0])] = True
    edges = _graph_edges(skeleton, padded_nodes)
    degree = np.zeros(len(padded_nodes), dtype=np.int16)
    if edges.size:
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)
    node_kind = np.where(
        degree == 0,
        0,
        np.where(degree == 1, 1, np.where(degree == 2, 2, 3)),
    ).astype(np.int8)
    component_labels, _ = ndimage.label(
        skeleton, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    component_id = component_labels[tuple(padded_nodes.T)].astype(np.int32) - 1
    node_radius = radii[tuple(padded_nodes.T)].astype(np.float32)
    nodes_zyx = padded_nodes + crop_min.astype(np.int32)[None, :] - 1
    origin_xyz = np.asarray(grid.origin_xyz_mm, dtype=np.float64)
    node_xyz = (
        origin_xyz[None, :]
        + nodes_zyx[:, ::-1].astype(np.float64) * spacing_xyz[None, :]
    ).astype(np.float32)

    vertices_zyx_mm, faces, _, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        spacing=tuple(float(value) for value in spacing_zyx),
        allow_degenerate=False,
    )
    padded_zero_zyx_mm = (crop_min.astype(np.float64) - 1.0) * spacing_zyx
    vertices_xyz = (
        vertices_zyx_mm[:, ::-1]
        + padded_zero_zyx_mm[::-1][None, :]
        + origin_xyz[None, :]
    ).astype(np.float32)
    surface_faces = np.asarray(faces, dtype=np.int32)
    _, nearest = cKDTree(node_xyz).query(vertices_xyz, k=1, workers=-1)
    vertex_radius = node_radius[np.asarray(nearest, dtype=np.int64)].astype(np.float32)
    return VascularGeometry(
        foreground_voxels=foreground,
        node_index_zyx=nodes_zyx.astype(np.int32),
        node_xyz_mm=node_xyz,
        node_radius_mm=node_radius,
        edge_node_indices=edges,
        node_degree=degree,
        node_kind=node_kind,
        component_id=component_id,
        surface_vertices_xyz_mm=vertices_xyz,
        surface_faces=surface_faces,
        surface_vertex_radius_mm=vertex_radius,
    )


def _radius_colors(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    radii = np.asarray(values, dtype=np.float32)
    minimum = float(radii.min()) if radii.size else 0.0
    maximum = float(radii.max()) if radii.size else 1.0
    if maximum <= minimum:
        maximum = minimum + max(abs(minimum) * 0.05, 1.0e-3)
    normalized = np.clip((radii - minimum) / (maximum - minimum), 0.0, 1.0)
    anchors = np.asarray(
        ((215, 48, 39), (254, 224, 139), (26, 152, 80)), dtype=np.float32
    )
    lower = normalized <= 0.5
    interpolation = np.where(lower, normalized * 2.0, (normalized - 0.5) * 2.0)
    colors = np.empty((len(radii), 3), dtype=np.float32)
    colors[lower] = anchors[0] + interpolation[lower, None] * (anchors[1] - anchors[0])
    colors[~lower] = anchors[1] + interpolation[~lower, None] * (anchors[2] - anchors[1])
    return np.rint(colors).astype(np.uint8), minimum, maximum


def _save_graph(path: Path, geometry: VascularGeometry, prediction: PredictionVolume) -> None:
    np.savez_compressed(
        path,
        representation=np.asarray("voxel_skeleton_graph"),
        derivation=np.asarray("threshold_lee_skeleton_26n_edt_radius"),
        coordinate_frame=np.asarray("deepca_model_grid_xyz_mm"),
        node_index_axis_order=np.asarray("ZYX"),
        node_index_zyx=geometry.node_index_zyx,
        node_xyz_mm=geometry.node_xyz_mm,
        node_radius_mm=geometry.node_radius_mm,
        edge_node_indices=geometry.edge_node_indices,
        node_degree=geometry.node_degree,
        node_kind=geometry.node_kind,
        node_kind_labels=np.asarray(("isolated", "endpoint", "regular", "junction")),
        component_id=geometry.component_id,
        source_volume_shape_zyx=np.asarray(prediction.grid.shape_zyx, dtype=np.int32),
        spacing_xyz_mm=np.asarray(prediction.grid.spacing_xyz_mm, dtype=np.float32),
        origin_xyz_mm=np.asarray(prediction.grid.origin_xyz_mm, dtype=np.float32),
        prediction_threshold=np.asarray(prediction.threshold, dtype=np.float32),
        foreground_voxels=np.asarray(geometry.foreground_voxels, dtype=np.int64),
    )


def _save_surface_ply(path: Path, geometry: VascularGeometry) -> None:
    vertices = geometry.surface_vertices_xyz_mm
    faces = geometry.surface_faces
    if not len(vertices) or not len(faces):
        raise ValueError("Cannot write a surface mesh for an empty prediction.")
    colors, _, _ = _radius_colors(geometry.surface_vertex_radius_mm)
    vertex_type = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("radius_mm", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertex_rows = np.empty(len(vertices), dtype=vertex_type)
    for axis, name in enumerate(("x", "y", "z")):
        vertex_rows[name] = vertices[:, axis]
    vertex_rows["radius_mm"] = geometry.surface_vertex_radius_mm
    vertex_rows["red"], vertex_rows["green"], vertex_rows["blue"] = colors.T
    face_type = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_rows = np.empty(len(faces), dtype=face_type)
    face_rows["count"] = 3
    face_rows["indices"] = faces
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment DeepCA visualization-only occupancy surface\n"
        "comment coordinate_frame deepca_model_grid_xyz_mm\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float radius_mm\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertex_rows.tofile(stream)
        face_rows.tofile(stream)


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _equal_axes(axis: Any, points: np.ndarray) -> None:
    if not points.size:
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_zlim(-1.0, 1.0)
        return
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float(np.max(maximum - minimum)) * 0.55, 0.5)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _plot_geometry(axis: Any, geometry: VascularGeometry, maximum_elements: int) -> Any:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    _, radius_min, radius_max = _radius_colors(geometry.node_radius_mm)
    colormap = LinearSegmentedColormap.from_list(
        "deepca_radius", ("#d73027", "#fee08b", "#1a9850")
    )
    normalization = Normalize(vmin=radius_min, vmax=radius_max, clip=True)
    face_ids = _sample_indices(len(geometry.surface_faces), maximum_elements)
    faces = geometry.surface_faces[face_ids]
    if len(faces):
        face_radii = geometry.surface_vertex_radius_mm[faces].mean(axis=1)
        axis.add_collection3d(
            Poly3DCollection(
                geometry.surface_vertices_xyz_mm[faces],
                facecolors=colormap(normalization(face_radii)),
                edgecolors="none",
                alpha=0.66,
            )
        )
    edge_ids = _sample_indices(len(geometry.edge_node_indices), maximum_elements)
    edges = geometry.edge_node_indices[edge_ids]
    if len(edges):
        axis.add_collection3d(
            Line3DCollection(
                geometry.node_xyz_mm[edges],
                colors="#17202a",
                linewidths=0.7,
                alpha=0.94,
            )
        )
    critical = np.flatnonzero(geometry.node_kind != 2)
    critical = critical[_sample_indices(len(critical), min(maximum_elements, 3000))]
    if len(critical):
        axis.scatter(
            geometry.node_xyz_mm[critical, 0],
            geometry.node_xyz_mm[critical, 1],
            geometry.node_xyz_mm[critical, 2],
            c=geometry.node_radius_mm[critical],
            cmap=colormap,
            norm=normalization,
            s=5.0,
            edgecolors="#111111",
            linewidths=0.25,
            depthshade=False,
        )
    _equal_axes(axis, geometry.surface_vertices_xyz_mm)
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    scalar = ScalarMappable(norm=normalization, cmap=colormap)
    scalar.set_array(geometry.node_radius_mm)
    return scalar


def _save_surface_image(
    path: Path,
    geometry: VascularGeometry,
    *,
    case_id: str,
    maximum_elements: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(7.2, 6.4), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    if geometry.foreground_voxels:
        scalar = _plot_geometry(axis, geometry, maximum_elements)
        figure.colorbar(scalar, ax=axis, shrink=0.64, pad=0.02).set_label(
            "Estimated centerline radius (mm)"
        )
        axis.view_init(elev=24.0, azim=35.0)
    else:
        axis.text2D(0.5, 0.5, "No predicted foreground", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title(f"{case_id}: predicted vascular surface and centerline")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_surface_gif(
    path: Path,
    geometry: VascularGeometry,
    *,
    case_id: str,
    frames: int,
    fps: int,
    maximum_elements: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    figure = plt.figure(figsize=(7.0, 6.4))
    axis = figure.add_subplot(111, projection="3d")
    scalar = _plot_geometry(axis, geometry, maximum_elements)
    figure.colorbar(scalar, ax=axis, shrink=0.64, pad=0.02).set_label(
        "Estimated centerline radius (mm)"
    )
    axis.set_title(f"{case_id}: predicted vascular surface and centerline")

    def rotate(frame: int) -> tuple[Any, ...]:
        phase = 2.0 * np.pi * frame / frames
        axis.view_init(elev=24.0 + 5.0 * np.sin(phase), azim=360.0 * frame / frames)
        return (axis,)

    animation = FuncAnimation(figure, rotate, frames=frames, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(figure)


def _slice_rgb(prediction: np.ndarray, target: Optional[np.ndarray]) -> np.ndarray:
    predicted = np.asarray(prediction, dtype=bool)
    if target is None:
        rgb = np.zeros((*predicted.shape, 3), dtype=np.float32)
        rgb[..., 0] = predicted
        rgb[..., 2] = predicted * 0.72
        return rgb
    reference = np.asarray(target, dtype=bool)
    rgb = np.zeros((*predicted.shape, 3), dtype=np.float32)
    rgb[..., 0] = predicted
    rgb[..., 2] = predicted * 0.72
    rgb[..., 1] = reference
    return rgb


def _save_orthogonal_comparison(
    path: Path,
    prediction_mask: np.ndarray,
    target_zyx: Optional[np.ndarray],
    *,
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if target_zyx is not None and target_zyx.shape != prediction_mask.shape:
        raise ValueError(
            f"Aligned target shape {target_zyx.shape} differs from prediction "
            f"{prediction_mask.shape}."
        )
    union = prediction_mask if target_zyx is None else prediction_mask | target_zyx
    occupied = np.argwhere(union)
    center = (
        np.rint(occupied.mean(axis=0)).astype(int)
        if occupied.size
        else np.asarray(prediction_mask.shape) // 2
    )
    z, y, x = (int(value) for value in center)
    slices = (
        (
            "Axial (XY)",
            prediction_mask[z],
            None if target_zyx is None else target_zyx[z],
        ),
        (
            "Coronal (XZ)",
            prediction_mask[:, y, :],
            None if target_zyx is None else target_zyx[:, y, :],
        ),
        (
            "Sagittal (YZ)",
            prediction_mask[:, :, x],
            None if target_zyx is None else target_zyx[:, :, x],
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for axis, (title, predicted_slice, target_slice) in zip(axes, slices):
        axis.imshow(_slice_rgb(predicted_slice, target_slice), origin="lower")
        axis.set_title(title)
        axis.set_axis_off()
    legend = [Patch(color=(1.0, 0.0, 0.72), label="prediction")]
    if target_zyx is not None:
        legend.append(Patch(color=(0.0, 1.0, 0.0), label="ground truth"))
    figure.legend(handles=legend, loc="lower center", ncol=len(legend))
    figure.suptitle(f"{case_id}: orthogonal volume comparison")
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_input_views(
    path: Path,
    images: np.ndarray,
    *,
    view_indices: Sequence[int],
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = np.asarray(images)
    if selected.ndim != 3 or not len(selected):
        raise ValueError(f"Projection images must have shape [V,H,W], got {selected.shape}.")
    figure, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), dpi=150)
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for position, (axis, image) in enumerate(zip(axes_array, selected)):
        axis.imshow(image, cmap="gray", origin="upper")
        source_index = view_indices[position] if position < len(view_indices) else position
        axis.set_title(f"View {source_index}")
        axis.set_axis_off()
    figure.suptitle(f"{case_id}: selected projection masks")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_prediction_visualization(
    output_dir: str | Path,
    prediction: PredictionVolume,
    *,
    target_zyx: Optional[np.ndarray] = None,
    projection_images: Optional[np.ndarray] = None,
    ground_truth_path: Optional[str | Path] = None,
    projection_path: Optional[str | Path] = None,
    gif_frames: int = 24,
    gif_fps: int = 6,
    maximum_elements: int = 20_000,
) -> dict[str, Any]:
    """Write an auditable AutoCar-like visualization bundle for one prediction."""

    frames = int(gif_frames)
    fps = int(gif_fps)
    maximum = int(maximum_elements)
    if frames < 0 or fps <= 0 or maximum <= 0:
        raise ValueError("gif_frames must be >= 0; gif_fps and maximum_elements must be > 0.")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    mask = np.asarray(prediction.volume_zyx >= prediction.threshold, dtype=bool)
    target = None if target_zyx is None else np.asarray(target_zyx, dtype=bool)
    geometry = extract_vascular_geometry(
        prediction.volume_zyx,
        threshold=prediction.threshold,
        grid=prediction.grid,
    )
    graph_name = "predicted_centerline_graph.npz"
    surface_name = "predicted_radius_surface.ply"
    surface_image_name = "predicted_surface_centerline_radius.png"
    surface_gif_name = (
        "predicted_surface_centerline_radius.gif"
        if frames > 0 and geometry.foreground_voxels > 0
        else None
    )
    comparison_name = "volume_comparison.png"
    input_views_name = "input_views.png" if projection_images is not None else None
    _save_graph(destination / graph_name, geometry, prediction)
    if geometry.foreground_voxels:
        _save_surface_ply(destination / surface_name, geometry)
    _save_surface_image(
        destination / surface_image_name,
        geometry,
        case_id=prediction.case_id,
        maximum_elements=maximum,
    )
    if surface_gif_name is not None:
        _save_surface_gif(
            destination / surface_gif_name,
            geometry,
            case_id=prediction.case_id,
            frames=frames,
            fps=fps,
            maximum_elements=maximum,
        )
    _save_orthogonal_comparison(
        destination / comparison_name,
        mask,
        target,
        case_id=prediction.case_id,
    )
    if projection_images is not None:
        _save_input_views(
            destination / str(input_views_name),
            projection_images,
            view_indices=prediction.view_indices,
            case_id=prediction.case_id,
        )
    manifest = {
        "schema_version": 1,
        "case_id": prediction.case_id,
        "prediction_path": str(prediction.path),
        "prediction_axis_order": "ZYX",
        "prediction_grid": prediction.grid.as_dict(),
        "prediction_threshold": prediction.threshold,
        "derivation": "lee_skeleton_26n_edt_radius_and_marching_cubes",
        "quantitative_metrics_use_this_postprocessing": False,
        "coordinate_frame": "deepca_model_grid_xyz_mm",
        "skeletonization": skeletonization_metadata(),
        "foreground_voxels": geometry.foreground_voxels,
        "graph_nodes": int(len(geometry.node_xyz_mm)),
        "graph_edges": int(len(geometry.edge_node_indices)),
        "graph_components": geometry.component_count,
        "surface_vertices": int(len(geometry.surface_vertices_xyz_mm)),
        "surface_faces": int(len(geometry.surface_faces)),
        "centerline_graph": graph_name,
        "radius_colored_surface_ply": (
            surface_name if geometry.foreground_voxels else None
        ),
        "surface_centerline_radius_image": surface_image_name,
        "surface_centerline_radius_gif": surface_gif_name,
        "volume_comparison": comparison_name,
        "ground_truth_overlay_included": target is not None,
        "ground_truth_path": (
            None
            if ground_truth_path is None
            else str(Path(ground_truth_path).expanduser().resolve())
        ),
        "input_views": input_views_name,
        "projection_path": (
            None
            if projection_path is None
            else str(Path(projection_path).expanduser().resolve())
        ),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "PredictionVolume",
    "VascularGeometry",
    "extract_vascular_geometry",
    "load_prediction_volume",
    "save_prediction_visualization",
]
