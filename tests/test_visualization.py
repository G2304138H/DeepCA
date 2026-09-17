from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deepca.metrics import PhysicalGrid
from deepca.visualization import (
    extract_vascular_geometry,
    load_prediction_volume,
    save_prediction_visualization,
)
from scripts.visualize_prediction import _read_evaluation_row, main


def _write_prediction(path: Path, volume: np.ndarray) -> None:
    np.savez_compressed(
        path,
        vol=volume,
        spacing=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        origin=np.asarray((-4.0, -4.0, -4.0), dtype=np.float32),
        axis_order=np.asarray("ZYX"),
        case_id=np.asarray("lca_0001"),
        view_indices=np.asarray((1, 3), dtype=np.int32),
        threshold=np.asarray(0.5, dtype=np.float32),
    )


class PredictionVolumeTestCase(unittest.TestCase):
    def test_loads_saved_deepca_prediction_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            volume = np.zeros((9, 9, 9), dtype=np.uint8)
            volume[2:7, 3:6, 3:6] = 1
            _write_prediction(path, volume)
            prediction = load_prediction_volume(path)
        self.assertEqual(prediction.case_id, "lca_0001")
        self.assertEqual(prediction.volume_zyx.shape, (9, 9, 9))
        self.assertEqual(prediction.grid.spacing_xyz_mm, (1.0, 1.0, 1.0))
        self.assertEqual(prediction.grid.origin_xyz_mm, (-4.0, -4.0, -4.0))
        self.assertEqual(prediction.view_indices, (1, 3))
        self.assertEqual(prediction.threshold, 0.5)

    def test_requires_explicit_zyx_axis_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            np.savez_compressed(
                path,
                vol=np.zeros((3, 3, 3), dtype=np.uint8),
                spacing=np.ones(3),
                origin=np.zeros(3),
            )
            with self.assertRaisesRegex(KeyError, "axis-order"):
                load_prediction_volume(path)

    def test_autocar_boundary_origin_is_converted_to_voxel_center(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            np.savez_compressed(
                path,
                prediction_volume_zyx=np.zeros((3, 3, 3), dtype=np.uint8),
                voxel_size_mm=np.asarray((2.0, 4.0, 6.0)),
                bbox_min_xyz_mm=np.asarray((-5.0, -6.0, -7.0)),
                volume_axis_order=np.asarray("ZYX"),
            )
            prediction = load_prediction_volume(path)
        self.assertEqual(prediction.grid.origin_xyz_mm, (-4.0, -4.0, -4.0))


class VascularGeometryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.volume = np.zeros((9, 9, 9), dtype=np.uint8)
        self.volume[2:7, 3:6, 3:6] = 1
        self.grid = PhysicalGrid(
            shape_zyx=self.volume.shape,
            spacing_xyz_mm=(1.0, 1.0, 1.0),
            origin_xyz_mm=(-4.0, -4.0, -4.0),
        )

    def test_surface_uses_deepca_voxel_center_origin(self) -> None:
        geometry = extract_vascular_geometry(
            self.volume,
            threshold=0.5,
            grid=self.grid,
        )
        self.assertEqual(geometry.foreground_voxels, 45)
        self.assertGreater(len(geometry.node_xyz_mm), 0)
        self.assertGreater(len(geometry.edge_node_indices), 0)
        self.assertGreater(len(geometry.surface_faces), 0)
        np.testing.assert_allclose(
            geometry.surface_vertices_xyz_mm.min(axis=0),
            (-1.5, -1.5, -2.5),
        )
        np.testing.assert_allclose(
            geometry.surface_vertices_xyz_mm.max(axis=0),
            (1.5, 1.5, 2.5),
        )
        self.assertTrue(np.all(geometry.node_radius_mm > 0.0))

    def test_empty_prediction_has_no_mesh_or_graph(self) -> None:
        geometry = extract_vascular_geometry(
            np.zeros_like(self.volume),
            threshold=0.5,
            grid=self.grid,
        )
        self.assertEqual(geometry.foreground_voxels, 0)
        self.assertEqual(geometry.node_xyz_mm.shape, (0, 3))
        self.assertEqual(geometry.surface_faces.shape, (0, 3))


class VisualizationBundleTestCase(unittest.TestCase):
    def test_bundle_contains_graph_mesh_images_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "prediction.npz"
            volume = np.zeros((9, 9, 9), dtype=np.uint8)
            volume[2:7, 3:6, 3:6] = 1
            _write_prediction(prediction_path, volume)
            prediction = load_prediction_volume(prediction_path)
            projection_path = root / "projection.npz"
            np.savez_compressed(
                projection_path,
                images=np.zeros((4, 8, 8), dtype=np.uint8),
            )
            output = root / "visualization"
            manifest = save_prediction_visualization(
                output,
                prediction,
                target_zyx=volume,
                projection_images=np.zeros((2, 8, 8), dtype=np.uint8),
                ground_truth_path=root / "ground_truth.npz",
                projection_path=projection_path,
                gif_frames=2,
                gif_fps=1,
                maximum_elements=200,
            )
            expected = {
                "predicted_centerline_graph.npz",
                "predicted_radius_surface.ply",
                "predicted_surface_centerline_radius.png",
                "predicted_surface_centerline_radius.gif",
                "volume_comparison.png",
                "input_views.png",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            saved = json.loads((output / "manifest.json").read_text())
            self.assertEqual(saved, manifest)
            self.assertFalse(saved["quantitative_metrics_use_this_postprocessing"])
            self.assertTrue(saved["ground_truth_overlay_included"])
            self.assertEqual(saved["graph_components"], 1)
            self.assertGreater(saved["surface_vertices"], 0)
            with np.load(output / "predicted_centerline_graph.npz") as graph:
                self.assertEqual(str(graph["coordinate_frame"]), "deepca_model_grid_xyz_mm")
                self.assertEqual(str(graph["node_index_axis_order"]), "ZYX")

    def test_cli_selects_evaluation_case_and_disables_gif(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation" / "test"
            prediction_path = evaluation / "predictions" / "views_2" / "lca_0001.npz"
            prediction_path.parent.mkdir(parents=True)
            volume = np.zeros((9, 9, 9), dtype=np.uint8)
            volume[2:7, 3:6, 3:6] = 1
            _write_prediction(prediction_path, volume)
            payload = {
                "context": {},
                "results": [
                    {
                        "case_id": "lca_0001",
                        "num_views": 2,
                        "prediction_path": str(prediction_path),
                        "ground_truth_path": None,
                        "projection_path": None,
                    }
                ],
            }
            (evaluation / "per_case.json").write_text(json.dumps(payload))
            output = root / "rendered"
            result = main(
                [
                    "--evaluation-dir",
                    str(evaluation),
                    "--case-id",
                    "lca_0001",
                    "--num-views",
                    "2",
                    "--output-dir",
                    str(output),
                    "--gif-frames",
                    "0",
                    "--max-elements",
                    "100",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertFalse((output / "predicted_surface_centerline_radius.gif").exists())

    def test_evaluation_row_requires_disambiguation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"case_id": "rca_0001", "num_views": 1, "prediction_path": "one.npz"},
                {"case_id": "rca_0001", "num_views": 2, "prediction_path": "two.npz"},
            ]
            (root / "per_case.json").write_text(json.dumps({"results": rows}))
            with self.assertRaisesRegex(ValueError, "disambiguate"):
                _read_evaluation_row(
                    root,
                    case_id="rca_0001",
                    num_views=None,
                    condition_id=None,
                )


if __name__ == "__main__":
    unittest.main()
