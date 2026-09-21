from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from deepca.data import (
    CasePair,
    ImageCASDataset,
    build_projection_index,
    load_ground_truth_npz,
    load_projection_npz,
    resolve_case_pairs,
)
from deepca.geometry import (
    GridSpec,
    binary_cone_backproject,
    make_cubic_grid,
    resample_binary_xyz_to_grid,
)

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None


class ImageCASDataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_projection(
        self,
        path: Path,
        *,
        vessel: str = "rca",
        number: int = 1,
        stored_calibration: bool = False,
        include_object_metadata: bool = False,
        image_size: int = 16,
        views: int = 2,
        center_m: tuple[float, float, float] = (0.0075, 0.0075, 0.0075),
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        values: dict[str, object] = {
            "sample_name": np.asarray(f"{vessel}_{number:04d}"),
            "source_relpath": np.asarray(f"{vessel}/{number}/prefix_02.npz"),
            "images": np.ones((views, image_size, image_size), dtype=np.float32),
            "theta_deg": np.linspace(0.0, 30.0, views, dtype=np.float32),
            "phi_deg": np.linspace(90.0, 70.0, views, dtype=np.float32),
            "projection_center_offset": np.asarray(center_m, dtype=np.float32),
        }
        theta = np.asarray(values["theta_deg"])
        phi = np.asarray(values["phi_deg"])
        values["view_features"] = np.stack(
            (
                np.sin(np.deg2rad(theta)),
                np.cos(np.deg2rad(theta)),
                np.sin(np.deg2rad(phi)),
                np.cos(np.deg2rad(phi)),
            ),
            axis=1,
        ).astype(np.float32)
        if stored_calibration:
            values.update(
                {
                    "sid": np.asarray(1.0, dtype=np.float32),
                    "imager_pixel_spacing": np.asarray(0.6, dtype=np.float32),
                    "imager_pixel_spacing_units": np.asarray("mm"),
                }
            )
        if include_object_metadata:
            values["source_case_id"] = np.asarray([str(number)], dtype=object)
        np.savez_compressed(path, **values)

    def write_gt(self, path: Path, *, shape: tuple[int, int, int] = (16, 16, 16)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        volume = np.zeros(shape, dtype=np.uint8)
        volume[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1
        np.savez_compressed(
            path,
            vol=volume,
            spacing=np.ones(3, dtype=np.float32),
        )

    def dataset_config(self, *, cache_enabled: bool = False) -> dict[str, object]:
        return {
            "experiment": {"seed": 3, "output_dir": str(self.root / "out")},
            "data": {
                "calibration": {
                    "fallback_sid_m": 0.9,
                    "fallback_detector_pixel_spacing_mm": 0.55,
                    "source_to_isocentre_m": 0.75,
                    "projection_center_offset_units": "m",
                },
                "ground_truth": {
                    "volume_key": "vol",
                    "spacing_key": "spacing",
                    "source_axis_order": "XYZ",
                    "require_nonempty_target": True,
                },
                "views": {
                    "count": 2,
                    "train_selection": "first",
                    "eval_selection": "first",
                    "indices": None,
                },
                "preprocessing": {
                    "volume_size": 32,
                    "fov_mode": "fixed",
                    "fov_mm": 32.0,
                    "projection_threshold": 0.0,
                    "backprojection_interpolation": "nearest",
                    "combine": "sum",
                    "chunk_depth": 4,
                },
                "cache": {
                    "enabled": cache_enabled,
                    "directory": str(self.root / "cache"),
                },
            },
        }

    def test_projection_loader_uses_fallbacks_and_skips_optional_object(self) -> None:
        path = self.root / "rca_0001.npz"
        self.write_projection(path, include_object_metadata=True)
        case = load_projection_npz(
            path,
            expected_case_id="rca_0001",
            expected_vessel="rca",
            fallback_sid_m=0.9,
            fallback_detector_pixel_spacing_mm=0.55,
        )
        self.assertEqual(case.images.shape, (2, 16, 16))
        self.assertEqual(case.sid_m, 0.9)
        self.assertEqual(case.sid_source, "fallback")
        self.assertEqual(case.detector_pixel_spacing_mm, 0.55)
        self.assertEqual(case.detector_pixel_spacing_source, "fallback")
        np.testing.assert_allclose(
            case.projection_center_offset_xyz_mm, (7.5, 7.5, 7.5)
        )

    def test_stored_calibration_precedes_fallback(self) -> None:
        path = self.root / "rca_0001.npz"
        self.write_projection(path, stored_calibration=True)
        case = load_projection_npz(
            path,
            expected_case_id="rca_0001",
            expected_vessel="rca",
            fallback_sid_m=0.9,
            fallback_detector_pixel_spacing_mm=0.55,
            expected_detector_pixel_spacing_mm=0.6,
        )
        self.assertAlmostEqual(case.sid_m, 1.0)
        self.assertEqual(case.sid_source, "archive")
        self.assertAlmostEqual(case.detector_pixel_spacing_mm, 0.6, places=6)
        self.assertEqual(case.detector_pixel_spacing_source, "archive")

    def test_inconsistent_stored_calibration_is_an_error(self) -> None:
        path = self.root / "rca_0001.npz"
        self.write_projection(path, stored_calibration=True)
        with self.assertRaisesRegex(ValueError, "authoritative"):
            load_projection_npz(
                path,
                expected_case_id="rca_0001",
                expected_vessel="rca",
                fallback_sid_m=0.9,
                fallback_detector_pixel_spacing_mm=0.55,
                expected_detector_pixel_spacing_mm=0.55,
            )

    def test_ground_truth_and_explicit_xyz_to_zyx(self) -> None:
        path = self.root / "1.npz"
        volume = np.zeros((2, 3, 4), dtype=np.uint8)
        volume[1, 2, 3] = 1
        np.savez(path, vol=volume, spacing=np.ones(3, dtype=np.float32))
        gt = load_ground_truth_npz(path)
        grid = GridSpec(
            shape_zyx=(4, 3, 2),
            spacing_xyz_mm=(1.0, 1.0, 1.0),
            center_xyz_mm=(0.5, 1.0, 1.5),
            origin_xyz_mm=(0.0, 0.0, 0.0),
        )
        result = resample_binary_xyz_to_grid(gt.volume_xyz, gt.spacing_xyz_mm, grid)
        np.testing.assert_array_equal(result, volume.transpose(2, 1, 0))
        self.assertEqual(result[3, 2, 1], 1)

    def test_pairing_requires_matching_vessel_and_exact_gt_path(self) -> None:
        projection_root = self.root / "projections"
        gt_root = self.root / "gt"
        self.write_projection(projection_root / "rca_0001.npz")
        self.write_gt(gt_root / "rca" / "1.npz")
        index = build_projection_index(projection_root, expected_vessel="rca")
        pairs, failures = resolve_case_pairs(
            ["rca_0001"],
            projection_root=projection_root,
            ground_truth_root=gt_root,
            vessel_type="rca",
            projection_index=index,
        )
        self.assertFalse(failures)
        self.assertEqual(pairs[0].ground_truth_path, (gt_root / "rca" / "1.npz").resolve())
        with self.assertRaisesRegex(RuntimeError, "not part"):
            resolve_case_pairs(
                ["lca_0001"],
                projection_root=projection_root,
                ground_truth_root=gt_root,
                vessel_type="rca",
                projection_index=index,
            )

    def test_binary_backprojection_maps_central_detector_pixel_to_isocentre(self) -> None:
        image = np.zeros((1, 5, 5), dtype=np.float32)
        image[0, 2, 2] = 1.0
        grid = make_cubic_grid(5, 5.0, (0.0, 0.0, 0.0))
        result = binary_cone_backproject(
            image,
            [0.0],
            [90.0],
            grid=grid,
            sid_m=0.9,
            source_to_isocentre_m=0.75,
            detector_pixel_spacing_mm=1.0,
        )
        self.assertEqual(result.shape, (5, 5, 5))
        self.assertEqual(result[2, 2, 2], 1.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_unsupported_configured_source_axis_order_is_rejected(self) -> None:
        projection = self.root / "rca_0001.npz"
        gt = self.root / "gt" / "rca" / "1.npz"
        self.write_projection(projection, image_size=32)
        self.write_gt(gt, shape=(32, 32, 32))
        pair = CasePair("rca_0001", "rca", 1, projection.resolve(), gt.resolve())
        config = self.dataset_config()
        config["data"]["ground_truth"]["source_axis_order"] = "ZYX"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "source_axis_order must be 'XYZ'"):
            ImageCASDataset([pair], config, training=False)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_cache_key_includes_ground_truth_interpretation(self) -> None:
        projection = self.root / "rca_0001.npz"
        gt = self.root / "gt" / "rca" / "1.npz"
        self.write_projection(
            projection,
            image_size=32,
            center_m=(0.0155, 0.0155, 0.0155),
        )
        gt.parent.mkdir(parents=True, exist_ok=True)
        first = np.zeros((32, 32, 32), dtype=np.uint8)
        second = np.zeros_like(first)
        first[15, 15, 15] = 1
        second[17, 16, 15] = 1
        np.savez_compressed(gt, vol=first, alternate=second, spacing=np.ones(3))
        pair = CasePair("rca_0001", "rca", 1, projection.resolve(), gt.resolve())

        first_config = self.dataset_config(cache_enabled=True)
        first_target = ImageCASDataset(
            [pair], first_config, training=False
        )[0]["target"]
        second_config = self.dataset_config(cache_enabled=True)
        second_config["data"]["ground_truth"]["volume_key"] = "alternate"  # type: ignore[index]
        second_target = ImageCASDataset(
            [pair], second_config, training=False
        )[0]["target"]

        self.assertFalse(torch.equal(first_target, second_target))
        self.assertEqual(len(list((self.root / "cache" / "rca").glob("*.npz"))), 2)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_timed_backprojection_bypasses_cached_input(self) -> None:
        projection = self.root / "rca_0001.npz"
        gt = self.root / "gt" / "rca" / "1.npz"
        self.write_projection(
            projection,
            image_size=32,
            center_m=(0.0155, 0.0155, 0.0155),
        )
        self.write_gt(gt, shape=(32, 32, 32))
        pair = CasePair("rca_0001", "rca", 1, projection.resolve(), gt.resolve())
        config = self.dataset_config(cache_enabled=True)

        ImageCASDataset([pair], config, training=False)[0]
        with mock.patch(
            "deepca.data.binary_cone_backproject",
            wraps=binary_cone_backproject,
        ) as backproject:
            item = ImageCASDataset(
                [pair],
                config,
                training=False,
                measure_backprojection_time=True,
            )[0]

        metadata = json.loads(item["metadata_json"])
        backproject.assert_called_once()
        self.assertTrue(metadata["backprojection_timed"])
        self.assertFalse(metadata["preprocessing_cache_used"])
        self.assertGreaterEqual(metadata["backprojection_seconds"], 0.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_image_replacement_uses_source_index_after_ordered_selection(self) -> None:
        projection = self.root / "rca_0001.npz"
        gt = self.root / "gt" / "rca" / "1.npz"
        self.write_projection(
            projection,
            image_size=32,
            views=3,
            center_m=(0.0155, 0.0155, 0.0155),
        )
        self.write_gt(gt, shape=(32, 32, 32))
        pair = CasePair("rca_0001", "rca", 1, projection.resolve(), gt.resolve())
        config = self.dataset_config()
        config["data"]["views"]["indices"] = [2, 0]  # type: ignore[index]

        replacement = np.zeros((32, 32), dtype=np.float64)
        dataset = ImageCASDataset(
            [pair],
            config,
            training=False,
            image_replacements={"RCA_0001": {np.int64(0): replacement}},
        )
        replacement.fill(7.0)  # The dataset owns a defensive copy.
        with mock.patch(
            "deepca.data.binary_cone_backproject",
            return_value=np.zeros((32, 32, 32), dtype=np.float32),
        ) as backproject:
            item = dataset[0]

        selected = backproject.call_args.args[0]
        np.testing.assert_array_equal(selected[0], np.ones((32, 32), dtype=np.float32))
        np.testing.assert_array_equal(selected[1], np.zeros((32, 32), dtype=np.float32))
        metadata = json.loads(item["metadata_json"])
        self.assertEqual(metadata["view_indices"], [2, 0])
        self.assertEqual(metadata["replaced_view_indices"], [0])
        self.assertEqual(metadata["theta_deg"], [30.0, 0.0])
        self.assertEqual(metadata["phi_deg"], [70.0, 90.0])
        self.assertEqual(
            sorted(metadata["replacement_images_sha256"]), ["0"]
        )
        self.assertEqual(len(metadata["replacement_images_sha256"]["0"]), 64)
        self.assertEqual(len(metadata["selected_images_sha256"]), 64)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_image_replacement_constructor_validation_is_strict(self) -> None:
        pair = CasePair(
            "rca_0001",
            "rca",
            1,
            (self.root / "rca_0001.npz").resolve(),
            (self.root / "1.npz").resolve(),
        )
        config = self.dataset_config()
        valid = np.zeros((32, 32), dtype=np.float32)
        invalid_replacements = (
            (
                {"rca_9999": {0: valid}},
                ValueError,
                "unknown case ID",
            ),
            (
                {"rca_0001": {"0": valid}},
                TypeError,
                "indices must be integers",
            ),
            (
                {"rca_0001": {-1: valid}},
                ValueError,
                "must be non-negative",
            ),
            (
                {"rca_0001": {0: np.zeros((2, 2, 2))}},
                ValueError,
                "must be a 2D detector image",
            ),
            (
                {"rca_0001": {0: np.full((32, 32), np.nan)}},
                ValueError,
                "contains NaN or infinity",
            ),
            (
                {"rca_0001": {0: np.full((32, 32), -1.0)}},
                ValueError,
                "contains negative values",
            ),
        )
        for replacements, error_type, message in invalid_replacements:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    ImageCASDataset(
                        [pair],
                        config,
                        training=False,
                        image_replacements=replacements,  # type: ignore[arg-type]
                    )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_image_replacement_validates_source_index_and_detector_shape(self) -> None:
        projection = self.root / "rca_0001.npz"
        gt = self.root / "gt" / "rca" / "1.npz"
        self.write_projection(
            projection,
            image_size=32,
            center_m=(0.0155, 0.0155, 0.0155),
        )
        self.write_gt(gt, shape=(32, 32, 32))
        pair = CasePair("rca_0001", "rca", 1, projection.resolve(), gt.resolve())
        config = self.dataset_config()

        outside = ImageCASDataset(
            [pair],
            config,
            training=False,
            image_replacements={"rca_0001": {2: np.zeros((32, 32))}},
        )
        with self.assertRaisesRegex(ValueError, "outside the available range"):
            outside[0]

        wrong_shape = ImageCASDataset(
            [pair],
            config,
            training=False,
            image_replacements={"rca_0001": {1: np.zeros((31, 32))}},
        )
        with self.assertRaisesRegex(ValueError, "has detector shape"):
            wrong_shape[0]

        one_view_config = self.dataset_config()
        one_view_config["data"]["views"]["count"] = 1  # type: ignore[index]
        not_selected = ImageCASDataset(
            [pair],
            one_view_config,
            training=False,
            image_replacements={"rca_0001": {1: np.zeros((32, 32))}},
        )
        with self.assertRaisesRegex(ValueError, "not in the ordered selected views"):
            not_selected[0]

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_cache_key_includes_exact_selected_replacement_images(self) -> None:
        projection = self.root / "rca_0001.npz"
        gt = self.root / "gt" / "rca" / "1.npz"
        self.write_projection(
            projection,
            image_size=32,
            center_m=(0.0155, 0.0155, 0.0155),
        )
        self.write_gt(gt, shape=(32, 32, 32))
        pair = CasePair("rca_0001", "rca", 1, projection.resolve(), gt.resolve())
        config = self.dataset_config(cache_enabled=True)

        first = ImageCASDataset(
            [pair],
            config,
            training=False,
            image_replacements={
                "rca_0001": {1: np.full((32, 32), 0.25, dtype=np.float32)}
            },
        )[0]
        second = ImageCASDataset(
            [pair],
            config,
            training=False,
            image_replacements={
                "rca_0001": {1: np.full((32, 32), 0.75, dtype=np.float32)}
            },
        )[0]

        first_metadata = json.loads(first["metadata_json"])
        second_metadata = json.loads(second["metadata_json"])
        self.assertNotEqual(
            first_metadata["selected_images_sha256"],
            second_metadata["selected_images_sha256"],
        )
        self.assertTrue(torch.equal(first["input"], second["input"]))
        self.assertEqual(len(list((self.root / "cache" / "rca").glob("*.npz"))), 2)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_one_cpu_dataloader_batch(self) -> None:
        projection = self.root / "rca_0001.npz"
        gt = self.root / "gt" / "rca" / "1.npz"
        self.write_projection(
            projection,
            image_size=32,
            center_m=(0.0155, 0.0155, 0.0155),
        )
        self.write_gt(gt, shape=(32, 32, 32))
        pair = CasePair("rca_0001", "rca", 1, projection.resolve(), gt.resolve())
        config = self.dataset_config()
        dataset = ImageCASDataset([pair], config, training=False)
        batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
        self.assertEqual(tuple(batch["input"].shape), (1, 1, 32, 32, 32))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 32, 32, 32))
        self.assertEqual(float(batch["input"].max()), 2.0)
        self.assertEqual(int(torch.count_nonzero(batch["target"])), 1)
        metadata = json.loads(batch["metadata_json"][0])
        self.assertEqual(metadata["replaced_view_indices"], [])
        self.assertEqual(metadata["replacement_images_sha256"], {})
        self.assertEqual(len(metadata["selected_images_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
