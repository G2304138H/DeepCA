from __future__ import annotations

import math
import unittest

import numpy as np

from deepca.metrics import (
    PhysicalGrid,
    aggregate_statistics,
    binary_cldice,
    binary_dice,
    skeletonization_metadata,
    validate_physical_grid_alignment,
)


def make_grid(
    shape: tuple[int, int, int],
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PhysicalGrid:
    return PhysicalGrid(
        shape_zyx=shape,
        spacing_xyz_mm=spacing,
        origin_xyz_mm=origin,
    )


class BinaryDiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (5, 5, 5)
        self.grid = make_grid(self.shape)

    def score(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return binary_dice(
            prediction,
            target,
            prediction_grid=self.grid,
            target_grid=self.grid,
        )

    def test_identical_masks(self) -> None:
        mask = np.zeros(self.shape, dtype=np.uint8)
        mask[2, 2, 1:4] = 1
        self.assertEqual(self.score(mask, mask.copy()), 1.0)

    def test_disjoint_masks(self) -> None:
        prediction = np.zeros(self.shape, dtype=bool)
        target = np.zeros(self.shape, dtype=bool)
        prediction[1, 1, 1] = True
        target[3, 3, 3] = True
        self.assertEqual(self.score(prediction, target), 0.0)

    def test_partially_overlapping_masks(self) -> None:
        prediction = np.zeros(self.shape, dtype=np.uint8)
        target = np.zeros(self.shape, dtype=np.uint8)
        prediction[2, 2, (1, 2)] = 1
        target[2, 2, (2, 3)] = 1
        self.assertEqual(self.score(prediction, target), 0.5)

    def test_empty_mask_behavior(self) -> None:
        empty = np.zeros(self.shape, dtype=bool)
        nonempty = empty.copy()
        nonempty[2, 2, 2] = True
        self.assertEqual(self.score(empty, empty), 1.0)
        self.assertEqual(self.score(empty, nonempty), 0.0)
        self.assertEqual(self.score(nonempty, empty), 0.0)

    def test_non_binary_values_are_rejected(self) -> None:
        prediction = np.zeros(self.shape, dtype=np.float32)
        prediction[2, 2, 2] = 0.5
        target = np.zeros(self.shape, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "strictly binary"):
            self.score(prediction, target)


class BinaryClDiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (7, 7, 7)
        self.grid = make_grid(self.shape)

    def score(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return binary_cldice(
            prediction,
            target,
            prediction_grid=self.grid,
            target_grid=self.grid,
        )

    def line(self, start: int, stop: int) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        mask[3, 3, start:stop] = True
        return mask

    def test_identical_centerlines(self) -> None:
        mask = self.line(1, 6)
        self.assertEqual(self.score(mask, mask.copy()), 1.0)

    def test_disconnected_prediction_penalizes_missing_centerline(self) -> None:
        target = self.line(1, 6)
        prediction = target.copy()
        prediction[3, 3, 3] = False
        # tprec=1 and tsens=4/5 because every predicted skeleton voxel lies
        # in the target while one of five target skeleton voxels is absent.
        self.assertAlmostEqual(self.score(prediction, target), 8.0 / 9.0)

    def test_partially_overlapping_centerlines(self) -> None:
        prediction = self.line(1, 4)
        target = self.line(2, 5)
        self.assertAlmostEqual(self.score(prediction, target), 2.0 / 3.0)

    def test_empty_mask_behavior(self) -> None:
        empty = np.zeros(self.shape, dtype=bool)
        nonempty = self.line(1, 6)
        self.assertEqual(self.score(empty, empty), 1.0)
        self.assertEqual(self.score(empty, nonempty), 0.0)
        self.assertEqual(self.score(nonempty, empty), 0.0)

    def test_skeletonizer_metadata_records_library_version_and_method(self) -> None:
        metadata = skeletonization_metadata()
        self.assertEqual(metadata["library"], "scikit-image")
        self.assertIsInstance(metadata["version"], str)
        self.assertTrue(metadata["version"])
        self.assertEqual(metadata["method"], "lee")
        self.assertEqual(
            metadata["function"], "skimage.morphology.skeletonize"
        )


class PhysicalGridAlignmentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.volume = np.zeros((4, 5, 6), dtype=bool)
        self.grid = make_grid(
            self.volume.shape,
            spacing=(0.7, 0.8, 0.9),
            origin=(-1.0, -2.0, -3.0),
        )

    def test_matching_grids_are_accepted(self) -> None:
        validate_physical_grid_alignment(
            self.volume,
            self.volume.copy(),
            prediction_grid=self.grid,
            target_grid=self.grid,
        )

    def test_array_shape_mismatch_is_rejected(self) -> None:
        target = np.zeros((4, 5, 7), dtype=bool)
        target_grid = make_grid(
            target.shape,
            spacing=self.grid.spacing_xyz_mm,
            origin=self.grid.origin_xyz_mm,
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_physical_grid_alignment(
                self.volume,
                target,
                prediction_grid=self.grid,
                target_grid=target_grid,
            )

    def test_array_to_declared_grid_shape_mismatch_is_rejected(self) -> None:
        wrong_grid = make_grid(
            (4, 5, 7),
            spacing=self.grid.spacing_xyz_mm,
            origin=self.grid.origin_xyz_mm,
        )
        with self.assertRaisesRegex(ValueError, "array/grid shape mismatch"):
            validate_physical_grid_alignment(
                self.volume,
                self.volume,
                prediction_grid=wrong_grid,
                target_grid=self.grid,
            )

    def test_spacing_mismatch_is_rejected(self) -> None:
        target_grid = make_grid(
            self.volume.shape,
            spacing=(0.7, 0.8, 1.0),
            origin=self.grid.origin_xyz_mm,
        )
        with self.assertRaisesRegex(ValueError, "spacing mismatch"):
            binary_dice(
                self.volume,
                self.volume,
                prediction_grid=self.grid,
                target_grid=target_grid,
            )

    def test_origin_mismatch_is_rejected(self) -> None:
        target_grid = make_grid(
            self.volume.shape,
            spacing=self.grid.spacing_xyz_mm,
            origin=(-1.0, -2.0, -2.5),
        )
        with self.assertRaisesRegex(ValueError, "origin mismatch"):
            binary_cldice(
                self.volume,
                self.volume,
                prediction_grid=self.grid,
                target_grid=target_grid,
            )


class AggregateStatisticsTestCase(unittest.TestCase):
    def test_sample_statistics(self) -> None:
        stats = aggregate_statistics([1.0, 2.0, 3.0])
        self.assertEqual(stats.n, 3)
        self.assertEqual(stats.mean, 2.0)
        self.assertEqual(stats.sample_std, 1.0)
        self.assertEqual(stats.median, 2.0)
        self.assertAlmostEqual(stats.standard_error, 1.0 / math.sqrt(3.0))
        self.assertEqual(stats.as_dict()["n"], 3)

    def test_single_case_has_zero_dispersion(self) -> None:
        stats = aggregate_statistics([0.75])
        self.assertEqual(stats.n, 1)
        self.assertEqual(stats.mean, 0.75)
        self.assertEqual(stats.sample_std, 0.0)
        self.assertEqual(stats.median, 0.75)
        self.assertEqual(stats.standard_error, 0.0)

    def test_empty_or_nonfinite_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            aggregate_statistics([])
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate_statistics([1.0, float("nan")])


if __name__ == "__main__":
    unittest.main()
