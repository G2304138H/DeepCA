from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np

from deepca.data import ProjectionCase
from deepca.translation import (
    DEFAULT_TRANSLATION_CONDITIONS,
    TranslationCondition,
    _artery_to_surface_rings,
    _render_mask,
    is_fixed_translation_mode,
    load_render_source,
    render_translation,
    resolve_translation_plan,
)


class TranslationPlanTestCase(unittest.TestCase):
    def test_default_conditions_are_fixed_ordered_and_preserve_total_norm(self) -> None:
        self.assertEqual(
            [condition.condition_id for condition in DEFAULT_TRANSLATION_CONDITIONS],
            [
                "Y-5",
                "Y-10",
                "Y-20",
                "XZ-5",
                "XZ-10",
                "XZ-20",
                "XYZ-5",
                "XYZ-10",
                "XYZ-20",
            ],
        )
        for condition in DEFAULT_TRANSLATION_CONDITIONS:
            self.assertAlmostEqual(
                math.dist((0.0, 0.0, 0.0), condition.translation_xyz_mm),
                condition.magnitude_mm,
                places=12,
            )
            self.assertEqual(condition.delta_theta_deg, 0.0)
            self.assertEqual(condition.delta_phi_deg, 0.0)
        np.testing.assert_allclose(
            DEFAULT_TRANSLATION_CONDITIONS[3].translation_xyz_mm,
            (5.0 / math.sqrt(2.0), 0.0, 5.0 / math.sqrt(2.0)),
        )
        np.testing.assert_allclose(
            DEFAULT_TRANSLATION_CONDITIONS[6].translation_xyz_mm,
            (5.0 / math.sqrt(3.0),) * 3,
        )
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_TRANSLATION_CONDITIONS[0].magnitude_mm = 7.0  # type: ignore[misc]

    def test_plan_requires_two_views_second_position_and_valid_thresholds(self) -> None:
        evaluation = {
            "mode": "fixed_two_view_translation",
            "fixed_two_view_translation": {
                "perturbed_input_position": 1,
                "minimum_clean_rerender_dice": 0.98,
                "visibility_warning_threshold": 0.9,
            },
        }
        plan = resolve_translation_plan(evaluation, [2])
        self.assertTrue(is_fixed_translation_mode(evaluation["mode"]))
        self.assertIs(plan.conditions, DEFAULT_TRANSLATION_CONDITIONS)
        self.assertEqual(plan.perturbed_input_position, 1)
        self.assertEqual(plan.settings["minimum_clean_rerender_dice"], 0.98)
        self.assertEqual(len(plan.as_dict()["conditions"]), 9)

        with self.assertRaisesRegex(ValueError, r"exactly \[2\]"):
            resolve_translation_plan(evaluation, [1, 2])
        with self.assertRaisesRegex(ValueError, "perturbed_input_position must be 1"):
            resolve_translation_plan(
                {
                    **evaluation,
                    "fixed_two_view_translation": {
                        "perturbed_input_position": 0
                    },
                },
                [2],
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            resolve_translation_plan(
                {
                    **evaluation,
                    "fixed_two_view_translation": {
                        "visibility_warning_threshold": 1.1
                    },
                },
                [2],
            )
        with self.assertRaisesRegex(ValueError, "cannot be below"):
            resolve_translation_plan(
                {
                    **evaluation,
                    "fixed_two_view_translation": {
                        "minimum_clean_rerender_dice": 0.97
                    },
                },
                [2],
            )
        with self.assertRaisesRegex(ValueError, r"must be in \[0, 0.01\]"):
            resolve_translation_plan(
                {
                    **evaluation,
                    "fixed_two_view_translation": {
                        "maximum_center_offset_difference_mm": 0.02
                    },
                },
                [2],
            )
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            resolve_translation_plan(
                {
                    **evaluation,
                    "fixed_two_view_translation": {"conditions": []},
                },
                [2],
            )

    def test_condition_rejects_nonzero_angle_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero angle deltas"):
            TranslationCondition(
                condition_id="invalid",
                pattern="Y",
                magnitude_mm=5.0,
                translation_xyz_mm=(0.0, 5.0, 0.0),
                delta_theta_deg=1.0,
            )


class TranslationRendererTestCase(unittest.TestCase):
    @staticmethod
    def _synthetic_artery() -> np.ndarray:
        artery = np.zeros((2, 18, 4), dtype=np.float32)
        artery[0, :, 0] = np.linspace(0.01, 0.02, 18)
        artery[0, :, 1] = -0.006
        artery[0, :, 2] = 0.003
        artery[0, :, 3] = 0.0008
        artery[1, :, 0] = np.linspace(0.02, 0.04, 18)
        artery[1, :, 1] = 0.01
        artery[1, :, 2] = 0.005
        artery[1, :, 3] = 0.001
        return artery

    def test_synthetic_clean_control_and_mm_translation_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lca_0001.npz"
            artery = self._synthetic_artery()
            _, center_m = _artery_to_surface_rings(
                artery[[1]],
                num_circle_points=48,
                center_reference_branch_indices=(0,),
            )
            np.savez_compressed(
                path,
                artery=artery,
                projected_branch_indices=np.asarray([1], dtype=np.int32),
                num_projected_branches=np.asarray(1, dtype=np.int32),
                projection_center_reference_branch_indices=np.asarray(
                    [1], dtype=np.int32
                ),
                projection_center_offset=np.asarray(center_m, dtype=np.float32),
                projection_center_offset_units=np.asarray("m"),
                mask_render_mode=np.asarray("filled"),
            )
            image_shape = (96, 96)
            projection = ProjectionCase(
                path=path,
                case_id="lca_0001",
                vessel_type="lca",
                case_number=1,
                images=np.zeros((2, *image_shape), dtype=np.float32),
                theta_deg=np.asarray([-25.0, 0.0], dtype=np.float32),
                phi_deg=np.asarray([10.0, 0.0], dtype=np.float32),
                sid_m=0.9,
                sid_source="archive",
                detector_pixel_spacing_mm=0.35,
                detector_pixel_spacing_source="archive",
                source_to_isocentre_m=0.75,
                projection_center_offset_xyz_mm=np.asarray(center_m) * 1000.0,
                projection_center_offset_source="archive",
                clinical_views=("view_0", "view_1"),
            )
            source = load_render_source(path, projection, 48, "filled")
            with self.assertRaisesRegex(ValueError, "requires the recorded"):
                load_render_source(
                    path,
                    replace(
                        projection,
                        projection_center_offset_xyz_mm=None,
                        projection_center_offset_source=(
                            "ground_truth_center_fallback"
                        ),
                    ),
                    48,
                    "filled",
                )
            with self.assertRaisesRegex(ValueError, "differs from the recorded"):
                load_render_source(
                    path,
                    replace(
                        projection,
                        projection_center_offset_xyz_mm=(
                            projection.projection_center_offset_xyz_mm
                            + np.asarray([1.0, 0.0, 0.0])
                        ),
                    ),
                    48,
                    "filled",
                )

            clean = _render_mask(
                source.surface_rings,
                theta_deg=0.0,
                phi_deg=0.0,
                image_shape_hw=image_shape,
                sid_m=source.sid_m,
                source_to_isocentre_m=source.source_to_isocentre_m,
                detector_pixel_spacing_mm=source.detector_pixel_spacing_mm,
                render_mode=source.render_mode,
            )
            replacement, diagnostics = render_translation(
                source,
                clean,
                0.0,
                0.0,
                (0.0, 5.0, 0.0),
                0.999,
                0.95,
                False,
            )

            self.assertEqual(clean.shape, image_shape)
            self.assertGreater(np.count_nonzero(clean), 0)
            self.assertEqual(replacement.shape, image_shape)
            self.assertFalse(np.array_equal(replacement, clean))
            self.assertEqual(diagnostics["condition_id"], "Y-5")
            self.assertAlmostEqual(
                diagnostics["clean_rerender_dice_vs_stored"], 1.0
            )
            np.testing.assert_allclose(
                diagnostics["artery_translation_xyz_m"], [0.0, 0.005, 0.0]
            )
            np.testing.assert_allclose(
                diagnostics["equivalent_system_translation_xyz_mm"],
                [-0.0, -5.0, -0.0],
            )
            self.assertAlmostEqual(diagnostics["translation_magnitude_mm"], 5.0)
            self.assertEqual(diagnostics["source_view_index"], 1)
            self.assertEqual(diagnostics["projected_branch_indices"], [1])
            self.assertEqual(
                diagnostics["projection_center_reference_branch_indices"], [1]
            )
            self.assertEqual(diagnostics["delta_theta_deg"], 0.0)
            self.assertEqual(diagnostics["delta_phi_deg"], 0.0)
            self.assertTrue(
                0.0 <= diagnostics["visible_centerline_fraction"] <= 1.0
            )
            self.assertTrue(
                0.0 <= diagnostics["visible_vessel_surface_fraction"] <= 1.0
            )
            json.dumps(diagnostics)

            with self.assertRaisesRegex(ValueError, "no foreground vessel pixels"):
                render_translation(
                    source,
                    np.zeros(image_shape, dtype=np.float32),
                    0.0,
                    0.0,
                    (0.0, 5.0, 0.0),
                    0.98,
                    0.95,
                    False,
                )


if __name__ == "__main__":
    unittest.main()
