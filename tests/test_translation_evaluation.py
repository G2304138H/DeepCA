from __future__ import annotations

import json
import tempfile
import unittest
import copy
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
    import yaml
except ImportError:
    torch = None
    yaml = None

if torch is not None and yaml is not None:
    import evaluate
    from deepca.data import CasePair, load_projection_npz
    from deepca.modeling import architecture_metadata, build_generator
    from deepca.splits import load_resolved_splits
    from deepca.translation import (
        _artery_to_surface_rings,
        _render_mask,
        load_render_source,
    )
    from deepca.translation_evaluation import run_fixed_translation_evaluation


@unittest.skipIf(torch is None or yaml is None, "PyTorch/PyYAML are not installed")
class FixedTranslationEvaluationTestCase(unittest.TestCase):
    @staticmethod
    def _artery() -> np.ndarray:
        artery = np.zeros((1, 12, 4), dtype=np.float32)
        artery[0, :, 0] = np.linspace(0.010, 0.020, 12)
        artery[0, :, 1] = 0.015
        artery[0, :, 2] = 0.015
        artery[0, :, 3] = 0.001
        return artery

    @staticmethod
    def _projection_payload(images: np.ndarray, center_m: np.ndarray) -> dict[str, object]:
        return {
            "sample_name": np.asarray("rca_0001"),
            "images": images.astype(np.float32),
            "theta_deg": np.asarray([25.0, 0.0], dtype=np.float32),
            "phi_deg": np.asarray([5.0, 0.0], dtype=np.float32),
            "artery": FixedTranslationEvaluationTestCase._artery(),
            "projected_branch_indices": np.asarray([0], dtype=np.int32),
            "projection_center_reference_branch_indices": np.asarray(
                [0], dtype=np.int32
            ),
            "projection_center_offset": np.asarray(center_m, dtype=np.float32),
            "projection_center_offset_units": np.asarray("m"),
            "sid": np.asarray(0.9, dtype=np.float32),
            "sid_units": np.asarray("m"),
            "imager_pixel_spacing": np.asarray(0.5, dtype=np.float32),
            "imager_pixel_spacing_units": np.asarray("mm"),
            "mask_render_mode": np.asarray("filled"),
        }

    def test_control_plus_nine_conditions_keep_target_views_and_angles_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection_root = root / "projections"
            projection_root.mkdir()
            projection_path = projection_root / "rca_0001.npz"
            gt_path = root / "gt" / "rca" / "1.npz"
            gt_path.parent.mkdir(parents=True)

            # Bootstrap the deterministic clean view-2 image, then store exactly
            # that evidence so the mandatory >=0.98 gate is meaningful.
            initial_images = np.zeros((2, 64, 64), dtype=np.float32)
            _, center_m = _artery_to_surface_rings(
                self._artery(),
                num_circle_points=24,
                center_reference_branch_indices=(0,),
            )
            np.savez_compressed(
                projection_path,
                **self._projection_payload(initial_images, center_m),
            )
            projection = load_projection_npz(
                projection_path,
                expected_case_id="rca_0001",
                expected_vessel="rca",
                fallback_sid_m=None,
                fallback_detector_pixel_spacing_mm=None,
            )
            source = load_render_source(
                projection_path, projection, 24, "filled"
            )
            clean_second = _render_mask(
                source.surface_rings,
                theta_deg=0.0,
                phi_deg=0.0,
                image_shape_hw=(64, 64),
                sid_m=source.sid_m,
                source_to_isocentre_m=source.source_to_isocentre_m,
                detector_pixel_spacing_mm=source.detector_pixel_spacing_mm,
                render_mode=source.render_mode,
            )
            clean_first = _render_mask(
                source.surface_rings,
                theta_deg=25.0,
                phi_deg=5.0,
                image_shape_hw=(64, 64),
                sid_m=source.sid_m,
                source_to_isocentre_m=source.source_to_isocentre_m,
                detector_pixel_spacing_mm=source.detector_pixel_spacing_mm,
                render_mode=source.render_mode,
            )
            stored_images = initial_images.copy()
            stored_images[0] = clean_first
            stored_images[1] = clean_second
            np.savez_compressed(
                projection_path,
                **self._projection_payload(stored_images, center_m),
            )

            target = np.zeros((32, 32, 32), dtype=np.uint8)
            target[10:21, 14:17, 14:17] = 1
            np.savez_compressed(
                gt_path,
                vol=target,
                spacing=np.ones(3, dtype=np.float32),
            )
            output_dir = root / "evaluation"
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps({"train": [], "val": [], "test": [1]}),
                encoding="utf-8",
            )
            config = {
                "experiment": {"seed": 1, "output_dir": str(root / "train")},
                "data": {
                    "vessel_type": "rca",
                    "projection_root": str(projection_root),
                    "ground_truth_root": str(root / "gt"),
                    "split_json": str(split_path),
                    "calibration": {
                        "source_to_isocentre_m": 0.75,
                        "fallback_sid_m": None,
                        "fallback_detector_pixel_spacing_mm": None,
                        "expected_detector_pixel_spacing_mm": 0.5,
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
                        "eval_selection": "first",
                        "train_selection": "first",
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
                    # The runner must still force this off at execution time.
                    "cache": {"enabled": True, "directory": str(root / "cache")},
                },
                "model": {
                    "volume_size": 32,
                    "generator": {
                        "in_channels": 1,
                        "output_channels": 1,
                        "base_filters": 8,
                        "batch_norm": True,
                        "sample": False,
                    },
                    "critic": {"channels": 2, "dim": 32},
                },
                "evaluation": {
                    "mode": "fixed_two_view_translation",
                    "view_counts": [2],
                    "device": "cpu",
                    "threshold": 0.5,
                    "split": "test",
                    "save_predictions": False,
                    "fixed_two_view_translation": {
                        "renderer_num_circle_points": 24,
                        "minimum_clean_rerender_dice": 0.98,
                        "visibility_warning_threshold": 0.0,
                    },
                },
            }
            pair = CasePair(
                "rca_0001",
                "rca",
                1,
                projection_path.resolve(),
                gt_path.resolve(),
            )

            class IdentityGenerator(torch.nn.Module):
                def forward(self, value):  # type: ignore[no-untyped-def]
                    return value

            context = run_fixed_translation_evaluation(
                config=config,
                pairs=[pair],
                generator=IdentityGenerator().eval(),
                device=torch.device("cpu"),
                threshold=0.5,
                checkpoint_path=root / "checkpoint.pt",
                checkpoint_sha256="synthetic",
                checkpoint_config_mismatch_override=False,
                config_fingerprint_value="synthetic-config",
                split="test",
                split_manifest={"synthetic": True},
                output_dir=output_dir,
                save_predictions=True,
            )

            result_dir = Path(context["result_directory"])
            payload = json.loads((result_dir / "per_case.json").read_text())
            rows = payload["results"]
            self.assertEqual(len(rows), 10)
            self.assertEqual(
                [row["condition_id"] for row in rows],
                [
                    "accurate_control",
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
            self.assertEqual({tuple(row["view_indices"]) for row in rows}, {(0, 1)})
            self.assertEqual({tuple(row["theta_deg"]) for row in rows}, {(25.0, 0.0)})
            self.assertEqual({tuple(row["phi_deg"]) for row in rows}, {(5.0, 0.0)})
            self.assertEqual(len({row["target_sha256"] for row in rows}), 1)
            self.assertTrue(
                all(row["delta_theta_deg"] == [0.0, 0.0] for row in rows)
            )
            self.assertTrue(
                all(row["delta_phi_deg"] == [0.0, 0.0] for row in rows)
            )
            self.assertTrue(
                all(row["clean_rerender_dice_vs_stored"] >= 0.98 for row in rows)
            )
            self.assertTrue(all(row["feature_cache_used"] is False for row in rows))
            self.assertTrue(all(row["refined_model_metrics"] is None for row in rows))
            self.assertTrue(all(row["refined_minus_coarse"] is None for row in rows))
            self.assertTrue(context["configured_cache_enabled"])
            self.assertFalse(context["effective_feature_cache_enabled"])
            self.assertEqual(context["successful_results"], 10)
            self.assertFalse(context["checkpoint_config_mismatch_override"])
            self.assertEqual(context["config_fingerprint"], "synthetic-config")
            prediction_dir = Path(context["prediction_directory"])
            self.assertEqual(len(list(prediction_dir.rglob("*.npz"))), 10)
            self.assertFalse(
                any(output_dir.glob(".*.staging"))
            )
            self.assertTrue(
                all(Path(row["prediction_path"]).is_file() for row in rows)
            )

            completed_before = set((output_dir / "runs").iterdir())
            with mock.patch(
                "deepca.translation_evaluation._write_csv",
                side_effect=OSError("synthetic publication failure"),
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic publication failure"
                ):
                    run_fixed_translation_evaluation(
                        config=config,
                        pairs=[pair],
                        generator=IdentityGenerator().eval(),
                        device=torch.device("cpu"),
                        threshold=0.5,
                        checkpoint_path=root / "checkpoint.pt",
                        checkpoint_sha256="synthetic-failure",
                        checkpoint_config_mismatch_override=True,
                        config_fingerprint_value="failure-config",
                        split="test",
                        split_manifest={"synthetic": True},
                        output_dir=output_dir,
                        save_predictions=True,
                    )
            self.assertEqual(
                set((output_dir / "runs").iterdir()), completed_before
            )
            self.assertFalse(any(output_dir.glob(".*.staging")))

            # Exercise the public evaluate.py dispatch as well as the direct
            # runner above.
            main_config = copy.deepcopy(config)
            checkpoint_path = root / "checkpoint.pt"
            main_output = root / "main_evaluation"
            main_config["evaluation"]["checkpoint"] = str(checkpoint_path)
            main_config["evaluation"]["output_dir"] = str(main_output)
            generator = build_generator(main_config, "cpu")
            torch.save(
                {
                    "schema_version": 2,
                    "generator": generator.state_dict(),
                    "architecture": architecture_metadata(generator),
                    "config": main_config,
                    "resolved_splits": load_resolved_splits(
                        split_path, "rca"
                    ).as_manifest(),
                },
                checkpoint_path,
            )
            config_path = root / "translation.yaml"
            config_path.write_text(
                yaml.safe_dump(main_config), encoding="utf-8"
            )
            self.assertEqual(
                evaluate.main(["--config", str(config_path), "--split", "test"]),
                0,
            )
            completed_runs = list((main_output / "test" / "runs").glob("run-*"))
            self.assertEqual(len(completed_runs), 1)
            main_payload = json.loads((completed_runs[0] / "per_case.json").read_text())
            self.assertEqual(len(main_payload["results"]), 10)
            self.assertEqual(
                main_payload["context"]["evaluation_mode"],
                "fixed_two_view_translation",
            )


if __name__ == "__main__":
    unittest.main()
