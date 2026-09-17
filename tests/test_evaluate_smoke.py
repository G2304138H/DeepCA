from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import yaml
except ImportError:
    torch = None
    yaml = None

if torch is not None and yaml is not None:
    import evaluate
    from deepca.modeling import architecture_metadata, build_generator
    from deepca.splits import load_resolved_splits


@unittest.skipIf(torch is None or yaml is None, "PyTorch/PyYAML are not installed")
class EvaluationSmokeTestCase(unittest.TestCase):
    def test_one_case_end_to_end_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection_root = root / "projections"
            gt_root = root / "ground_truth"
            projection_root.mkdir()
            (gt_root / "rca").mkdir(parents=True)
            for number in (1, 2, 3):
                images = np.ones((2, 32, 32), dtype=np.float32)
                theta = np.asarray([0.0, 30.0], dtype=np.float32)
                phi = np.asarray([90.0, 70.0], dtype=np.float32)
                np.savez_compressed(
                    projection_root / f"rca_{number:04d}.npz",
                    sample_name=np.asarray(f"rca_{number:04d}"),
                    images=images,
                    theta_deg=theta,
                    phi_deg=phi,
                    projection_center_offset=np.asarray(
                        [0.0155, 0.0155, 0.0155], dtype=np.float32
                    ),
                )
                volume = np.zeros((32, 32, 32), dtype=np.uint8)
                volume[16, 16, 12:20] = 1
                np.savez_compressed(
                    gt_root / "rca" / f"{number}.npz",
                    vol=volume,
                    spacing=np.ones(3, dtype=np.float32),
                )
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps({"train": [1], "val": [2], "test": [3]}),
                encoding="utf-8",
            )
            output_dir = root / "evaluation"
            checkpoint_path = root / "checkpoint.pt"
            config = {
                "experiment": {"seed": 1, "output_dir": str(root / "train")},
                "data": {
                    "vessel_type": "rca",
                    "projection_root": str(projection_root),
                    "ground_truth_root": str(gt_root),
                    "split_json": str(split_path),
                    "calibration": {
                        "source_to_isocentre_m": 0.75,
                        "fallback_sid_m": 0.9,
                        "fallback_detector_pixel_spacing_mm": 0.55,
                        "expected_detector_pixel_spacing_mm": 0.55,
                        "projection_center_offset_units": "m",
                    },
                    "ground_truth": {
                        "volume_key": "vol",
                        "spacing_key": "spacing",
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
                    "cache": {"enabled": False, "directory": str(root / "cache")},
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
                    "device": "cpu",
                    "checkpoint": str(checkpoint_path),
                    "split": "test",
                    "threshold": 0.5,
                    "view_counts": [2],
                    "save_predictions": True,
                    "output_dir": str(output_dir),
                },
            }
            generator = build_generator(config, "cpu")
            torch.save(
                {
                    "schema_version": 2,
                    "generator": generator.state_dict(),
                    "architecture": architecture_metadata(generator),
                    "config": config,
                    "resolved_splits": load_resolved_splits(
                        split_path, "rca"
                    ).as_manifest(),
                },
                checkpoint_path,
            )
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            exit_code = evaluate.main(["--config", str(config_path), "--split", "test"])
            self.assertEqual(exit_code, 0)
            result_root = output_dir / "test"
            per_case = json.loads((result_root / "per_case.json").read_text())
            failures = json.loads((result_root / "failures.json").read_text())
            summary = json.loads((result_root / "summary.json").read_text())
            self.assertEqual(len(per_case["results"]), 1)
            self.assertEqual(per_case["results"][0]["case_id"], "rca_0003")
            self.assertFalse(failures["failures"])
            self.assertEqual(summary["summary"]["overall"]["n"], 1)
            prediction_path = Path(per_case["results"][0]["prediction_path"])
            self.assertTrue(prediction_path.is_file())

            mismatched = copy.deepcopy(config)
            mismatched["data"]["vessel_type"] = "lca"
            mismatched["evaluation"]["output_dir"] = str(root / "mismatch")
            mismatch_config_path = root / "mismatch.yaml"
            mismatch_config_path.write_text(
                yaml.safe_dump(mismatched), encoding="utf-8"
            )
            mismatch_exit = evaluate.main(
                ["--config", str(mismatch_config_path), "--split", "test"]
            )
            self.assertEqual(mismatch_exit, 2)
            mismatch_failures = json.loads(
                (root / "mismatch" / "test" / "failures.json").read_text()
            )
            self.assertEqual(len(mismatch_failures["failures"]), 1)
            self.assertEqual(
                mismatch_failures["failures"][0]["case_id"], "lca_0003"
            )
            self.assertIn(
                "contract differs", mismatch_failures["failures"][0]["reason"]
            )


if __name__ == "__main__":
    unittest.main()
