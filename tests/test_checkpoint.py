from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deepca.config import (
    assert_resume_config_compatible,
    normalize_config_for_resume,
)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from deepca.checkpoint import (
        atomic_torch_save,
        make_training_checkpoint,
        restore_rng_state,
        safe_torch_load,
    )
    from deepca.engine import make_grad_scaler
    from train import _validate_resume_config


def _resume_config() -> dict:
    return {
        "_config_path": "/configs/original.yaml",
        "experiment": {
            "name": "original-run",
            "output_dir": "/outputs/original",
            "seed": 17,
        },
        "data": {"vessel_type": "rca", "views": {"count": 2}},
        "model": {"volume_size": 32, "generator": {"base_filters": 8}},
        "training": {
            "device": "cuda:0",
            "epochs": 100,
            "resume": None,
            "workers": 4,
            "batch_size": 3,
            "mixed_precision": True,
            "l1_weight": 100.0,
            "generator_optimizer": {"name": "adam", "learning_rate": 1.0e-4},
            "scheduler": {"name": "step", "step_size": 20, "gamma": 0.5},
        },
        "evaluation": {"device": "cuda:0", "threshold": 0.5},
    }


class ResumeConfigTestCase(unittest.TestCase):
    def test_runtime_only_changes_are_compatible(self) -> None:
        checkpoint_config = _resume_config()
        current_config = copy.deepcopy(checkpoint_config)
        current_config["_config_path"] = "/configs/moved.yaml"
        current_config["experiment"]["name"] = "continued-run"
        current_config["experiment"]["output_dir"] = "/outputs/continued"
        current_config["training"].update(
            {"device": "cuda:1", "epochs": 150, "resume": "/checkpoints/last.pt", "workers": 8}
        )
        current_config["evaluation"] = {"device": "cpu", "threshold": 0.75}

        assert_resume_config_compatible(current_config, checkpoint_config)
        self.assertEqual(
            normalize_config_for_resume(current_config),
            normalize_config_for_resume(checkpoint_config),
        )

    def test_scientific_and_training_changes_are_rejected(self) -> None:
        changes = {
            "experiment.seed": lambda config: config["experiment"].update(seed=18),
            "data.views.count": lambda config: config["data"]["views"].update(count=3),
            "model.generator.base_filters": lambda config: config["model"][
                "generator"
            ].update(base_filters=16),
            "training.batch_size": lambda config: config["training"].update(
                batch_size=2
            ),
            "training.mixed_precision": lambda config: config["training"].update(
                mixed_precision=False
            ),
            "training.l1_weight": lambda config: config["training"].update(
                l1_weight=50.0
            ),
            "training.generator_optimizer.learning_rate": lambda config: config[
                "training"
            ]["generator_optimizer"].update(learning_rate=2.0e-4),
            "training.scheduler.gamma": lambda config: config["training"][
                "scheduler"
            ].update(gamma=0.1),
        }
        checkpoint_config = _resume_config()
        for expected_path, mutate in changes.items():
            with self.subTest(expected_path=expected_path):
                current_config = copy.deepcopy(checkpoint_config)
                mutate(current_config)
                with self.assertRaisesRegex(ValueError, expected_path.replace(".", r"\.")):
                    assert_resume_config_compatible(current_config, checkpoint_config)

    def test_unknown_new_fields_are_not_silently_ignored(self) -> None:
        checkpoint_config = _resume_config()
        current_config = copy.deepcopy(checkpoint_config)
        current_config["training"]["future_training_control"] = "changed"
        with self.assertRaisesRegex(ValueError, "future_training_control"):
            assert_resume_config_compatible(current_config, checkpoint_config)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTestCase(unittest.TestCase):
    def test_resume_rejects_checkpoint_without_resolved_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not contain.*resolved 'config'"):
            _validate_resume_config({"schema_version": 2}, _resume_config())

    def test_full_checkpoint_save_and_resume_state(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        generator = torch.nn.Linear(3, 2)
        critic = torch.nn.Linear(5, 1)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1.0e-3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2.0e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=1)
        scaler = make_grad_scaler(False)

        generator_optimizer.zero_grad(set_to_none=True)
        generator(torch.randn(4, 3)).square().mean().backward()
        generator_optimizer.step()
        critic_optimizer.zero_grad(set_to_none=True)
        critic(torch.randn(4, 5)).square().mean().backward()
        critic_optimizer.step()
        scheduler.step()

        payload = make_training_checkpoint(
            epoch=4,
            global_step=17,
            generator=generator,
            critic=critic,
            generator_optimizer=generator_optimizer,
            critic_optimizer=critic_optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_validation_l1=0.25,
            config={"model": {"volume_size": 32}},
            config_fingerprint="abc",
            resolved_splits={"splits": {"train": ["rca_0001"]}},
        )
        expected_random = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = torch.rand(3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save(payload, path)
            loaded = safe_torch_load(path, "cpu")

        self.assertEqual(loaded["schema_version"], 2)
        self.assertEqual(loaded["epoch"], 4)
        self.assertIn("generator_optimizer", loaded)
        self.assertIn("critic_optimizer", loaded)
        self.assertIn("scheduler", loaded)
        self.assertIn("scaler", loaded)
        replacement = torch.nn.Linear(3, 2)
        replacement.load_state_dict(loaded["generator"], strict=True)
        for expected, actual in zip(generator.parameters(), replacement.parameters()):
            torch.testing.assert_close(expected, actual)

        replacement_critic = torch.nn.Linear(5, 1)
        replacement_critic.load_state_dict(loaded["critic"], strict=True)
        replacement_generator_optimizer = torch.optim.Adam(
            replacement.parameters(), lr=1.0e-3
        )
        replacement_critic_optimizer = torch.optim.Adam(
            replacement_critic.parameters(), lr=2.0e-3
        )
        replacement_scheduler = torch.optim.lr_scheduler.StepLR(
            replacement_generator_optimizer, step_size=1
        )
        replacement_scaler = make_grad_scaler(False)
        replacement_generator_optimizer.load_state_dict(loaded["generator_optimizer"])
        replacement_critic_optimizer.load_state_dict(loaded["critic_optimizer"])
        replacement_scheduler.load_state_dict(loaded["scheduler"])
        replacement_scaler.load_state_dict(loaded["scaler"])

        self.assertEqual(
            replacement_generator_optimizer.param_groups[0]["lr"],
            generator_optimizer.param_groups[0]["lr"],
        )
        self.assertEqual(
            replacement_critic_optimizer.param_groups[0]["lr"],
            critic_optimizer.param_groups[0]["lr"],
        )
        self.assertEqual(replacement_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(replacement_scaler.state_dict(), scaler.state_dict())
        for optimizer, replacement_optimizer in (
            (generator_optimizer, replacement_generator_optimizer),
            (critic_optimizer, replacement_critic_optimizer),
        ):
            expected_states = list(optimizer.state.values())
            actual_states = list(replacement_optimizer.state.values())
            self.assertEqual(len(expected_states), len(actual_states))
            for expected_state, actual_state in zip(expected_states, actual_states):
                self.assertEqual(set(expected_state), set(actual_state))
                for key in expected_state:
                    if torch.is_tensor(expected_state[key]):
                        torch.testing.assert_close(expected_state[key], actual_state[key])
                    else:
                        self.assertEqual(expected_state[key], actual_state[key])

        random.random()
        np.random.random()
        torch.rand(3)
        restore_rng_state(loaded["rng_state"])
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(float(np.random.random()), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch)


if __name__ == "__main__":
    unittest.main()
