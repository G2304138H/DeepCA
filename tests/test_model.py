from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised on dependency-light machines.
    torch = None

if torch is not None:
    from deepca.engine import autocast_context, make_grad_scaler
    from deepca.modeling import (
        architecture_metadata,
        build_models,
        extract_generator_state_dict,
    )
    from train_models.networks.CCT_3d import CCT
    from train_models.networks.discriminator import Discriminator
    from train_models.networks.generator import Generator


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ReleasedModelTestCase(unittest.TestCase):
    def test_small_configurable_volume_forward(self) -> None:
        model = Generator(num_filters=8, volume_size=32).eval()
        sample = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)
        with torch.no_grad():
            output = model(sample)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32, 32))
        self.assertEqual(model.viTrans.trans.final_linear.in_features, 16)
        self.assertEqual(model.viTrans.trans.final_linear.out_features, 8)

    def test_small_critic_forward_and_backward(self) -> None:
        critic = Discriminator(torch.device("cpu"), channels=2).train()
        sample = torch.randn(
            (1, 2, 32, 32, 32), dtype=torch.float32, requires_grad=True
        )
        output = critic(sample)
        self.assertEqual(tuple(output.shape), (1, 1, 1, 1, 1))
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.mean().backward()
        self.assertIsNotNone(sample.grad)
        self.assertTrue(bool(torch.isfinite(sample.grad).all()))

    def test_disabled_amp_helpers_on_cpu(self) -> None:
        scaler = make_grad_scaler(False)
        self.assertFalse(scaler.is_enabled())
        sample = torch.ones(2, dtype=torch.float32)
        with autocast_context(False, torch.device("cpu")):
            output = sample * 2.0
        torch.testing.assert_close(output, torch.full((2,), 2.0))

    def test_released_128_latent_projection_shape_is_unchanged(self) -> None:
        # Small channel width avoids allocating the full 166M-parameter model;
        # token and spatial projection dimensions do not depend on that width.
        cct = CCT(
            vol_size=8,
            n_input_channels=8,
            embedding_dim=8,
            num_layers=1,
            num_heads=8,
        )
        self.assertEqual(tuple(cct.trans.final_linear.weight.shape), (512, 640))
        self.assertEqual(tuple(cct.trans.final_linear.bias.shape), (512,))

    def test_volume_size_validation(self) -> None:
        for invalid in (16, 31, 33, 48.0, True):
            with self.subTest(volume_size=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    Generator(num_filters=8, volume_size=invalid)

        model = Generator(num_filters=8, volume_size=32)
        with self.assertRaisesRegex(ValueError, "5D NCDHW"):
            model(torch.zeros((1, 1, 32, 32), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "cubic spatial shape"):
            model(torch.zeros((1, 1, 32, 32, 48), dtype=torch.float32))

    def test_factory_metadata_and_checkpoint_formats(self) -> None:
        config = {
            "preprocessing": {"volume_size": 32},
            "model": {
                "generator": {"base_filters": 8},
                "discriminator": {"channels": 2},
            },
        }
        generator, discriminator = build_models(config, "cpu")
        metadata = architecture_metadata(generator, discriminator)
        self.assertEqual(metadata["volume_size"], 32)
        self.assertEqual(metadata["latent_sequence_length"], 16)
        self.assertEqual(metadata["latent_projection_in_features"], 16)
        self.assertEqual(metadata["latent_projection_out_features"], 8)
        self.assertEqual(metadata["output_activation"], "identity")
        self.assertEqual(metadata["discriminator_input_channels"], 2)

        state = generator.state_dict()
        self.assertIs(extract_generator_state_dict(state), state)
        self.assertIs(extract_generator_state_dict({"network": state}), state)
        self.assertIs(
            extract_generator_state_dict({"models": {"generator": state}}),
            state,
        )

        prefixed = {f"module.{key}": value for key, value in state.items()}
        extracted = extract_generator_state_dict({"state_dict": prefixed})
        self.assertEqual(set(extracted), set(state))

    def test_conflicting_model_and_preprocessing_sizes_are_rejected(self) -> None:
        config = {
            "preprocessing": {"volume_size": 32},
            "model": {"volume_size": 64, "base_filters": 8},
        }
        with self.assertRaisesRegex(ValueError, "Conflicting configured volume"):
            build_models(config, "cpu")


if __name__ == "__main__":
    unittest.main()
