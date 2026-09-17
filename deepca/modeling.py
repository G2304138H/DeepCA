"""Construction and checkpoint helpers for the released DeepCA networks.

The factory keeps the released architecture defaults intact.  In particular,
the generator produces an unbounded single-channel regression volume; applying
the paper's binary threshold is an evaluation concern, not part of the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from train_models.networks.discriminator import Discriminator
from train_models.networks.generator import Generator


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _model_sections(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return the model, generator, and discriminator configuration sections."""

    root = _mapping(config, name="config")
    model = root.get("model", root)
    model = _mapping(model, name="model configuration")

    generator = model.get("generator", model)
    generator = _mapping(generator, name="generator configuration")

    discriminator = model.get("discriminator", model.get("critic", model))
    discriminator = _mapping(discriminator, name="discriminator configuration")
    return model, generator, discriminator


def _configured_volume_size(
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    generator: Mapping[str, Any],
) -> int:
    candidates: list[tuple[str, object]] = []
    if "volume_size" in model:
        candidates.append(("model.volume_size", model["volume_size"]))
    if generator is not model and "volume_size" in generator:
        candidates.append(("model.generator.volume_size", generator["volume_size"]))

    preprocessing = config.get("preprocessing")
    if preprocessing is not None:
        preprocessing = _mapping(preprocessing, name="preprocessing configuration")
        if "volume_size" in preprocessing:
            candidates.append(
                ("preprocessing.volume_size", preprocessing["volume_size"])
            )

    if not candidates:
        return 128
    for name, value in candidates:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {value!r}.")
    values = {int(value) for _, value in candidates}
    if len(values) != 1:
        rendered = ", ".join(f"{name}={value!r}" for name, value in candidates)
        raise ValueError(f"Conflicting configured volume sizes: {rendered}.")
    return int(candidates[0][1])


def _integer(
    section: Mapping[str, Any],
    primary: str,
    alias: str,
    default: int,
) -> int:
    has_primary = primary in section
    has_alias = alias in section
    if has_primary and has_alias and section[primary] != section[alias]:
        raise ValueError(
            f"Conflicting model settings: {primary}={section[primary]!r}, "
            f"{alias}={section[alias]!r}."
        )
    value = section[primary] if has_primary else section.get(alias, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{primary} must be a positive integer, got {value!r}.")
    return value


def _boolean(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be boolean, got {value!r}.")
    return value


def build_generator(
    config: Mapping[str, Any],
    device: str | torch.device | None = None,
) -> Generator:
    """Build the released generator from a full or model-only configuration."""

    model, generator_config, _ = _model_sections(config)
    volume_size = _configured_volume_size(config, model, generator_config)
    generator = Generator(
        in_channels=_integer(generator_config, "in_channels", "input_channels", 1),
        num_filters=_integer(generator_config, "base_filters", "num_filters", 64),
        class_num=_integer(
            generator_config, "output_channels", "class_num", 1
        ),
        batch_norm=_boolean(generator_config, "batch_norm", True),
        sample=_boolean(generator_config, "sample", False),
        volume_size=volume_size,
    )
    if device is not None:
        generator = generator.to(torch.device(device))
    return generator


def build_discriminator(
    config: Mapping[str, Any],
    device: str | torch.device,
) -> Discriminator:
    """Build the released dynamic-snake conditional critic."""

    _, _, discriminator_config = _model_sections(config)
    resolved_device = torch.device(device)
    discriminator = Discriminator(
        resolved_device,
        channels=_integer(discriminator_config, "channels", "in_channels", 2),
        dim=_integer(discriminator_config, "dim", "base_dimension", 128),
    )
    return discriminator.to(resolved_device)


def build_models(
    config: Mapping[str, Any],
    device: str | torch.device,
) -> tuple[Generator, Discriminator]:
    """Build the generator and conditional critic on ``device``."""

    return build_generator(config, device), build_discriminator(config, device)


def architecture_metadata(
    generator: Generator,
    discriminator: Discriminator | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable details needed to interpret a checkpoint."""

    first_conv = generator.down1.conv_block.conv_block[0]
    output_conv = generator.conv_class
    latent = generator.viTrans.trans
    final_linear = latent.final_linear
    metadata: dict[str, Any] = {
        "architecture": "released_deepca",
        "generator_class": type(generator).__name__,
        "volume_size": int(generator.volume_size),
        "latent_volume_size": int(latent.vol_size),
        "input_channels": int(first_conv.in_channels),
        "output_channels": int(output_conv.out_channels),
        "base_filters": int(first_conv.out_channels),
        "latent_sequence_length": int(latent.sequence_length),
        "latent_projection_in_features": int(final_linear.in_features),
        "latent_projection_out_features": int(final_linear.out_features),
        "output_activation": "identity",
        "prediction_axis_order": "ZYX",
        "minimum_volume_size": 32,
        "volume_size_multiple": 16,
        "released_default_volume_size": 128,
    }
    if discriminator is not None:
        critic_input = discriminator.pre_module[0]
        metadata.update(
            {
                "discriminator_class": type(discriminator).__name__,
                "discriminator_input_channels": int(critic_input.in_channels),
                "discriminator_output_activation": "tanh",
                "discriminator_output_kind": "patch",
            }
        )
    return metadata


def _is_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(
            torch.is_tensor(item) or isinstance(item, nn.Parameter)
            for item in value.values()
        )
    )


def _strip_uniform_prefix(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor] | None:
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        return None
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def extract_generator_state_dict(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Extract generator weights from raw, released, or richer checkpoints.

    The released training script stores weights under ``network``.  Newer
    experiment checkpoints commonly use ``generator`` or a nested ``models``
    mapping.  Uniform DataParallel/combined-model prefixes are removed without
    changing any native generator key.
    """

    if _is_state_dict(checkpoint):
        candidate: Mapping[str, torch.Tensor] = checkpoint  # type: ignore[assignment]
    else:
        candidate_value: object | None = None
        models = checkpoint.get("models")
        if isinstance(models, Mapping):
            candidate_value = models.get("generator")
        if candidate_value is None:
            for key in (
                "generator",
                "network",
                "generator_state_dict",
                "model_state_dict",
                "state_dict",
                "model",
            ):
                if key in checkpoint:
                    candidate_value = checkpoint[key]
                    break
        if not _is_state_dict(candidate_value):
            available = ", ".join(sorted(str(key) for key in checkpoint))
            raise KeyError(
                "Checkpoint does not contain a recognizable generator state "
                f"dictionary. Available keys: {available or '<none>'}."
            )
        candidate = candidate_value  # type: ignore[assignment]

    for prefix in ("module.generator.", "generator."):
        stripped = _strip_uniform_prefix(candidate, prefix)
        if stripped is not None:
            return stripped
        matching = {
            key[len(prefix):]: value
            for key, value in candidate.items()
            if key.startswith(prefix)
        }
        if matching:
            return matching
    stripped = _strip_uniform_prefix(candidate, "module.")
    if stripped is not None:
        return stripped
    return candidate


def load_generator_state(
    generator: Generator,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool = True,
) -> nn.modules.module._IncompatibleKeys:
    """Load generator weights from any format accepted by the extractor."""

    return generator.load_state_dict(
        extract_generator_state_dict(checkpoint), strict=strict
    )
