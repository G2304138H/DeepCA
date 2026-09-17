"""Training primitives for the released DeepCA conditional WGAN-GP."""

from __future__ import annotations

import contextlib
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int, deterministic: bool = True) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_grad_scaler(enabled: bool) -> Any:
    amp_module = getattr(torch, "amp", None)
    scaler_class = getattr(amp_module, "GradScaler", None)
    if scaler_class is not None:
        try:
            return scaler_class("cuda", enabled=enabled)
        except TypeError:
            return scaler_class(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool, device: torch.device) -> Any:
    if not enabled:
        return contextlib.nullcontext()
    amp_module = getattr(torch, "amp", None)
    autocast = getattr(amp_module, "autocast", None)
    if autocast is not None:
        try:
            return autocast(device_type=device.type, enabled=True)
        except TypeError:
            return autocast(device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def conditional_pair(condition: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    if condition.shape != volume.shape:
        raise ValueError(
            f"Condition and volume must have equal [B,1,D,H,W] shapes, got "
            f"{tuple(condition.shape)} and {tuple(volume.shape)}."
        )
    return torch.cat((condition, volume), dim=1)


def gradient_penalty(
    critic: torch.nn.Module,
    real_pair: torch.Tensor,
    fake_pair: torch.Tensor,
    *,
    weight: float = 10.0,
) -> torch.Tensor:
    """Published WGAN-GP norm over every non-batch conditional-pair element."""

    if real_pair.shape != fake_pair.shape:
        raise ValueError("real_pair and fake_pair must have equal shapes.")
    batch_size = real_pair.shape[0]
    eta = torch.rand(
        (batch_size,) + (1,) * (real_pair.ndim - 1),
        device=real_pair.device,
        dtype=real_pair.dtype,
    )
    interpolated = (eta * real_pair + (1.0 - eta) * fake_pair).requires_grad_(True)
    patch_scores = critic(interpolated)
    scores = patch_scores.reshape(batch_size, -1).mean(dim=1)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norms = gradients.reshape(batch_size, -1).norm(2, dim=1)
    return ((norms - 1.0) ** 2).mean() * float(weight)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter], config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    name = str(config.get("name", "adam")).lower()
    learning_rate = float(config.get("learning_rate", config.get("lr", 1.0e-4)))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name in {"adam", "adamw"}:
        betas = tuple(float(value) for value in config.get("betas", (0.5, 0.9)))
        if len(betas) != 2:
            raise ValueError("Adam betas must contain two values.")
        optimizer_type = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        return optimizer_type(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=float(config.get("eps", 1.0e-8)),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer {name!r}; choose adam, adamw, or sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Optional[Mapping[str, Any]]
) -> Optional[Any]:
    if not config or str(config.get("name", "none")).lower() == "none":
        return None
    name = str(config["name"]).lower()
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 50)),
            gamma=float(config.get("gamma", 0.5)),
        )
    if name in {"plateau", "reduce_on_plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 10)),
            min_lr=float(config.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler {name!r}; choose none, step, or plateau.")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    condition = batch["input"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
    if condition.ndim != 5 or condition.shape[1] != 1 or target.shape != condition.shape:
        raise ValueError(
            "Expected matching input/target tensors [B,1,D,H,W], got "
            f"{tuple(condition.shape)} and {tuple(target.shape)}."
        )
    return condition, target


def _finite(name: str, value: torch.Tensor, batch: Mapping[str, Any]) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(
            f"Non-finite {name} for cases {batch.get('case_id', 'unknown')}."
        )


def _mean_metrics(totals: Mapping[str, float], counts: Mapping[str, int]) -> dict[str, float]:
    return {
        name: (float(totals[name] / counts[name]) if counts[name] else math.nan)
        for name in totals
    }


def train_one_epoch(
    *,
    loader: Iterable[Mapping[str, Any]],
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    critic_steps_per_generator: int,
    l1_weight: float,
    gradient_penalty_weight: float,
    gradient_clip_norm: Optional[float],
    global_step: int,
) -> tuple[dict[str, float], int]:
    generator.train()
    critic.train()
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    critic_interval = int(critic_steps_per_generator)
    if critic_interval <= 0:
        raise ValueError("critic_steps_per_generator must be positive.")

    for batch_index, batch in enumerate(loader):
        condition, target = _move_batch(batch, device)
        batch_size = int(condition.shape[0])

        set_requires_grad(generator, False)
        set_requires_grad(critic, True)
        critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with autocast_context(amp_enabled, device):
                prediction = generator(condition)
        with autocast_context(amp_enabled, device):
            real_score = critic(conditional_pair(condition, target)).mean()
            fake_score = critic(conditional_pair(condition, prediction).detach()).mean()
        # Gradient penalty is evaluated in FP32 for stable second derivatives.
        with autocast_context(False, device):
            gp = gradient_penalty(
                critic,
                conditional_pair(condition.float(), target.float()),
                conditional_pair(condition.float(), prediction.float()).detach(),
                weight=gradient_penalty_weight,
            )
            critic_loss = fake_score.float() - real_score.float() + gp
        _finite("critic loss", critic_loss, batch)
        scaler.scale(critic_loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(critic_optimizer)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(gradient_clip_norm))
        scaler.step(critic_optimizer)
        scaler.update()

        totals["critic_loss"] += float(critic_loss.detach()) * batch_size
        totals["gradient_penalty"] += float(gp.detach()) * batch_size
        totals["wasserstein"] += float((real_score - fake_score).detach()) * batch_size
        counts["critic_loss"] += batch_size
        counts["gradient_penalty"] += batch_size
        counts["wasserstein"] += batch_size

        if (batch_index + 1) % critic_interval == 0:
            set_requires_grad(generator, True)
            set_requires_grad(critic, False)
            generator_optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled, device):
                prediction = generator(condition)
                adversarial = -critic(conditional_pair(condition, prediction)).mean()
                l1 = F.l1_loss(prediction, target)
                generator_loss = adversarial + float(l1_weight) * l1
            _finite("generator loss", generator_loss, batch)
            scaler.scale(generator_loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(generator_optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), float(gradient_clip_norm))
            scaler.step(generator_optimizer)
            scaler.update()
            totals["generator_loss"] += float(generator_loss.detach()) * batch_size
            totals["adversarial_loss"] += float(adversarial.detach()) * batch_size
            totals["l1_loss"] += float(l1.detach()) * batch_size
            counts["generator_loss"] += batch_size
            counts["adversarial_loss"] += batch_size
            counts["l1_loss"] += batch_size
        global_step += 1

    set_requires_grad(generator, True)
    set_requires_grad(critic, True)
    metrics = _mean_metrics(totals, counts)
    if counts["generator_loss"] == 0:
        raise RuntimeError(
            "No generator update occurred. Increase the number of training batches, "
            "disable drop_last, or reduce critic_steps_per_generator."
        )
    return metrics, global_step


@torch.no_grad()
def validate(
    *,
    loader: Iterable[Mapping[str, Any]],
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    l1_weight: float,
) -> dict[str, float]:
    generator.eval()
    critic.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        condition, target = _move_batch(batch, device)
        batch_size = int(condition.shape[0])
        with autocast_context(amp_enabled, device):
            prediction = generator(condition)
            adversarial = -critic(conditional_pair(condition, prediction)).mean()
            l1 = F.l1_loss(prediction, target)
            combined = adversarial + float(l1_weight) * l1
        _finite("validation L1", l1, batch)
        totals["l1_loss"] += float(l1) * batch_size
        totals["adversarial_loss"] += float(adversarial) * batch_size
        totals["combined_loss"] += float(combined) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("Validation loader produced no cases.")
    return {name: value / count for name, value in totals.items()}
