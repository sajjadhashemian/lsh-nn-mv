"""Adversarial attack utilities."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def _gradient_sign(model: Callable[[Tensor], Tensor], x: Tensor, y: Tensor) -> Tensor:
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    return x.grad.detach()


def fgsm(
    model: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    eps: float,
) -> Tensor:
    grad = _gradient_sign(model, x, y)
    adv = x + eps * grad.sign()
    return torch.clamp(adv, 0.0, 1.0)


def pgd(
    model: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    eps: float,
    steps: int,
    step_size: float,
) -> Tensor:
    original = x.clone().detach()
    adv = x.clone().detach()
    for _ in range(steps):
        grad = _gradient_sign(model, adv, y)
        adv = adv + step_size * grad.sign()
        perturbation = torch.clamp(adv - original, min=-eps, max=eps)
        adv = torch.clamp(original + perturbation, 0.0, 1.0)
    return adv
