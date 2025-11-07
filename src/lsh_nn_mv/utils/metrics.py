"""Utility metrics for evaluation."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def accuracy(preds: Tensor, targets: Tensor) -> float:
    return (preds == targets).float().mean().item()


def confusion_matrix(preds: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, t in zip(preds, targets):
        cm[t.long(), p.long()] += 1
    return cm


def robust_metrics(clean_preds: Tensor, adv_preds: Tensor, targets: Tensor) -> Dict[str, float]:
    clean_acc = accuracy(clean_preds, targets)
    adv_acc = accuracy(adv_preds, targets)
    flip_rate = (clean_preds != adv_preds).float().mean().item()
    return {
        "clean_accuracy": clean_acc,
        "robust_accuracy": adv_acc,
        "flip_rate": flip_rate,
    }
