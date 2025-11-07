"""Calibration utilities for selecting sigma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader

from ..models.voters import MLPVoter, ConvVoter


@dataclass
class CalibrationResult:
    sigma: float
    score: float
    p_near: float
    p_far: float


def _sample_pairs(distances: torch.Tensor, threshold: float, greater: bool) -> torch.Tensor:
    if greater:
        mask = distances > threshold
    else:
        mask = distances <= threshold
    indices = mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return indices


def calibrate_sigma(
    model_family: str,
    data_loader: DataLoader,
    r: float,
    c: float,
    tau: float,
    grid_sigma: Iterable[float],
    max_pairs: int = 20000,
    device: str | torch.device = "cpu",
) -> CalibrationResult:
    """Calibrate sigma by maximising p_near - p_far subject to p_near >= tau."""

    device = torch.device(device)
    xs, ys = [], []
    for xb, yb in data_loader:
        xs.append(xb.to(device))
        ys.append(yb.to(device))
    data = torch.cat(xs, dim=0)
    labels = torch.cat(ys, dim=0)

    flat = data.view(data.size(0), -1)
    dist_matrix = torch.cdist(flat, flat, p=2)
    near_pairs = _sample_pairs(dist_matrix, r, greater=False)
    far_pairs = _sample_pairs(dist_matrix, c * r, greater=True)

    if near_pairs.numel() > max_pairs:
        perm = torch.randperm(near_pairs.size(0))[:max_pairs]
        near_pairs = near_pairs[perm]
    if far_pairs.numel() > max_pairs:
        perm = torch.randperm(far_pairs.size(0))[:max_pairs]
        far_pairs = far_pairs[perm]

    def voter_fn(sigma: float) -> torch.nn.Module:
        if model_family == "mlp":
            input_dim = flat.size(1)
            num_classes = len(torch.unique(labels))
            return MLPVoter(input_dim, num_classes, sigma=sigma)
        if model_family == "conv":
            num_classes = len(torch.unique(labels))
            channels = data.size(1)
            image_size = data.size(2)
            return ConvVoter(channels, num_classes, sigma=sigma, image_size=image_size)
        raise ValueError(f"Unknown model family: {model_family}")

    best_result = CalibrationResult(sigma=float("nan"), score=float("-inf"), p_near=0.0, p_far=1.0)
    for sigma in grid_sigma:
        if near_pairs.numel() == 0 or far_pairs.numel() == 0:
            break
        voter = voter_fn(sigma)
        voter.eval()
        with torch.no_grad():
            outputs = voter.hash(data)
        p_near = (outputs[near_pairs[:, 0]] == outputs[near_pairs[:, 1]]).float().mean().item()
        p_far = (outputs[far_pairs[:, 0]] == outputs[far_pairs[:, 1]]).float().mean().item()
        if p_near >= tau:
            score = p_near - p_far
            if score > best_result.score:
                best_result = CalibrationResult(sigma=sigma, score=score, p_near=p_near, p_far=p_far)
    return best_result
