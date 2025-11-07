"""Hash statistics for LSH ensembles."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from ..models.ensemble import LSHEnsemble


def gibbs_risk(ensemble: LSHEnsemble, loader: DataLoader, num_classes: int) -> float:
    ensemble.eval()
    losses = []
    for xb, yb in loader:
        probs = ensemble.predict_proba(xb, num_classes=num_classes)
        voter_error = 1.0 - probs[torch.arange(len(yb)), yb]
        losses.append(voter_error.mean().item())
    return float(sum(losses) / len(losses)) if losses else 0.0


def disagreement(ensemble: LSHEnsemble, loader: DataLoader) -> float:
    ensemble.eval()
    disagreements = []
    for xb, _ in loader:
        votes = ensemble.forward(xb)
        n_voters = votes.size(0)
        total_pairs = n_voters * (n_voters - 1) / 2
        diff = 0.0
        for i in range(n_voters):
            for j in range(i + 1, n_voters):
                diff += (votes[i] != votes[j]).float().mean().item()
        disagreements.append(diff / total_pairs if total_pairs > 0 else 0.0)
    return float(sum(disagreements) / len(disagreements)) if disagreements else 0.0


def c_bound_proxy(risk: float, disagreement_rate: float) -> float:
    denom = 1 - 2 * disagreement_rate
    if denom <= 0:
        return float("nan")
    return 1 - ((1 - 2 * risk) ** 2) / denom


def summary_metrics(
    ensemble: LSHEnsemble,
    loader: DataLoader,
    num_classes: int,
) -> Tuple[float, float, float]:
    g = gibbs_risk(ensemble, loader, num_classes)
    d = disagreement(ensemble, loader)
    c = c_bound_proxy(g, d)
    return g, d, c
