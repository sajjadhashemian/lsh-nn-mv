"""Plotting utilities for experiment summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


plt.style.use("ggplot")


def plot_error_vs_n(ns: Sequence[int], errors: Sequence[float], hoeffding: Sequence[float], path: str | Path) -> None:
    fig, ax = plt.subplots()
    ax.semilogy(ns, errors, marker="o", label="Empirical error")
    ax.semilogy(ns, hoeffding, linestyle="--", label="Hoeffding heuristic")
    ax.set_xlabel("Number of voters (N)")
    ax.set_ylabel("Error rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_risk_disagreement(
    ns: Sequence[int],
    gibbs: Sequence[float],
    disagreement: Sequence[float],
    c_bound: Sequence[float],
    path: str | Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(ns, gibbs, marker="o", label="Gibbs risk")
    ax.plot(ns, disagreement, marker="s", label="Disagreement")
    ax.plot(ns, c_bound, marker="^", label="C-bound proxy")
    ax.set_xlabel("Number of voters (N)")
    ax.set_ylabel("Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_collision_curves(sigmas: Iterable[float], p_near: Iterable[float], p_far: Iterable[float], path: str | Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(list(sigmas), list(p_near), marker="o", label="p_near")
    ax.plot(list(sigmas), list(p_far), marker="s", label="p_far")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Collision probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
