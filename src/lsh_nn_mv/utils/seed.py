"""Reproducibility utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)
