"""Utilities for deterministic train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class TrainValTestSplit:
    """Indices describing a train/validation/test split."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    @classmethod
    def from_sizes(
        cls, n_samples: int, train_frac: float, val_frac: float, seed: int
    ) -> "TrainValTestSplit":
        """Create a split from fractional allocations.

        Parameters
        ----------
        n_samples:
            Total number of samples.
        train_frac, val_frac:
            Fractions for the train and validation splits. The remainder is used
            for testing.
        seed:
            Random seed for reproducibility.
        """

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        n_train = int(n_samples * train_frac)
        n_val = int(n_samples * val_frac)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
        return cls(train_idx, val_idx, test_idx)

    def as_slices(self) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
        """Return the split as index sequences."""

        return self.train_indices, self.val_indices, self.test_indices

    def iter_splits(self) -> Iterable[np.ndarray]:
        """Iterate over the index splits."""

        yield self.train_indices
        yield self.val_indices
        yield self.test_indices
