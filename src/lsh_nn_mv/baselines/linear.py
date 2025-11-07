"""Linear model baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


@dataclass
class LinearConfig:
    Cs: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


@dataclass
class LinearResult:
    model: str
    C: float
    accuracy: float


class LinearBaselines:
    def __init__(self, config: LinearConfig) -> None:
        self.config = config

    def fit_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> List[LinearResult]:
        results: List[LinearResult] = []
        for C in self.config.Cs:
            lr = LogisticRegression(C=C, max_iter=1000, n_jobs=1)
            lr.fit(X_train, y_train)
            results.append(LinearResult("logistic_regression", C, lr.score(X_test, y_test)))

            svm = LinearSVC(C=C)
            svm.fit(X_train, y_train)
            results.append(LinearResult("linear_svm", C, svm.score(X_test, y_test)))
        return results
