"""k-NN baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class KNNConfig:
    ks: Tuple[int, ...] = (1, 3, 5, 11)
    metrics: Tuple[str, ...] = ("euclidean", "cosine")
    pca_components: Tuple[int, ...] = (0, 50, 100)


@dataclass
class KNNResult:
    k: int
    metric: str
    pca: int
    accuracy: float


class KNNBaseline:
    def __init__(self, config: KNNConfig) -> None:
        self.config = config

    def fit_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> List[KNNResult]:
        results: List[KNNResult] = []
        for k in self.config.ks:
            for metric in self.config.metrics:
                for pca_dim in self.config.pca_components:
                    X_train_proc, X_test_proc = X_train, X_test
                    if pca_dim and pca_dim < X_train.shape[1]:
                        pca = PCA(n_components=pca_dim, whiten=True)
                        X_train_proc = pca.fit_transform(X_train)
                        X_test_proc = pca.transform(X_test)
                    clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
                    clf.fit(X_train_proc, y_train)
                    acc = clf.score(X_test_proc, y_test)
                    results.append(KNNResult(k=k, metric=metric, pca=pca_dim, accuracy=acc))
        return results
