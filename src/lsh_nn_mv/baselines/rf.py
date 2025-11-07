"""Random forest baseline."""

from __future__ import annotations

from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier


@dataclass
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int | None = None


@dataclass
class RandomForestResult:
    accuracy: float


class RandomForestBaseline:
    def __init__(self, config: RandomForestConfig | None = None) -> None:
        self.config = config or RandomForestConfig()

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test) -> RandomForestResult:
        clf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return RandomForestResult(accuracy=clf.score(X_test, y_test))
