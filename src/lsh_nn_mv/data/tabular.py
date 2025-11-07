"""Tabular dataset loaders and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..utils.seed import set_all_seeds

try:  # pragma: no cover - optional dependency in tests
    import openml
except Exception:  # pragma: no cover
    openml = None


def _preprocess_dataframe(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col])
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if X[c].dtype != "object"]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers)
    X_processed = preprocessor.fit_transform(X)
    return X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, y


def load_custom_csv(path: str | Path, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return _preprocess_dataframe(df, label_col)


def _load_openml(name: str) -> Tuple[np.ndarray, np.ndarray]:
    if openml is None:  # pragma: no cover - fallback when openml unavailable
        raise ImportError("openml is required to fetch tabular datasets")
    dataset = openml.datasets.get_dataset(name)
    df, *_ = dataset.get_dataframe()
    label_col = dataset.default_target_attribute
    return _preprocess_dataframe(df, label_col)


def get_tabular_dataloaders(
    dataset: str,
    batch_size: int,
    val_batch_size: int | None = None,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Return dataloaders for a tabular dataset."""

    set_all_seeds(seed)

    dataset_lower = dataset.lower()
    if dataset_lower == "iris":
        from sklearn.datasets import load_iris

        data = load_iris()
        X, y = data.data, data.target
        num_classes = len(np.unique(y))
    elif dataset_lower == "wine":
        from sklearn.datasets import load_wine

        data = load_wine()
        X, y = data.data, data.target
        num_classes = len(np.unique(y))
    elif dataset_lower == "adult":
        X, y = _load_openml("adult")
        y = (y == '>50K').astype(int)
        num_classes = 2
    else:
        raise ValueError(f"Unsupported tabular dataset: {dataset}")

    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_train = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        features = torch.from_numpy(X[idx]).float()
        labels = torch.from_numpy(y[idx]).long()
        dataset = TensorDataset(features, labels)
        return DataLoader(
            dataset,
            batch_size=batch_size if shuffle else (val_batch_size or batch_size),
            shuffle=shuffle,
        )

    train_loader = _make_loader(train_idx, True)
    val_loader = _make_loader(val_idx, False)
    test_loader = _make_loader(test_idx, False)
    meta = {"input_dim": X.shape[1], "num_classes": num_classes}
    return train_loader, val_loader, test_loader, meta
