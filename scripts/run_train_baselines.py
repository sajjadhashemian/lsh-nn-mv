"""Train and evaluate baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lsh_nn_mv.data.vision import get_vision_dataloaders
from lsh_nn_mv.data.tabular import get_tabular_dataloaders
from lsh_nn_mv.baselines.knn import KNNBaseline, KNNConfig
from lsh_nn_mv.baselines.linear import LinearBaselines, LinearConfig
from lsh_nn_mv.baselines.rf import RandomForestBaseline
from lsh_nn_mv.models.trained import TrainedMLP
from lsh_nn_mv.utils.config import ExperimentConfig
from lsh_nn_mv.utils.io import load_config, timestamp_dir, write_json

VISION_DATASETS = {"mnist", "cifar10"}
TABULAR_DATASETS = {"iris", "wine", "adult"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--knn", action="store_true")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--rf", action="store_true")
    return parser.parse_args()


def _flatten_loader(loader):
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb.view(xb.size(0), -1).detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_dict(load_config(args.config))
    dataset_name = config.dataset.name.lower()

    if dataset_name in VISION_DATASETS:
        train_loader, val_loader, test_loader = get_vision_dataloaders(
            dataset=dataset_name,
            root=config.dataset.root or "./data",
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
            seed=config.seed,
        )
        X_train, y_train = _flatten_loader(train_loader)
        X_test, y_test = _flatten_loader(test_loader)
    elif dataset_name in TABULAR_DATASETS:
        train_loader, val_loader, test_loader, meta = get_tabular_dataloaders(
            dataset=dataset_name,
            batch_size=config.dataset.batch_size,
            seed=config.seed,
        )
        X_train, y_train = _flatten_loader(train_loader)
        X_test, y_test = _flatten_loader(test_loader)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    output_dir = timestamp_dir(Path(config.output_dir) / dataset_name, "baselines")
    results = {}

    if args.knn:
        knn = KNNBaseline(KNNConfig())
        res = knn.fit_and_evaluate(X_train, y_train, X_test, y_test)
        results["knn"] = [r.__dict__ for r in res]

    if args.linear:
        linear = LinearBaselines(LinearConfig())
        res = linear.fit_and_evaluate(X_train, y_train, X_test, y_test)
        results["linear"] = [r.__dict__ for r in res]

    if args.rf:
        rf = RandomForestBaseline()
        res = rf.fit_and_evaluate(X_train, y_train, X_test, y_test)
        results["random_forest"] = res.__dict__

    if args.mlp:
        num_classes = len(np.unique(y_train))
        model = TrainedMLP(input_dim=X_train.shape[1], num_classes=num_classes)
        tensor_train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
        )
        tensor_val = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
        )
        train_dl = torch.utils.data.DataLoader(tensor_train, batch_size=128, shuffle=True)
        val_dl = torch.utils.data.DataLoader(tensor_val, batch_size=128)
        model.fit(train_dl, val_dl, epochs=5)
        acc = (model.predict(torch.from_numpy(X_test).float()) == torch.from_numpy(y_test).long()).float().mean().item()
        results["trained_mlp"] = {"accuracy": acc}

    write_json(output_dir / "baselines.json", results)
    print(f"Baselines saved to {output_dir}")


if __name__ == "__main__":
    main()
