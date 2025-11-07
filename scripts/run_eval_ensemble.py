"""Evaluate LSH ensemble."""

from __future__ import annotations

"""Evaluate LSH ensemble."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lsh_nn_mv.data.vision import get_vision_dataloaders
from lsh_nn_mv.data.tabular import get_tabular_dataloaders
from lsh_nn_mv.methods.hash_stats import summary_metrics
from lsh_nn_mv.models.ensemble import LSHEnsemble
from lsh_nn_mv.models.voters import MLPVoter, ConvVoter
from lsh_nn_mv.utils.config import ExperimentConfig
from lsh_nn_mv.utils.io import load_config, read_json, timestamp_dir
from lsh_nn_mv.utils.metrics import accuracy
from lsh_nn_mv.utils.plotting import plot_error_vs_n, plot_risk_disagreement

VISION_DATASETS = {"mnist", "cifar10"}
TABULAR_DATASETS = {"iris", "wine", "adult"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSH ensemble")
    parser.add_argument("--config", required=True)
    parser.add_argument("--voter", choices=["mlp", "conv"], required=True)
    parser.add_argument("--n-voters", type=int, required=True)
    parser.add_argument("--sigma", required=True, help="Sigma value or from:path")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_sigma(spec: str) -> float:
    if spec.startswith("from:"):
        payload = read_json(spec.split("from:")[1])
        return float(payload["sigma"])
    return float(spec)


def _predict_dataset(ensemble: LSHEnsemble, loader, num_classes: int, device: str):
    ensemble.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds.append(ensemble.predict(xb, num_classes).cpu())
        targets.append(yb.cpu())
    return torch.cat(preds), torch.cat(targets)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_dict(load_config(args.config))
    dataset_name = config.dataset.name.lower()
    sigma = _resolve_sigma(args.sigma)
    device = torch.device(args.device)

    if dataset_name in VISION_DATASETS:
        train_loader, val_loader, test_loader = get_vision_dataloaders(
            dataset=dataset_name,
            root=config.dataset.root or "./data",
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
            seed=config.seed,
        )
        first_batch = next(iter(train_loader))
        channels = first_batch[0].shape[1]
        image_size = first_batch[0].shape[2]
        num_classes = 10 if dataset_name in {"mnist", "cifar10"} else first_batch[1].max().item() + 1

        def voter_ctor(seed: int):
            return ConvVoter(
                input_channels=channels,
                num_classes=num_classes,
                sigma=sigma,
                seed=seed,
                image_size=image_size,
            )
    elif dataset_name in TABULAR_DATASETS:
        train_loader, val_loader, test_loader, meta = get_tabular_dataloaders(
            dataset=dataset_name,
            batch_size=config.dataset.batch_size,
            seed=config.seed,
        )
        num_classes = meta["num_classes"]
        def voter_ctor(seed: int):
            return MLPVoter(
                input_dim=meta["input_dim"],
                num_classes=num_classes,
                width=args.width,
                depth=args.depth,
                sigma=sigma,
                seed=seed,
            )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    ensemble = LSHEnsemble(voter_ctor=voter_ctor, n_voters=args.n_voters, voter_kwargs={}, device=device)

    train_preds, train_targets = _predict_dataset(ensemble, train_loader, num_classes, device)
    val_preds, val_targets = _predict_dataset(ensemble, val_loader, num_classes, device)
    test_preds, test_targets = _predict_dataset(ensemble, test_loader, num_classes, device)

    metrics = {
        "train_accuracy": accuracy(train_preds, train_targets),
        "val_accuracy": accuracy(val_preds, val_targets),
        "test_accuracy": accuracy(test_preds, test_targets),
    }

    g, d, c = summary_metrics(ensemble, val_loader, num_classes)
    metrics.update({"gibbs_risk": g, "disagreement": d, "c_bound": c})

    output_dir = timestamp_dir(Path(config.output_dir) / dataset_name, "ensemble")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_error_vs_n([args.n_voters], [1 - metrics["test_accuracy"]], [np.nan], output_dir / "error_vs_n.png")
    plot_risk_disagreement([args.n_voters], [g], [d], [c], output_dir / "risk_disagreement.png")

    print(f"Metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
