"""Reproducible experiment script for theoretical evaluation of LSH-NN-MV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lsh_nn_mv.data.vision import get_vision_dataloaders
from lsh_nn_mv.methods.hash_stats import summary_metrics
from lsh_nn_mv.models.ensemble import LSHEnsemble
from lsh_nn_mv.models.voters import ConvVoter
from lsh_nn_mv.utils.io import timestamp_dir
from lsh_nn_mv.utils.metrics import accuracy, confusion_matrix
from lsh_nn_mv.utils.plotting import plot_error_vs_n, plot_risk_disagreement
from lsh_nn_mv.utils.seed import set_all_seeds


SUPPORTED_VISION_DATASETS = {"mnist", "cifar10"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end evaluation harness for LSH-NN-MV suitable for theoretical papers. "
            "The script instantiates random convolutional voters, aggregates them via majority "
            "vote, and produces publication-grade metrics and plots."
        )
    )
    parser.add_argument(
        "--datasets",
        default="mnist,cifar10",
        help="Comma separated list of datasets (supported: mnist, cifar10).",
    )
    parser.add_argument(
        "--n-voters",
        default="1,2,4,8,16",
        help="Comma separated ensemble sizes to evaluate.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.75,
        help="Initialisation scale for random voters controlling locality sensitivity.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used across train/validation/test loaders.",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root directory used for torchvision dataset downloads.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/theoretical",
        help="Directory in which experiment artefacts (CSV/JSON/plots) are stored.",
    )
    parser.add_argument("--device", default="cpu", help="Device identifier (cpu or cuda).")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of batches per split to reduce runtime during smoke tests.",
    )
    return parser.parse_args()


def _split_argument_list(spec: str) -> List[str]:
    return [item.strip().lower() for item in spec.split(",") if item.strip()]


@torch.no_grad()
def _collect_predictions(
    ensemble: LSHEnsemble,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    max_batches: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds, targets = [], []
    for batch_idx, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        xb = xb.to(device)
        yb = yb.to(device)
        preds.append(ensemble.predict(xb, num_classes).cpu())
        targets.append(yb.cpu())
    if not preds:
        raise RuntimeError("No batches were processed. Reduce --max-batches or check dataloader.")
    return torch.cat(preds), torch.cat(targets)


def _plot_accuracy_curves(
    ns: Sequence[int],
    train_acc: Sequence[float],
    val_acc: Sequence[float],
    test_acc: Sequence[float],
    path: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(ns, train_acc, marker="o", label="Train")
    ax.plot(ns, val_acc, marker="s", label="Validation")
    ax.plot(ns, test_acc, marker="^", label="Test")
    ax.set_xlabel("Number of voters (N)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="magma")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _hoeffding_proxy(ns: Sequence[int], sample_size: int) -> List[float]:
    return [math.exp(-2 * n * sample_size ** -0.5) for n in ns]


def _write_csv(path: Path, metrics: Iterable[dict]) -> None:
    metrics = list(metrics)
    if not metrics:
        return
    fieldnames = sorted({key for item in metrics for key in item.keys() if key != "confusion_matrix"})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            row = {k: v for k, v in item.items() if k != "confusion_matrix"}
            writer.writerow(row)


def _evaluate_dataset(
    dataset_name: str,
    ns: Sequence[int],
    sigma: float,
    batch_size: int,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
    max_batches: int | None,
) -> None:
    set_all_seeds(seed)
    train_loader, val_loader, test_loader = get_vision_dataloaders(
        dataset=dataset_name,
        root=data_root,
        batch_size=batch_size,
        num_workers=2,
        seed=seed,
    )

    first_batch = next(iter(train_loader))
    channels = first_batch[0].shape[1]
    image_size = first_batch[0].shape[2]
    num_classes = 10
    class_names = [str(i) for i in range(num_classes)]

    dataset_dir = timestamp_dir(output_dir / dataset_name, "experiment")

    metrics = []
    confusion_matrices = {}

    for n in ns:
        ensemble = LSHEnsemble(
            voter_ctor=lambda seed_val: ConvVoter(
                input_channels=channels,
                num_classes=num_classes,
                sigma=sigma,
                seed=seed_val,
                image_size=image_size,
            ),
            n_voters=n,
            voter_kwargs={},
            device=device,
            seed=seed,
        )

        train_preds, train_targets = _collect_predictions(
            ensemble, train_loader, num_classes, device, max_batches
        )
        val_preds, val_targets = _collect_predictions(
            ensemble, val_loader, num_classes, device, max_batches
        )
        test_preds, test_targets = _collect_predictions(
            ensemble, test_loader, num_classes, device, max_batches
        )

        g, d, c = summary_metrics(ensemble, val_loader, num_classes)
        cm = confusion_matrix(test_preds, test_targets, num_classes)

        entry = {
            "dataset": dataset_name,
            "n_voters": n,
            "train_accuracy": accuracy(train_preds, train_targets),
            "val_accuracy": accuracy(val_preds, val_targets),
            "test_accuracy": accuracy(test_preds, test_targets),
            "gibbs_risk": g,
            "disagreement": d,
            "c_bound_proxy": c,
        }
        metrics.append(entry)
        confusion_matrices[n] = cm

    json_path = dataset_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _write_csv(dataset_dir / "metrics.csv", metrics)

    ns_list = [m["n_voters"] for m in metrics]
    train_curve = [m["train_accuracy"] for m in metrics]
    val_curve = [m["val_accuracy"] for m in metrics]
    test_curve = [m["test_accuracy"] for m in metrics]

    errors = [1 - acc for acc in test_curve]
    hoeffding = _hoeffding_proxy(ns_list, sample_size=len(train_loader.dataset))

    plot_error_vs_n(ns_list, errors, hoeffding, dataset_dir / "error_vs_n.png")
    plot_risk_disagreement(
        ns_list,
        [m["gibbs_risk"] for m in metrics],
        [m["disagreement"] for m in metrics],
        [m["c_bound_proxy"] for m in metrics],
        dataset_dir / "risk_disagreement.png",
    )
    _plot_accuracy_curves(ns_list, train_curve, val_curve, test_curve, dataset_dir / "accuracy_curves.png")

    best_index = int(np.argmax(test_curve))
    best_n = ns_list[best_index]
    cm = confusion_matrices[best_n].numpy()
    _plot_confusion_matrix(cm, class_names, dataset_dir / f"confusion_matrix_N{best_n}.png")

    summary = {
        "dataset": dataset_name,
        "best_n": best_n,
        "best_test_accuracy": test_curve[best_index],
        "sigma": sigma,
        "n_candidates": list(ns_list),
        "output_dir": str(dataset_dir),
    }
    with (dataset_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{dataset_name}] results stored in {dataset_dir}")


def main() -> None:
    args = parse_args()
    datasets = _split_argument_list(args.datasets)
    for dataset in datasets:
        if dataset not in SUPPORTED_VISION_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {sorted(SUPPORTED_VISION_DATASETS)}")

    ns = [int(x) for x in _split_argument_list(args.n_voters)]
    if not ns:
        raise ValueError("At least one ensemble size must be provided via --n-voters.")

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    for dataset in datasets:
        _evaluate_dataset(
            dataset_name=dataset,
            ns=ns,
            sigma=args.sigma,
            batch_size=args.batch_size,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
            seed=args.seed,
            max_batches=args.max_batches,
        )


if __name__ == "__main__":
    main()
