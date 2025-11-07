"""Adversarial evaluation for LSH ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lsh_nn_mv.data.vision import get_vision_dataloaders
from lsh_nn_mv.data.tabular import get_tabular_dataloaders
from lsh_nn_mv.methods.adversarial import fgsm, pgd
from lsh_nn_mv.models.ensemble import LSHEnsemble
from lsh_nn_mv.models.voters import MLPVoter, ConvVoter
from lsh_nn_mv.utils.config import ExperimentConfig
from lsh_nn_mv.utils.io import load_config, read_json, timestamp_dir
from lsh_nn_mv.utils.metrics import robust_metrics

VISION_DATASETS = {"mnist", "cifar10"}
TABULAR_DATASETS = {"iris", "wine", "adult"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--voter", choices=["mlp", "conv"], required=True)
    parser.add_argument("--n-voters", type=int, required=True)
    parser.add_argument("--sigma", required=True)
    parser.add_argument("--attack", choices=["fgsm", "pgd"], required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_sigma(spec: str) -> float:
    if spec.startswith("from:"):
        payload = read_json(spec.split("from:")[1])
        return float(payload["sigma"])
    return float(spec)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_dict(load_config(args.config))
    sigma = _resolve_sigma(args.sigma)
    dataset_name = config.dataset.name.lower()
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
        num_classes = 10

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
                sigma=sigma,
                width=config.model.width,
                depth=config.model.depth,
                seed=seed,
            )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    ensemble = LSHEnsemble(voter_ctor=voter_ctor, n_voters=args.n_voters, voter_kwargs={}, device=device)
    ensemble.eval()

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        logits = []
        for voter in ensemble:
            logits.append(voter(x.to(device)))
        return torch.stack(logits, dim=0).mean(dim=0)

    clean_preds, adv_preds, targets = [], [], []
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        clean_logits = logits_fn(xb)
        clean_pred = clean_logits.argmax(dim=-1)
        if args.attack == "fgsm":
            adv = fgsm(logits_fn, xb, yb, eps=args.eps)
        else:
            adv = pgd(logits_fn, xb, yb, eps=args.eps, steps=args.steps, step_size=args.step_size)
        adv_logits = logits_fn(adv)
        adv_pred = adv_logits.argmax(dim=-1)
        clean_preds.append(clean_pred.cpu())
        adv_preds.append(adv_pred.cpu())
        targets.append(yb.cpu())

    clean_preds = torch.cat(clean_preds)
    adv_preds = torch.cat(adv_preds)
    targets = torch.cat(targets)

    metrics = robust_metrics(clean_preds, adv_preds, targets)
    output_dir = timestamp_dir(Path(config.output_dir) / dataset_name, "adversarial")
    with (output_dir / "robustness.json").open("w", encoding="utf-8") as f:
        json.dump({
            "attack": args.attack,
            "eps": args.eps,
            "steps": args.steps,
            "step_size": args.step_size,
            "metrics": metrics,
        }, f, indent=2)
    print(f"Robust metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
