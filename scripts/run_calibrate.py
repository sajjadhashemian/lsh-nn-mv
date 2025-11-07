"""CLI for sigma calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from lsh_nn_mv.data.vision import get_vision_dataloaders
from lsh_nn_mv.data.tabular import get_tabular_dataloaders
from lsh_nn_mv.methods.calibrate import calibrate_sigma
from lsh_nn_mv.utils.config import ExperimentConfig
from lsh_nn_mv.utils.io import load_config, timestamp_dir, write_json


VISION_DATASETS = {"mnist", "cifar10"}
TABULAR_DATASETS = {"iris", "wine", "adult"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate sigma for LSH voters")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--voter", choices=["mlp", "conv"], required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--grid-sigma", type=str, required=True)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dict = load_config(args.config)
    config = ExperimentConfig.from_dict(config_dict)

    sigma_grid = [float(x) for x in args.grid_sigma.split(",")]

    dataset_name = config.dataset.name.lower()
    seed = config.seed

    if dataset_name in VISION_DATASETS:
        train_loader, val_loader, _ = get_vision_dataloaders(
            dataset=dataset_name,
            root=config.dataset.root or "./data",
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
            seed=seed,
        )
        loader = val_loader
    elif dataset_name in TABULAR_DATASETS:
        train_loader, val_loader, _, _ = get_tabular_dataloaders(
            dataset=dataset_name,
            batch_size=config.dataset.batch_size,
            seed=seed,
        )
        loader = val_loader
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    result = calibrate_sigma(
        model_family=args.voter,
        data_loader=loader,
        r=args.r,
        c=args.c,
        tau=args.tau,
        grid_sigma=sigma_grid,
        max_pairs=args.max_pairs,
    )

    output_dir = args.output
    if output_dir is None:
        output_dir = timestamp_dir(Path(config.output_dir) / dataset_name, "calibration")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sigma": result.sigma,
        "score": result.score,
        "p_near": result.p_near,
        "p_far": result.p_far,
        "grid": sigma_grid,
        "params": {"r": args.r, "c": args.c, "tau": args.tau},
    }
    write_json(Path(output_dir) / "calibration.json", payload)
    print(f"Calibration complete. Best sigma={result.sigma:.4f}, score={result.score:.4f}")


if __name__ == "__main__":
    main()
