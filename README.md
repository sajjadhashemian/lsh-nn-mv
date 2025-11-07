# LSH-NN-MV

This repository implements **LSH-NN-MV**, an ensemble of locality-sensitive hashing neural networks that aggregate predictions via majority vote. The project is fully reproducible, CPU friendly, and ships with configurable experiments, baselines, adversarial robustness checks, and theory-inspired diagnostics.

## Features

- Randomly initialised MLP and convolutional "voter" architectures.
- Data-driven calibration of the initialisation scale to encourage locality sensitivity.
- Majority-vote ensembles with persistence utilities.
- Baselines: k-NN (with optional PCA), logistic regression, linear SVM, trained MLP, and optional random forest.
- Unified tabular and vision dataset loaders (MNIST, CIFAR-10, Iris, Wine, Adult, custom CSV).
- Adversarial evaluation with FGSM and PGD.
- Rich experiment scripts that log CSV metrics and produce informative plots (test error vs. ensemble size, Gibbs risk, disagreement, C-bound proxy, calibration curves).
- Deterministic seeding utilities and simple configuration management via YAML.
- Fast unit tests that exercise key primitives.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, install the package in editable mode:

```bash
pip install -e .
```

## Quickstart

### Calibration

```bash
python -m scripts.run_calibrate --config configs/mnist.yaml \
  --voter mlp --r 0.5 --c 3.0 --tau 0.9 \
  --grid-sigma 0.25,0.5,1.0,1.5 --max-pairs 20000
```

### Baselines

```bash
python -m scripts.run_train_baselines --config configs/mnist.yaml
```

### Ensemble evaluation

```bash
python -m scripts.run_eval_ensemble --config configs/mnist.yaml \
  --voter mlp --n-voters 32 --sigma from:results/mnist/calibration.json \
  --width 512 --depth 2
```

### Adversarial robustness

```bash
python -m scripts.run_adversarial_eval --config configs/mnist.yaml \
  --n-voters 32 --attack fgsm --eps 0.1
```

### Ensemble size sweep

```bash
python -m scripts.run_sweep_N --config configs/mnist.yaml --N 1,2,4,8,16,32,64
```

## Repository Layout

```
lsh_nn_mv/
  README.md
  pyproject.toml
  requirements.txt
  src/lsh_nn_mv/
    data/
    models/
    methods/
    baselines/
    utils/
  scripts/
  configs/
  tests/
  results/
```

The `results/` directory is intentionally empty and will be populated at runtime with CSV logs and plots.

## Datasets

Dataset loaders rely on `torchvision` for vision datasets and `openml`/`pandas` for tabular data. All loaders share a common interface returning PyTorch dataloaders with deterministic seeding where feasible.

## Configuration

Example YAML configuration files for MNIST, CIFAR-10, and Adult datasets live under `configs/`. These set dataset-specific defaults (batch sizes, ensemble sizes, attack radii, etc.).

## Testing

Run the included unit tests with:

```bash
pytest
```

The tests focus on deterministic behaviour of voters, ensembles, calibration monotonicity on synthetic blobs, and the efficacy of adversarial attacks on a toy problem.

## Results & Interpretation

Scripts emit both CSV summaries and Matplotlib figures. Plots include:

- Test error vs. ensemble size with a heuristic Hoeffding bound overlay.
- Gibbs risk, disagreement, and C-bound proxy vs. ensemble size.
- Calibration collision curves (`p_near` and `p_far`).

Interpret the figures to assess how random LSH voters combine to form a robust classifier, how calibration influences locality, and how ensemble size impacts accuracy and robustness.

## License

This project is released under the MIT License (see `LICENSE`).
