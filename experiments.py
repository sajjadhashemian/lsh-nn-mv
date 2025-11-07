# ============================
# Code Block 2 — Experiments: run_experiments.py
# ============================
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone as sk_clone

# Import estimator (assumes both files are in same directory or PYTHONPATH)
from neural_voter_ensemble import (
    NeuralVoterEnsemble,
    InitDistribution,
    set_all_seeds,
    _to_tensor,
)


# -----------------------
# Version / environment
# -----------------------
def env_summary() -> str:
    import sklearn

    return (
        f"Python {sys.version.split()[0]} | "
        f"torch {torch.__version__} | "
        f"sklearn {sklearn.__version__} | "
        f"cuda {'available' if torch.cuda.is_available() else 'not-available'}"
    )


# -----------------------
# Architectures
# -----------------------
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect (B,1,28,28)
        z = self.net(x)
        z = z.view(z.size(0), -1)
        return self.fc(z)


class TinyLSTM(nn.Module):
    def __init__(
        self, input_dim: int, out_dim: int, hidden_size: int = 64, num_layers: int = 1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect (B, T, D)
        out, _ = self.lstm(x)
        # last timestep
        last = out[:, -1, :]
        return self.fc(last)


# -----------------------
# Datasets (small & quick)
# -----------------------
def load_tabular(
    n_samples: int = 3000, n_features: int = 30, n_classes: int = 3, seed: int = 0
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 3),
        n_redundant=0,
        n_classes=n_classes,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return (X_train, y_train), (X_test, y_test)


def load_mnist_like(dataset: str = "mnist", limit: int = 6000, seed: int = 0):
    # Use torchvision but guard if unavailable
    try:
        import torchvision
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError("torchvision is required for image datasets.") from e

    torch.manual_seed(seed)
    tfm = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        dtrain = torchvision.datasets.MNIST(
            root="./data", train=True, transform=tfm, download=True
        )
        dtest = torchvision.datasets.MNIST(
            root="./data", train=False, transform=tfm, download=True
        )
        n_classes = 10
    else:
        dtrain = torchvision.datasets.FashionMNIST(
            root="./data", train=True, transform=tfm, download=True
        )
        dtest = torchvision.datasets.FashionMNIST(
            root="./data", train=False, transform=tfm, download=True
        )
        n_classes = 10

    # Subsample for speed
    idx = torch.randperm(len(dtrain))[:limit]
    Xtr = torch.stack([dtrain[i][0] for i in idx])  # (N,1,28,28)
    ytr = torch.tensor([dtrain[i][1] for i in idx])
    Xte = torch.stack(
        [dtest[i][0] for i in range(min(len(dtest), max(2000, limit // 3)))]
    )
    yte = torch.tensor(
        [dtest[i][1] for i in range(min(len(dtest), max(2000, limit // 3)))]
    )

    return (Xtr.numpy(), ytr.numpy()), (Xte.numpy(), yte.numpy()), n_classes


def load_sequence_synthetic(
    n_samples: int = 4000, T: int = 30, D: int = 8, n_classes: int = 3, seed: int = 0
):
    """
    Simple synthetic sequence: class = argmax over which coordinate has largest cumulative sum + noise.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, T, D).astype(np.float32)
    scores = X.sum(axis=1)  # (n, D)
    labels = scores.argmax(axis=1) % n_classes
    # reduce to n_classes by mapping classes
    labels = labels % n_classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=seed
    )
    return (X_train, y_train), (X_test, y_test)


# -----------------------
# Heuristic bound
# -----------------------
def heuristic_bound(
    num_voters: int,
    sample_size: int,
    gibbs_risk: float,
    disagreement: float,
    delta: float = 0.05,
) -> float:
    """
    Return a heuristic upper bound on the majority-vote (plurality) risk.

    We expose a PAC-Bayes C-bound–style surrogate using empirical Gibbs risk R_G and disagreement D:
        R_MV  \le  1 - (1 - 2 * R_G)^2 / (1 - 2 * D + epsilon)

    where epsilon is a tiny stabilizer (1e-12). To make it data-dependent, we add
    a Hoeffding-style concentration term to R_G and to D:

        R_G_hat_plus = R_G + sqrt(ln(2/delta)/(2 * sample_size))
        D_hat_plus   = D   + sqrt(ln(2/delta)/(2 * sample_size))

    Finally we clip to [0, 1]. This function is *pluggable*: if you want to replace it by a bound
    directly from your paper (e.g., exp(-2N(p1-1/2)^2) when you can estimate p1), you can edit here.

    Notes:
    - num_voters appears implicitly in the margin amplification, but not explicitly in this C-bound form.
      If you prefer an explicit N-dependence (e.g., via a margin estimator), you can adapt accordingly.
    """
    eps = 1e-12
    alpha = math.sqrt(math.log(2.0 / max(delta, 1e-12)) / (2.0 * max(1, sample_size)))
    rG = min(1.0, max(0.0, gibbs_risk + alpha))
    Dp = min(0.5, max(0.0, disagreement + alpha))  # disagreement ∈ [0, 0.5] for K>=2
    denom = max(eps, 1.0 - 2.0 * Dp)
    val = 1.0 - ((1.0 - 2.0 * rG) ** 2) / denom
    return float(min(1.0, max(0.0, val)))


# -----------------------
# Plot helpers
# -----------------------
def save_plot_disagreement_vs_gibbs(
    csv_rows: List[Dict[str, str]], out_png: str, out_csv: str
):
    import pandas as pd

    df = pd.DataFrame(csv_rows)
    df["gibbs_risk"] = df["gibbs_risk"].astype(float)
    df["disagreement"] = df["disagreement"].astype(float)

    plt.figure()
    families = sorted(df["arch_family"].unique())
    for fam in families:
        sub = df[df["arch_family"] == fam]
        plt.scatter(sub["disagreement"], sub["gibbs_risk"], label=fam, alpha=0.8)
    plt.xlabel("Pairwise disagreement")
    plt.ylabel("Gibbs risk")
    plt.legend()
    plt.title("Disagreement vs Gibbs risk")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    df.to_csv(out_csv, index=False)


def save_plot_accuracy_vs_voters(
    csv_rows: List[Dict[str, str]], out_png: str, out_csv: str
):
    import pandas as pd

    df = pd.DataFrame(csv_rows)
    # choose a single dataset/arch/aggregate for a clean plot (or facet if desired)
    # Here we aggregate by num_voters median accuracy and draw a bound curve using median gibbs/disagreement
    df["num_voters"] = df["num_voters"].astype(int)
    df["accuracy"] = df["accuracy"].astype(float)
    df["gibbs_risk"] = df["gibbs_risk"].astype(float)
    df["disagreement"] = df["disagreement"].astype(float)
    df["seed"] = df["seed"].astype(int)

    key_cols = ["dataset", "arch_family", "aggregate"]
    if len(df) == 0:
        return
    # pick most frequent triplet
    triplet = (
        df.groupby(key_cols).size().sort_values(ascending=False).index[0]
        if len(df.groupby(key_cols)) > 0
        else (df["dataset"].iloc[0], df["arch_family"].iloc[0], df["aggregate"].iloc[0])
    )
    sub = df[
        (df["dataset"] == triplet[0])
        & (df["arch_family"] == triplet[1])
        & (df["aggregate"] == triplet[2])
    ]
    # aggregate
    agg = (
        sub.groupby("num_voters")
        .agg(
            accuracy=("accuracy", "median"),
            gibbs_risk=("gibbs_risk", "median"),
            disagreement=("disagreement", "median"),
            sample_size=("seed", "count"),
        )
        .reset_index()
    )

    plt.figure()
    plt.plot(agg["num_voters"], agg["accuracy"], marker="o", label="Accuracy")
    # heuristic bound curve
    bound_vals = [
        1.0 - heuristic_bound(int(n), int(s), float(rg), float(d))
        for n, s, rg, d in zip(
            agg["num_voters"],
            agg["sample_size"],
            agg["gibbs_risk"],
            agg["disagreement"],
        )
    ]
    plt.plot(
        agg["num_voters"],
        bound_vals,
        marker="x",
        linestyle="--",
        label="Heuristic bound (1 - R_MV^bound)",
    )
    plt.xlabel("# voters")
    plt.ylabel("Accuracy / Bound")
    plt.title(f"Accuracy vs #voters [{triplet[0]} | {triplet[1]} | {triplet[2]}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    sub.to_csv(out_csv, index=False)


# -----------------------
# Builders for sweeps
# -----------------------
def make_mlp_builders(
    num_voters: int, in_dim: int, out_dim: int, hidden: Tuple[int, ...]
) -> List[Callable[[], nn.Module]]:
    return [
        lambda hd=hidden: MLP(in_dim, out_dim, hidden=hd) for _ in range(num_voters)
    ]


def make_cnn_builders(num_voters: int, out_dim: int) -> List[Callable[[], nn.Module]]:
    return [lambda: SimpleCNN(out_dim) for _ in range(num_voters)]


def make_lstm_builders(
    num_voters: int, in_dim: int, out_dim: int, hidden: int
) -> List[Callable[[], nn.Module]]:
    return [
        lambda hs=hidden: TinyLSTM(in_dim, out_dim, hidden_size=hs)
        for _ in range(num_voters)
    ]


# -----------------------
# Run one configuration
# -----------------------
def run_one(
    dataset_name: str,
    arch_family: str,
    arch_size: str,
    num_voters: int,
    aggregate: str,
    seed: int,
    device: str,
    out_dir: str,
) -> Dict[str, str]:
    set_all_seeds(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Load data and choose builders
    if dataset_name == "tabular":
        (Xtr, ytr), (Xte, yte) = load_tabular(
            n_samples=3000, n_features=30, n_classes=3, seed=seed
        )
        scaler = StandardScaler()
        in_dim = Xtr.shape[1]
        out_dim = len(np.unique(ytr))
        # hidden sizes by arch_size
        hidden = (
            (128, 128)
            if arch_size == "m"
            else (64,) if arch_size == "s" else (256, 256, 256)
        )
        builders = make_mlp_builders(num_voters, in_dim, out_dim, hidden)
        transformer = scaler
        knn_Xtr = scaler.fit_transform(Xtr)
        knn_Xte = scaler.transform(Xte)
        knn = KNeighborsClassifier(n_neighbors=5)
        t0 = time.time()
        knn.fit(knn_Xtr, ytr)
        knn_acc = float(knn.score(knn_Xte, yte))
        knn_time = time.time() - t0

    elif dataset_name in ("mnist", "fashion"):
        (Xtr, ytr), (Xte, yte), n_classes = load_mnist_like(
            dataset=dataset_name, limit=6000, seed=seed
        )
        out_dim = n_classes
        builders = make_cnn_builders(num_voters, out_dim)
        transformer = None
        # knn baseline: flatten + standardize
        Xtr_flat = Xtr.reshape(len(Xtr), -1)
        Xte_flat = Xte.reshape(len(Xte), -1)
        scaler = StandardScaler()
        knn_Xtr = scaler.fit_transform(Xtr_flat)
        knn_Xte = scaler.transform(Xte_flat)
        knn = KNeighborsClassifier(n_neighbors=3)
        t0 = time.time()
        knn.fit(knn_Xtr, ytr)
        knn_acc = float(knn.score(knn_Xte, yte))
        knn_time = time.time() - t0

    elif dataset_name == "sequence":
        (Xtr, ytr), (Xte, yte) = load_sequence_synthetic(
            n_samples=4000, T=30, D=8, n_classes=3, seed=seed
        )
        in_dim = Xtr.shape[-1]
        out_dim = len(np.unique(ytr))
        hidden = 64 if arch_size == "s" else 128 if arch_size == "m" else 256
        builders = make_lstm_builders(num_voters, in_dim, out_dim, hidden)
        transformer = None
        # knn baseline: mean-pool over time then standardize
        Xtr_vec = Xtr.mean(axis=1)
        Xte_vec = Xte.mean(axis=1)
        scaler = StandardScaler()
        knn_Xtr = scaler.fit_transform(Xtr_vec)
        knn_Xte = scaler.transform(Xte_vec)
        knn = KNeighborsClassifier(n_neighbors=5)
        t0 = time.time()
        knn.fit(knn_Xtr, ytr)
        knn_acc = float(knn.score(knn_Xte, yte))
        knn_time = time.time() - t0

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Init distributions (same for all voters for simplicity)
    initd = InitDistribution(scheme="kaiming_normal", nonlinearity="relu")

    est = NeuralVoterEnsemble(
        voter_builders=builders,
        voter_init_distributions=initd,
        aggregate=aggregate,
        device=device,
        random_state=seed,
        epochs=5 if dataset_name != "mnist" and dataset_name != "fashion" else 3,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        val_split=0.1,
        early_stopping_patience=2,
        class_weight=None,
        num_workers=0,
        sklearn_transformer=transformer,
        verbose=1 if num_voters <= 5 else 0,
    )

    t0 = time.time()
    est.fit(Xtr, ytr)
    train_time_s = time.time() - t0

    acc = float(est.score(Xte, yte))
    mets = est.metrics_on_dataset(Xte, yte)
    gibbs = float(mets["gibbs_risk"])
    disag = float(mets["disagreement"])
    num_params = int(sum(est.voter_num_params_))

    row = {
        "dataset": dataset_name,
        "arch_family": arch_family,
        "arch_size": arch_size,
        "num_params": str(num_params),
        "num_voters": str(num_voters),
        "init_scheme": "kaiming_normal",
        "aggregate": aggregate,
        "seed": str(seed),
        "train_time_s": f"{train_time_s:.4f}",
        "accuracy": f"{acc:.6f}",
        "knn_accuracy": f"{knn_acc:.6f}",
        "gibbs_risk": f"{gibbs:.6f}",
        "disagreement": f"{disag:.6f}",
    }
    return row


# -----------------------
# Main sweep
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Neural Voter Ensemble Experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tabular",
        choices=["tabular", "mnist", "fashion", "sequence"],
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--voters", type=int, nargs="+", default=[1, 3, 5, 9, 15])
    parser.add_argument(
        "--aggregates",
        type=str,
        nargs="+",
        default=["mean_logits", "mean_probs", "argmax"],
    )
    parser.add_argument("--arch_size", type=str, default="m", choices=["s", "m", "l"])
    args = parser.parse_args()

    print(f"[env] {env_summary()}")
    os.makedirs(args.out_dir, exist_ok=True)

    # choose arch_family from dataset
    arch_family = {
        "tabular": "MLP",
        "mnist": "CNN",
        "fashion": "CNN",
        "sequence": "LSTM",
    }[args.dataset]

    rows: List[Dict[str, str]] = []
    for seed in args.seeds:
        for nv in args.voters:
            for agg in args.aggregates:
                row = run_one(
                    dataset_name=args.dataset,
                    arch_family=arch_family,
                    arch_size=args.arch_size,
                    num_voters=nv,
                    aggregate=agg,
                    seed=seed,
                    device=args.device,
                    out_dir=args.out_dir,
                )
                rows.append(row)
                print("[row]", row)

    # write consolidated CSV
    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "arch_family",
                "arch_size",
                "num_params",
                "num_voters",
                "init_scheme",
                "aggregate",
                "seed",
                "train_time_s",
                "accuracy",
                "knn_accuracy",
                "gibbs_risk",
                "disagreement",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[write] {csv_path}")

    # plots
    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)
    save_plot_disagreement_vs_gibbs(
        rows,
        out_png=os.path.join(args.out_dir, "plots", "disagreement_vs_gibbs.png"),
        out_csv=os.path.join(args.out_dir, "plots", "disagreement_vs_gibbs.csv"),
    )
    save_plot_accuracy_vs_voters(
        rows,
        out_png=os.path.join(args.out_dir, "plots", "accuracy_vs_voters.png"),
        out_csv=os.path.join(args.out_dir, "plots", "accuracy_vs_voters_data.csv"),
    )
    print("[done]")


if __name__ == "__main__":
    main()
