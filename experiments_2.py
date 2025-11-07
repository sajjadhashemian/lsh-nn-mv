from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    input_shape: Tuple[int, ...]
    num_classes: int
    baseline_transformer: Optional[StandardScaler]
    baseline_data: Tuple[np.ndarray, np.ndarray]


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


def _hidden_by_size(arch_size: str, base: Sequence[int]) -> Tuple[int, ...]:
    if arch_size == "s":
        return tuple(int(max(8, h // 2)) for h in base)
    if arch_size == "l":
        return tuple(int(h * 2) for h in base)
    return tuple(int(h) for h in base)


def _lstm_hidden_by_size(arch_size: str, base: int) -> int:
    if arch_size == "s":
        return max(16, base // 2)
    if arch_size == "l":
        return base * 2
    return base


def load_uci_knn(dataset: str = "wine", seed: int = 0) -> PreparedDataset:
    if dataset == "wine":
        data = load_wine()
    elif dataset == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unsupported UCI dataset: {dataset}")

    X = data.data.astype(np.float32)
    y = data.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)
    return PreparedDataset(
        X_train=X_train_sc,
        y_train=y_train,
        X_test=X_test_sc,
        y_test=y_test,
        input_shape=(X.shape[1],),
        num_classes=len(np.unique(y)),
        baseline_transformer=scaler,
        baseline_data=(X_train_sc, X_test_sc),
    )


def prepare_dataset(
    dataset_name: str,
    arch_family: str,
    arch_size: str,
    seed: int,
) -> Tuple[PreparedDataset, Callable[[int], List[Callable[[], nn.Module]]]]:
    arch_family = arch_family.lower()
    if dataset_name == "mnist":
        (Xtr, ytr), (Xte, yte), n_classes = load_mnist_like(
            dataset="mnist", limit=6000, seed=seed
        )
        if arch_family == "cnn":
            Xtr_flat = Xtr.reshape(len(Xtr), -1)
            Xte_flat = Xte.reshape(len(Xte), -1)
            scaler = StandardScaler()
            Xtr_knn = scaler.fit_transform(Xtr_flat).astype(np.float32)
            Xte_knn = scaler.transform(Xte_flat).astype(np.float32)
            prepared = PreparedDataset(
                X_train=Xtr,
                y_train=ytr,
                X_test=Xte,
                y_test=yte,
                input_shape=(1, 28, 28),
                num_classes=n_classes,
                baseline_transformer=scaler,
                baseline_data=(Xtr_knn, Xte_knn),
            )

            def builders_factory(num_voters: int) -> List[Callable[[], nn.Module]]:
                return make_cnn_builders(num_voters, n_classes)

            return prepared, builders_factory
        elif arch_family == "mlp":
            Xtr_flat = Xtr.reshape(len(Xtr), -1)
            Xte_flat = Xte.reshape(len(Xte), -1)
            scaler = StandardScaler()
            Xtr_scaled = scaler.fit_transform(Xtr_flat).astype(np.float32)
            Xte_scaled = scaler.transform(Xte_flat).astype(np.float32)
            hidden = _hidden_by_size(arch_size, (256, 256))
            prepared = PreparedDataset(
                X_train=Xtr_scaled,
                y_train=ytr,
                X_test=Xte_scaled,
                y_test=yte,
                input_shape=(Xtr_scaled.shape[1],),
                num_classes=n_classes,
                baseline_transformer=scaler,
                baseline_data=(Xtr_scaled, Xte_scaled),
            )

            def builders_factory(num_voters: int) -> List[Callable[[], nn.Module]]:
                return make_mlp_builders(
                    num_voters, Xtr_scaled.shape[1], n_classes, hidden
                )

            return prepared, builders_factory
        elif arch_family == "rnn":
            # Treat each image row as a timestep
            Xtr_seq = Xtr.reshape(len(Xtr), 28, 28)
            Xte_seq = Xte.reshape(len(Xte), 28, 28)
            scaler = StandardScaler()
            Xtr_flat = scaler.fit_transform(Xtr_seq.reshape(len(Xtr_seq), -1)).astype(
                np.float32
            )
            Xte_flat = scaler.transform(Xte_seq.reshape(len(Xte_seq), -1)).astype(
                np.float32
            )
            hidden = _lstm_hidden_by_size(arch_size, 128)
            prepared = PreparedDataset(
                X_train=Xtr_seq,
                y_train=ytr,
                X_test=Xte_seq,
                y_test=yte,
                input_shape=(28, 28),
                num_classes=n_classes,
                baseline_transformer=scaler,
                baseline_data=(Xtr_flat, Xte_flat),
            )

            def builders_factory(num_voters: int) -> List[Callable[[], nn.Module]]:
                return make_lstm_builders(num_voters, 28, n_classes, hidden)

            return prepared, builders_factory
        else:
            raise ValueError(f"Unsupported architecture {arch_family} for MNIST")
    elif dataset_name == "uci_wine":
        prepared = load_uci_knn(dataset="wine", seed=seed)
        if arch_family == "mlp":
            hidden = _hidden_by_size(arch_size, (128, 64))

            def builders_factory(num_voters: int) -> List[Callable[[], nn.Module]]:
                return make_mlp_builders(
                    num_voters, prepared.input_shape[0], prepared.num_classes, hidden
                )

            return prepared, builders_factory
        elif arch_family == "rnn":
            Xtr_seq = prepared.X_train.reshape(len(prepared.X_train), -1, 1)
            Xte_seq = prepared.X_test.reshape(len(prepared.X_test), -1, 1)
            hidden = _lstm_hidden_by_size(arch_size, 64)
            prepared_seq = PreparedDataset(
                X_train=Xtr_seq,
                y_train=prepared.y_train,
                X_test=Xte_seq,
                y_test=prepared.y_test,
                input_shape=(prepared.input_shape[0], 1),
                num_classes=prepared.num_classes,
                baseline_transformer=prepared.baseline_transformer,
                baseline_data=prepared.baseline_data,
            )

            def builders_factory(num_voters: int) -> List[Callable[[], nn.Module]]:
                return make_lstm_builders(
                    num_voters, 1, prepared_seq.num_classes, hidden
                )

            return prepared_seq, builders_factory
        else:
            raise ValueError(
                f"Architecture {arch_family} not supported for dataset {dataset_name}"
            )
    elif dataset_name == "uci_cancer":
        prepared = load_uci_knn(dataset="breast_cancer", seed=seed)
        hidden = _hidden_by_size(arch_size, (64, 64))

        def builders_factory(num_voters: int) -> List[Callable[[], nn.Module]]:
            return make_mlp_builders(
                num_voters, prepared.input_shape[0], prepared.num_classes, hidden
            )

        return prepared, builders_factory
    elif dataset_name == "sequence":
        (Xtr, ytr), (Xte, yte) = load_sequence_synthetic(
            n_samples=4000, T=30, D=8, n_classes=3, seed=seed
        )
        scaler = StandardScaler()
        Xtr_vec = scaler.fit_transform(Xtr.mean(axis=1)).astype(np.float32)
        Xte_vec = scaler.transform(Xte.mean(axis=1)).astype(np.float32)
        hidden = _lstm_hidden_by_size(arch_size, 64)
        prepared = PreparedDataset(
            X_train=Xtr,
            y_train=ytr,
            X_test=Xte,
            y_test=yte,
            input_shape=(Xtr.shape[1], Xtr.shape[2]),
            num_classes=len(np.unique(ytr)),
            baseline_transformer=scaler,
            baseline_data=(Xtr_vec, Xte_vec),
        )

        def builders_factory(num_voters: int) -> List[Callable[[], nn.Module]]:
            return make_lstm_builders(
                num_voters, Xtr.shape[-1], prepared.num_classes, hidden
            )

        return prepared, builders_factory
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# -----------------------
# Run one configuration
# -----------------------
def run_one(
    dataset_name: str,
    arch_family: str,
    arch_size: str,
    num_voters: int,
    aggregate: str,
    init_scheme: str,
    seed: int,
    device: str,
    out_dir: str,
) -> Dict[str, str]:
    set_all_seeds(seed)
    os.makedirs(out_dir, exist_ok=True)

    prepared, builders_factory = prepare_dataset(
        dataset_name=dataset_name,
        arch_family=arch_family,
        arch_size=arch_size,
        seed=seed,
    )
    builders = builders_factory(num_voters)

    knn = KNeighborsClassifier(n_neighbors=5)
    Xtr_knn, Xte_knn = prepared.baseline_data
    t0 = time.time()
    knn.fit(Xtr_knn, prepared.y_train)
    knn_acc = float(knn.score(Xte_knn, prepared.y_test))
    knn_time = time.time() - t0

    # Init distributions (same for all voters for simplicity)
    initd = InitDistribution(scheme=init_scheme, nonlinearity="relu")

    est = NeuralVoterEnsemble(
        voter_builders=builders,
        voter_init_distributions=initd,
        aggregate=aggregate,
        device=device,
        random_state=seed,
        epochs=5 if dataset_name not in {"mnist"} else 3,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        val_split=0.1,
        early_stopping_patience=2,
        class_weight=None,
        num_workers=0,
        sklearn_transformer=None,
        verbose=1 if num_voters <= 5 else 0,
    )

    t0 = time.time()
    est.fit(prepared.X_train, prepared.y_train)
    train_time_s = time.time() - t0

    acc = float(est.score(prepared.X_test, prepared.y_test))
    mets = est.metrics_on_dataset(prepared.X_test, prepared.y_test)
    gibbs = float(mets["gibbs_risk"])
    disag = float(mets["disagreement"])
    num_params = int(sum(est.voter_num_params_))

    row = {
        "dataset": dataset_name,
        "arch_family": arch_family,
        "arch_size": arch_size,
        "num_params": str(num_params),
        "num_voters": str(num_voters),
        "init_scheme": init_scheme,
        "aggregate": aggregate,
        "seed": str(seed),
        "train_time_s": f"{train_time_s:.4f}",
        "accuracy": f"{acc:.6f}",
        "knn_accuracy": f"{knn_acc:.6f}",
        "knn_time_s": f"{knn_time:.4f}",
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
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "uci_wine"],
        help="Datasets to evaluate (mnist, uci_wine, uci_cancer, sequence)",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        default=["cnn", "mlp", "rnn"],
        help="Architectures to evaluate",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--voters", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument(
        "--aggregates",
        type=str,
        nargs="+",
        default=["mean_logits", "argmax"],
    )
    parser.add_argument("--arch_sizes", type=str, nargs="+", default=["s", "m"])
    parser.add_argument(
        "--init_schemes",
        type=str,
        nargs="+",
        default=["kaiming_normal", "kaiming_uniform"],
    )
    args = parser.parse_args()

    print(f"[env] {env_summary()}")
    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for dataset_name in args.datasets:
        for arch_family in args.architectures:
            for arch_size in args.arch_sizes:
                for init_scheme in args.init_schemes:
                    skip_combo = False
                    for seed in args.seeds:
                        if skip_combo:
                            break
                        for nv in args.voters:
                            if skip_combo:
                                break
                            for agg in args.aggregates:
                                try:
                                    row = run_one(
                                        dataset_name=dataset_name,
                                        arch_family=arch_family,
                                        arch_size=arch_size,
                                        num_voters=nv,
                                        aggregate=agg,
                                        init_scheme=init_scheme,
                                        seed=seed,
                                        device=args.device,
                                        out_dir=args.out_dir,
                                    )
                                except ValueError as exc:
                                    warnings.warn(
                                        f"Skipping configuration (dataset={dataset_name}, arch={arch_family}, size={arch_size}) due to: {exc}"
                                    )
                                    skip_combo = True
                                    break
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
                "knn_time_s",
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
