from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from sklearn.base import clone as sk_clone

ArrayLike = Union[np.ndarray, torch.Tensor]
AggregatorName = Literal["argmax", "mean_logits", "mean_probs", "ceil", "ceil_sum"]


# --------------------
# Utilities
# --------------------
def _to_tensor(
    x: ArrayLike, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
        if t.dtype.is_floating_point is False:
            # for inputs like long ints (e.g., token ids), don't force to float unless 2D+ real features expected
            return t.to(device)
        return t.to(device=device, dtype=dtype)
    x = np.asarray(x)
    if x.dtype.kind in ("i", "u", "b"):
        return torch.from_numpy(x).to(device)
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def set_all_seeds(seed: int) -> None:
    # Best-effort determinism
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def default_init_(
    m: nn.Module, scheme: str = "kaiming_normal", nonlinearity: str = "relu"
) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        if scheme == "kaiming_normal":
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        elif scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        elif scheme == "xavier_uniform":
            nn.init.xavier_uniform_(m.weight)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(m.weight)
        else:
            # no-op or user-specified handled elsewhere
            pass
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@dataclass
class InitDistribution:
    """
    Describes how to initialize a voter.
    - scheme: one of {'kaiming_normal','kaiming_uniform','xavier_uniform','xavier_normal','none'}
    - nonlinearity: e.g., 'relu'
    - custom_init: optional callable(module) -> None that overrides scheme if provided
    """

    scheme: str = "kaiming_normal"
    nonlinearity: str = "relu"
    custom_init: Optional[Callable[[nn.Module], None]] = None

    def apply(self, module: nn.Module) -> None:
        if self.custom_init is not None:
            self.custom_init(module)
        else:
            module.apply(
                lambda m: default_init_(
                    m, scheme=self.scheme, nonlinearity=self.nonlinearity
                )
            )


def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def _plurality_vote(preds: torch.Tensor) -> torch.Tensor:
    # preds: (V, N) int64
    # returns (N,) by breaking ties with smallest class index (torch mode does this deterministically)
    mode_vals, _ = torch.mode(preds, dim=0)
    return mode_vals


def _aggregate_logits(
    logits_stack: torch.Tensor, aggregate: AggregatorName | Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits_stack: (V, N, C)
    returns: (scores_for_argmax: (N,C), probs_for_predict_proba: (N,C))
    """
    if callable(aggregate):
        scores = aggregate(logits_stack)  # expect (N, C)
        if scores.dim() != 2:
            raise ValueError("Custom aggregator must return (N, C) scores.")
        probs = _softmax_logits(scores)
        return scores, probs

    if aggregate == "mean_logits":
        scores = logits_stack.mean(dim=0)  # (N, C)
        probs = _softmax_logits(scores)
        return scores, probs
    elif aggregate == "mean_probs":
        probs = _softmax_logits(logits_stack).mean(dim=0)  # (N, C)
        # Scores that respect the probability ordering (use log)
        scores = torch.log(probs.clamp_min(1e-12))
        return scores, probs
    elif aggregate in ("ceil", "ceil_sum"):
        # Interpret as summing logits then elementwise ceil before argmax (toy/ablation aggregator)
        summed = logits_stack.sum(dim=0)
        scores = torch.ceil(summed)
        probs = _softmax_logits(summed)
        return scores, probs
    elif aggregate == "argmax":
        # Special case handled in predict: plurality over per-voter argmax.
        # For predict_proba, we will return mean_probs.
        probs = _softmax_logits(logits_stack).mean(dim=0)
        scores = torch.log(probs.clamp_min(1e-12))
        return scores, probs
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")


# --------------------
# Core Estimator
# --------------------
class NeuralVoterEnsemble(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible ensemble whose voters are user-supplied PyTorch neural networks.

    Parameters
    ----------
    voter_builders : list[Callable[[], nn.Module]] | Callable[[], list[nn.Module]]
        Factories that build voters (heterogeneous allowed). If a callable is provided,
        it is called at fit-time to obtain the list.
    voter_init_distributions : list[InitDistribution] | InitDistribution | None
        Initialization distribution(s). If a single InitDistribution is given, it is applied to all voters.
        If a list is given, its length must match the number of voters.
    aggregate : {"argmax","mean_logits","mean_probs","ceil","ceil_sum"} | Callable
        Aggregation rule across voter outputs (logits). If callable, it must accept a
        tensor of shape (V, N, C) with logits and return (N, C)-shaped scores.
        For "argmax", predictions are plurality over per-voter argmax; predict_proba returns mean of probabilities.
        For "mean_logits", predictions are argmax of averaged logits; predict_proba is softmax(mean logits).
        For "mean_probs", predictions are argmax of averaged probabilities; predict_proba is that same average.
    device : {"auto","cpu","cuda"}
        Device selection. "auto" picks CUDA if available.
    random_state : int | None
        Seed for reproducibility (numpy + torch + python).
    epochs : int
        Training epochs per voter.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate (Adam).
    weight_decay : float
        Adam weight decay.
    val_split : float
        Fraction of training data used for validation (for early stopping). 0 disables validation.
    early_stopping_patience : int
        Stop if validation doesn't improve for these many epochs. Ignored if val_split == 0.
    class_weight : "balanced" | None | np.ndarray
        Class weighting for cross-entropy.
    num_workers : int
        DataLoader workers (set 0 for determinism).
    """

    def __init__(
        self,
        voter_builders: Union[
            List[Callable[[], nn.Module]],
            Callable[[], List[Callable[[], nn.Module]]],
            Callable[[], List[nn.Module]],
        ],
        voter_init_distributions: Optional[
            Union[InitDistribution, List[InitDistribution]]
        ] = None,
        aggregate: Union[AggregatorName, Callable] = "mean_logits",
        device: Literal["auto", "cpu", "cuda"] = "auto",
        random_state: Optional[int] = 0,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        val_split: float = 0.1,
        early_stopping_patience: int = 5,
        class_weight: Optional[Union[str, np.ndarray]] = None,
        num_workers: int = 0,
        # optional sklearn transformer for preprocessing tabular X
        sklearn_transformer: Optional[Any] = None,
        verbose: int = 0,
    ):
        self.voter_builders = voter_builders
        self.voter_init_distributions = voter_init_distributions
        self.aggregate = aggregate
        self.device = device
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.early_stopping_patience = early_stopping_patience
        self.class_weight = class_weight
        self.num_workers = num_workers
        self.sklearn_transformer = sklearn_transformer
        self.verbose = verbose

    # BaseEstimator supplies get_params / set_params via __init__ signature.

    # -------------
    # API: fit/predict/etc.
    # -------------
    def fit(self, X: ArrayLike, y: ArrayLike) -> "NeuralVoterEnsemble":
        rs = check_random_state(self.random_state)
        if self.random_state is not None:
            set_all_seeds(int(self.random_state))

        # device selection
        if self.device == "auto":
            self._device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device_ = torch.device(self.device)

        # encode labels
        self._le_: LabelEncoder = LabelEncoder()
        y_np = np.asarray(y)
        self._le_.fit(y_np)
        y_enc = self._le_.transform(y_np)
        self._n_classes_ = int(len(self._le_.classes_))

        # optional preprocessing
        self._preprocess_ = None
        X_np = (
            np.asarray(X)
            if not isinstance(X, torch.Tensor)
            else X.detach().cpu().numpy()
        )
        if self.sklearn_transformer is not None:
            self._preprocess_ = sk_clone(self.sklearn_transformer)
            X_np = self._preprocess_.fit_transform(X_np, y_enc)

        X_t = _to_tensor(X_np, self._device_)
        y_t = _to_tensor(y_enc, self._device_, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)

        # validation split
        if self.val_split and self.val_split > 0.0:
            n_total = len(dataset)
            n_val = max(1, int(round(self.val_split * n_total)))
            n_train = n_total - n_val
            gen = torch.Generator().manual_seed(int(rs.randint(0, 2**31 - 1)))
            train_ds, val_ds = random_split(
                dataset, lengths=[n_train, n_val], generator=gen
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            train_ds = dataset
            val_loader = None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

        # class weights
        if isinstance(self.class_weight, str) and self.class_weight == "balanced":
            # inverse frequency
            counts = np.bincount(y_enc, minlength=self._n_classes_)
            inv = 1.0 / np.clip(counts, 1, None)
            w = inv / inv.sum() * self._n_classes_
            class_weights = torch.tensor(w, dtype=torch.float32, device=self._device_)
        elif isinstance(self.class_weight, np.ndarray):
            class_weights = torch.tensor(
                self.class_weight.astype(np.float32), device=self._device_
            )
        else:
            class_weights = None

        # build voters
        if callable(self.voter_builders):
            builders = self.voter_builders()
            # If a callable returns modules directly, accept them; otherwise call each builder
            if len(builders) > 0 and isinstance(builders[0], nn.Module):
                voters = builders  # type: ignore
            else:
                voters = [b() for b in builders]  # type: ignore
        else:
            voters = [b() for b in self.voter_builders]

        # init distributions
        if self.voter_init_distributions is None:
            inits = [InitDistribution()] * len(voters)
        elif isinstance(self.voter_init_distributions, InitDistribution):
            inits = [self.voter_init_distributions] * len(voters)
        else:
            if len(self.voter_init_distributions) != len(voters):
                raise ValueError(
                    "Length of voter_init_distributions must match number of voters."
                )
            inits = self.voter_init_distributions

        # move to device + initialize
        self._voters_: List[nn.Module] = []
        self._voter_num_params_: List[int] = []
        for v, initd in zip(voters, inits):
            v = v.to(self._device_)
            initd.apply(v)
            self._voters_.append(v)
            self._voter_num_params_.append(count_params(v))

        # training hyperparams
        self._hist_: Dict[str, Any] = {
            "val_acc": [],
            "best_epoch": [],
            "best_val_acc": [],
        }
        self._train_time_s_: float = 0.0

        ce = nn.CrossEntropyLoss(weight=class_weights)

        t0 = time.time()
        for idx, voter in enumerate(self._voters_):
            opt = torch.optim.Adam(
                voter.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            best_val = -math.inf
            best_state = None
            patience_left = self.early_stopping_patience

            for epoch in range(self.epochs):
                voter.train()
                for xb, yb in train_loader:
                    opt.zero_grad(set_to_none=True)
                    logits = voter(xb)  # (B, C)
                    loss = ce(logits, yb)
                    loss.backward()
                    opt.step()

                # validation
                if val_loader is not None:
                    voter.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            logits = voter(xb)
                            pred = logits.argmax(dim=-1)
                            correct += (pred == yb).sum().item()
                            total += yb.size(0)
                    val_acc = correct / max(1, total)
                    if self.verbose >= 2:
                        print(f"[Voter {idx:02d}] epoch {epoch} val_acc={val_acc:.4f}")
                    if val_acc > best_val + 1e-12:
                        best_val = val_acc
                        best_state = {
                            k: v.clone().detach() for k, v in voter.state_dict().items()
                        }
                        patience_left = self.early_stopping_patience
                    else:
                        patience_left -= 1
                        if patience_left <= 0:
                            break

            if best_val > -math.inf and best_state is not None:
                voter.load_state_dict(best_state)
            self._hist_["val_acc"].append(best_val if best_val > -math.inf else None)
            self._hist_["best_epoch"].append(
                self.epochs - patience_left if val_loader is not None else self.epochs
            )
            self._hist_["best_val_acc"].append(
                best_val if best_val > -math.inf else None
            )

        self._train_time_s_ = time.time() - t0
        return self

    def _transform_X(self, X: ArrayLike) -> torch.Tensor:
        check_is_fitted(self, attributes=["_voters_", "_le_", "_device_"])
        X_np = (
            np.asarray(X)
            if not isinstance(X, torch.Tensor)
            else X.detach().cpu().numpy()
        )
        if getattr(self, "_preprocess_", None) is not None:
            X_np = self._preprocess_.transform(X_np)
        return _to_tensor(X_np, self._device_)

    def classes_(self) -> np.ndarray:
        check_is_fitted(self, attributes=["_le_"])
        return self._le_.classes_

    def _forward_all_logits(
        self, X_t: torch.Tensor, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Returns stacked logits of shape (V, N, C)
        """
        V = len(self._voters_)
        N = X_t.shape[0]
        C = self._n_classes_
        batch_size = batch_size or self.batch_size
        out = []
        for v in self._voters_:
            v.eval()
            logits_parts = []
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    xb = X_t[i : i + batch_size]
                    logits_parts.append(v(xb))  # (b, C)
            out.append(torch.cat(logits_parts, dim=0))
        stack = torch.stack(out, dim=0)  # (V, N, C)
        if stack.shape[-1] != C:
            raise RuntimeError(
                f"Voter produced {stack.shape[-1]} classes but label encoder has {C}."
            )
        return stack

    def predict(self, X: ArrayLike) -> np.ndarray:
        X_t = self._transform_X(X)
        logits_stack = self._forward_all_logits(X_t)  # (V, N, C)
        if self.aggregate == "argmax":
            preds_per_voter = logits_stack.argmax(dim=-1)  # (V, N)
            preds = _plurality_vote(preds_per_voter)  # (N,)
        else:
            scores, _ = _aggregate_logits(logits_stack, self.aggregate)
            preds = scores.argmax(dim=-1)
        return self._le_.inverse_transform(_to_numpy(preds).astype(int))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        X_t = self._transform_X(X)
        logits_stack = self._forward_all_logits(X_t)
        _, probs = _aggregate_logits(logits_stack, self.aggregate)
        return _to_numpy(probs)

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        X_t = self._transform_X(X)
        logits_stack = self._forward_all_logits(X_t)
        scores, _ = _aggregate_logits(logits_stack, "mean_logits")
        return _to_numpy(scores)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y_true))

    # -------------
    # Analysis utilities
    # -------------
    @torch.no_grad()
    def voter_outputs(self, X: ArrayLike) -> Dict[str, np.ndarray]:
        """
        Returns per-voter outputs for analysis.
        - 'logits': shape (V, N, C)
        - 'probs' : shape (V, N, C)
        - 'preds' : shape (V, N)
        """
        X_t = self._transform_X(X)
        logits_stack = self._forward_all_logits(X_t)
        probs_stack = _softmax_logits(logits_stack)
        preds_stack = logits_stack.argmax(dim=-1)
        return {
            "logits": _to_numpy(logits_stack),
            "probs": _to_numpy(probs_stack),
            "preds": _to_numpy(preds_stack).astype(int),
        }

    @torch.no_grad()
    def metrics_on_dataset(self, X: ArrayLike, y: ArrayLike) -> Dict[str, float]:
        """
        Computes:
        - accuracy: ensemble accuracy
        - gibbs_risk: mean individual-voter error rate
        - disagreement: fraction of voter pairs that disagree on prediction
        """
        y_np = np.asarray(y)
        y_enc = self._le_.transform(y_np)
        X_t = self._transform_X(X)
        logits_stack = self._forward_all_logits(X_t)
        probs_stack = _softmax_logits(logits_stack)
        preds_stack = logits_stack.argmax(dim=-1)  # (V, N)

        # Ensemble prediction consistent with aggregate
        if self.aggregate == "argmax":
            ens_pred = _plurality_vote(preds_stack)  # (N,)
        else:
            scores, _ = _aggregate_logits(logits_stack, self.aggregate)
            ens_pred = scores.argmax(dim=-1)

        N = preds_stack.shape[1]
        V = preds_stack.shape[0]
        correct_ens = (ens_pred.cpu().numpy().astype(int) == y_enc).mean()

        # Gibbs risk: average error probability of a randomly sampled voter
        # empirical: mean over voters of their 0/1 error on (X,y)
        voter_correct = (preds_stack.cpu().numpy().astype(int) == y_enc[None, :]).mean(
            axis=1
        )  # (V,)
        gibbs_risk = float(1.0 - voter_correct.mean())

        # Pairwise disagreement: for uniformly sampled pair of voters (i != j), fraction of x with different predictions
        # Efficient vectorization: compute disagreements for all pairs by counting collisions per sample
        # Disagreement per sample = 1 - sum_c (count_c choose 2) / (V choose 2)
        preds_np = preds_stack.cpu().numpy()  # (V, N)
        disagreement_per_sample = []
        from collections import Counter

        # Vectorized: per column counts
        for j in range(N):
            counts = np.bincount(preds_np[:, j], minlength=self._n_classes_)
            # total pairs
            tot_pairs = V * (V - 1) // 2
            agree_pairs = int(np.sum(counts * (counts - 1) // 2))
            disagreement_per_sample.append(1.0 - (agree_pairs / max(1, tot_pairs)))
        disagreement = float(np.mean(disagreement_per_sample))

        return {
            "accuracy": float(correct_ens),
            "gibbs_risk": gibbs_risk,
            "disagreement": disagreement,
        }

    # -------------
    # Introspection
    # -------------
    @property
    def voters_(self) -> List[nn.Module]:
        check_is_fitted(self, attributes=["_voters_"])
        return self._voters_

    @property
    def voter_num_params_(self) -> List[int]:
        check_is_fitted(self, attributes=["_voter_num_params_"])
        return self._voter_num_params_

    @property
    def train_time_s_(self) -> float:
        check_is_fitted(self, attributes=["_train_time_s_"])
        return float(self._train_time_s_)
