# Neural Voter Ensemble — Reproducible, sklean-style PyTorch Majority-Vote Classifier

A modular **scikit-learn estimator** that aggregates **user-supplied PyTorch voters** (MLP/CNN/LSTM or any `nn.Module`) into a majority-vote classifier, plus a companion **experiment runner** that evaluates accuracy, **Gibbs risk**, and **pairwise disagreement** on tabular, image, and sequence data. The design emphasizes **determinism**, **sklearn interoperability**, and **theoretically motivated diagnostics**.

---

## Contents

* [Motivation & Overview](#motivation--overview)
* [Algorithmic Framework](#algorithmic-framework)

  * [Voters, Predictions, and Aggregation](#voters-predictions-and-aggregation)
  * [Training Regime & Early Stopping](#training-regime--early-stopping)
  * [Determinism & Reproducibility](#determinism--reproducibility)
* [Theoretical Details](#theoretical-details)

  * [Gibbs Risk and Disagreement](#gibbs-risk-and-disagreement)
  * [Majority-Vote Intuition & Bounds](#majorityvote-intuition--bounds)
  * [Aggregation Semantics](#aggregation-semantics)
  * [Computational Complexity](#computational-complexity)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [API Reference](#api-reference)

  * [`NeuralVoterEnsemble`](#neuralvoterensemble)
  * [Utility: `metrics_on_dataset`](#utility-metrics_on_dataset)
* [Experiment Runner](#experiment-runner)

  * [Datasets](#datasets)
  * [Metrics & CSV Schema](#metrics--csv-schema)
  * [Plots & the Heuristic Bound](#plots--the-heuristic-bound)
* [Extending & Customizing](#extending--customizing)
* [Limitations & Notes](#limitations--notes)
* [Reproducibility Checklist](#reproducibility-checklist)
* [License](#license)

---

## Motivation & Overview

Ensembling weakly correlated classifiers often **amplifies margins** and reduces risk. This repository provides:

1. **A library-quality estimator** `NeuralVoterEnsemble` that looks and feels like `sklearn.neighbors.KNeighborsClassifier`, but whose base learners (“voters”) are **user-supplied PyTorch models**.
2. **A reproducible experiment driver** that runs controlled sweeps across architectures, ensemble sizes, and aggregation rules; logs tidy CSVs; and produces diagnostic plots that connect to theory (Gibbs risk, disagreement, and a majority-vote bound heuristic).

The estimator integrates into sklearn pipelines (fit/predict/predict_proba/score, `get_params`/`set_params`, label encoding, optional sklearn preprocessing), while training and inference are performed in PyTorch.

---

## Algorithmic Framework

### Voters, Predictions, and Aggregation

* We are given **voters** ( f_i : \mathcal{X} \to \mathbb{R}^K ) (PyTorch `nn.Module`s) that output **logits** over (K) classes.
* Per voter prediction:
  [
  \hat{y}*i(x) ;=; \arg\max*{k \in [K]} f_i(x)_k.
  ]
* The **ensemble** aggregates the (V) voters into a final score/probability and prediction. Supported **aggregators**:

  * `"argmax"`: **plurality vote** over per-voter argmax predictions.

    * `predict_proba`: mean of per-voter softmaxes (\frac{1}{V}\sum_i \mathrm{softmax}(f_i(x))).
  * `"mean_logits"`: average logits then argmax.

    * `predict_proba`: (\mathrm{softmax}!\big(\frac{1}{V}\sum_i f_i(x)\big)).
  * `"mean_probs"`: average probabilities then argmax.

    * `predict_proba`: (\frac{1}{V}\sum_i \mathrm{softmax}(f_i(x))).
  * `"ceil"` / `"ceil_sum"`: (ablation) elementwise `ceil(sum(logits))` for argmax; probs from softmax of the sum.
  * **Custom callable** `f(logits_stack: Tensor[V,N,C]) -> Tensor[N,C]` returning scores for argmax; probabilities are taken as `softmax(scores)`.

**Tie-breaking**: plurality ties are broken deterministically by the lowest class index (PyTorch’s `mode`), ensuring reproducibility.

### Training Regime & Early Stopping

Each voter is trained independently with cross-entropy on the same dataset split:

* Optimizer: Adam (configurable `lr`, `weight_decay`).
* Mini-batches with configurable `batch_size`.
* Optional validation split (`val_split`) enables early stopping (`early_stopping_patience`) on validation accuracy.
* Optional class weights: `"balanced"` or custom array.

### Determinism & Reproducibility

Setting `random_state` seeds **Python**, **NumPy**, and **PyTorch** (CPU & CUDA). We disable cuDNN auto-tuning to reduce nondeterminism. Data splits and loaders use fixed seeds; we allow `num_workers=0` for fully deterministic data iteration.

---

## Theoretical Details

### Gibbs Risk and Disagreement

Let ( V ) be the number of voters. On a dataset ( {(x_j,y_j)}_{j=1}^N ):

* **Gibbs risk** (empirical): the error of a **random** voter drawn uniformly from the ensemble,
  [
  \widehat{R}*G
  ;=;
  \frac{1}{V}\sum*{i=1}^V \frac{1}{N}\sum_{j=1}^N \mathbf{1}{\hat{y}_i(x_j) \neq y_j}.
  ]
  Intuitively, this is the expected error if we pick one voter at random at test time.

* **Pairwise disagreement** (empirical): the probability that two independently drawn voters disagree **on prediction** for the same input,
  [
  \widehat{D}
  ;=;
  \frac{1}{N}\sum_{j=1}^N
  \Pr_{i \neq i'}\big[\hat{y}*i(x_j) \neq \hat{y}*{i'}(x_j)\big].
  ]
  Empirically we compute, for each sample, the fraction of disagreeing pairs among the ( \binom{V}{2} ) unordered pairs and then average over samples.

These quantities characterize **strength** and **diversity**: lower ( \widehat{R}_G ) means stronger average voters; higher ( \widehat{D} ) means more diversity (but excessive disagreement with weak voters can harm the majority).

### Majority-Vote Intuition & Bounds

For **plurality/majority aggregation**, classic analyses (e.g., PAC-Bayes C-bounds) link the majority-vote risk ( R_{\mathrm{MV}} ) to (R_G) and (D). In particular, with suitable assumptions, a bound of the following *shape* often appears:

[
R_{\mathrm{MV}}
;\lesssim;
1
-

\frac{\big(1 - 2R_G\big)^2}{,1 - 2D,}.
]

Our experiment runner includes a **pluggable heuristic** that replaces the population terms with **empirical** estimates plus Hoeffding-style deviations (see below). This is not meant as a certified bound but a **diagnostic curve** that typically tracks the trend: **as voters get better (lower (R_G)) and remain sufficiently non-redundant (moderate (D)), the majority vote improves.**

Separately, when voters are **independent and locally stable**, a Hoeffding argument yields exponential tail decay for the event that more than half of the voters flip on a small perturbation—this explains **margin amplification** in ensembles.

### Aggregation Semantics

* `"argmax"` is **voting on discrete predictions**; probabilities are reported as the **mean of softmaxes** (Gibbs probabilities), which aligns with the “randomized classifier” interpretation.
* `"mean_logits"` vs. `"mean_probs"` differ subtly:

  * Averaging **logits** then softmax corresponds to a log-opinion-pool and preserves calibrated margins in many setups.
  * Averaging **probabilities** emphasizes consensus in probability space and is less sensitive to per-voter logit scale.
* For **evaluation**, we ensure `predict` is consistent with the chosen aggregator; `predict_proba` matches its semantics.

### Computational Complexity

Let (V) be voters, (N) samples, and (T_{\text{fwd}}) the average forward-pass time per sample per voter.

* **Training**: (V) independent trainings; cost scales roughly linearly in (V). Early stopping can reduce epochs heterogeneously.
* **Inference**: one forward pass per voter; cost (O(V \cdot N \cdot T_{\text{fwd}})). Aggregation itself is (O(VNC)).

---

## Installation

```bash
# Python >= 3.9 recommended
pip install -r requirements.txt
# or minimal:
pip install torch torchvision scikit-learn matplotlib pandas
```

> If you plan to run the image experiments (MNIST/FashionMNIST), ensure `torchvision` is installed.

---

## Quick Start

```python
from neural_voter_ensemble import NeuralVoterEnsemble, InitDistribution
from torch import nn
import numpy as np

# 1) Build voters (heterogeneous allowed)
class TinyMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))
    def forward(self, x): 
        if x.dim() > 2: x = x.view(x.size(0), -1)
        return self.net(x)

in_dim, out_dim = 20, 3
builders = [lambda: TinyMLP(in_dim, out_dim) for _ in range(5)]

# 2) Create estimator
est = NeuralVoterEnsemble(
    voter_builders=builders,
    voter_init_distributions=InitDistribution(scheme="kaiming_normal", nonlinearity="relu"),
    aggregate="mean_logits",
    random_state=0,
    epochs=5,
    batch_size=128,
)

# 3) Fit / predict (sklearn-style)
X = np.random.randn(1000, in_dim).astype("float32")
y = np.random.randint(0, out_dim, size=(1000,))
est.fit(X, y)
yhat = est.predict(X)
proba = est.predict_proba(X)
print("Accuracy:", est.score(X, y))
print("Per-voter metrics:", est.metrics_on_dataset(X, y))
```

---

## API Reference

### `NeuralVoterEnsemble`

**Constructor (key arguments)**

* `voter_builders`: `List[Callable[[], nn.Module]]` **or** a callable returning a list; each builds one voter. Heterogeneous voters allowed.
* `voter_init_distributions`: `InitDistribution` or `List[InitDistribution]`. Applies custom or standard (Kaiming/Xavier) initialization.
* `aggregate`: `"argmax" | "mean_logits" | "mean_probs" | "ceil" | "ceil_sum" | Callable`.
* `device`: `"auto" | "cpu" | "cuda"`.
* `random_state`: seed for full determinism.
* Training controls: `epochs`, `batch_size`, `lr`, `weight_decay`, `val_split`, `early_stopping_patience`, `class_weight`, `num_workers`.
* `sklearn_transformer`: optional sklearn transformer (e.g., `StandardScaler`) applied to `X` inside the estimator for tabular use.
* `verbose`: integer verbosity.

**Methods**

* `fit(X, y) -> self`
  Accepts NumPy/Torch arrays; learns label encoder, optional transformer; trains each voter independently with early stopping.
* `predict(X) -> np.ndarray[(n_samples,)]`
  Ensemble prediction consistent with `aggregate`.
* `predict_proba(X) -> np.ndarray[(n_samples, n_classes)]`
  Probabilities defined per aggregation semantics.
* `decision_function(X) -> np.ndarray[(n_samples, n_classes)]`
  Mean-logit scores (useful for calibration/ROC).
* `score(X, y) -> float`
  Accuracy.
* `voter_outputs(X) -> dict`
  Returns per-voter `logits`, `probs`, and `preds` for analysis.
* `metrics_on_dataset(X, y) -> dict`
  Returns `{accuracy, gibbs_risk, disagreement}`.

**Attributes (post-fit)**

* `voters_`: list of trained `nn.Module`s.
* `voter_num_params_`: list of parameter counts per voter.
* `train_time_s_`: total training wall-clock time (seconds).
* `classes_()`: returns label encoder’s class list.

### Utility: `metrics_on_dataset`

* **accuracy**: ensemble accuracy under chosen aggregation.
* **gibbs_risk**: mean individual-voter error.
* **disagreement**: mean fraction of disagreeing voter pairs per example.

All computations are vectorized and GPU-friendly.

---

## Experiment Runner

**Script**: `run_experiments.py`
Produces a consolidated CSV and two plots per sweep.

### Datasets

* **Tabular (MLP)**: `sklearn.datasets.make_classification` (configurable size).
* **Image (CNN)**: `MNIST` or `FashionMNIST` (via `torchvision`; sub-sampled for fast runs).
* **Sequence (LSTM)**: synthetic sequence classification (label = argmax cumulative sum dimension).

### Usage

```bash
# Tabular sweep, medium MLP, three seeds, 1/3/5/9 voters, three aggregators
python run_experiments.py \
  --dataset tabular \
  --arch_size m \
  --voters 1 3 5 9 \
  --aggregates mean_logits mean_probs argmax \
  --seeds 0 1 2 \
  --out_dir ./results

# MNIST quick pass with small CNN voters
python run_experiments.py --dataset mnist --arch_size s --voters 1 3 5 --seeds 0 --out_dir ./results

# Sequence synthetic with LSTM voters
python run_experiments.py --dataset sequence --arch_size m --voters 1 3 5 9 --seeds 0 1 --out_dir ./results
```

### Metrics & CSV Schema

One row per configuration:

| column         | description                                           |
| -------------- | ----------------------------------------------------- |
| `dataset`      | `tabular` / `mnist` / `fashion` / `sequence`          |
| `arch_family`  | `MLP` / `CNN` / `LSTM`                                |
| `arch_size`    | small/medium/large: `s`/`m`/`l` (affects depth/width) |
| `num_params`   | total trainable parameters across voters              |
| `num_voters`   | ensemble size (V)                                     |
| `init_scheme`  | initialization summary (e.g., `kaiming_normal`)       |
| `aggregate`    | aggregator name                                       |
| `seed`         | random seed                                           |
| `train_time_s` | wall-clock training time (seconds)                    |
| `accuracy`     | ensemble accuracy on test set                         |
| `knn_accuracy` | baseline k-NN accuracy (reasonable default grid)      |
| `gibbs_risk`   | empirical Gibbs risk on test set                      |
| `disagreement` | empirical pairwise disagreement on test set           |

### Plots & the Heuristic Bound

The script writes:

1. **Disagreement vs Gibbs risk** (`plots/disagreement_vs_gibbs.png`):
   Visualizes the strength/diversity trade-off; typically, you want **low Gibbs risk** with **non-trivial disagreement**.

2. **Accuracy vs #voters** (`plots/accuracy_vs_voters.png`):
   Overlays a **heuristic majority-vote bound** curve:

   ```python
   def heuristic_bound(num_voters, sample_size, gibbs_risk, disagreement, delta=0.05):
       """
       R_MV  <=  1 - (1 - 2 * R_G_hat_plus)^2 / (1 - 2 * D_hat_plus + eps)
       where R_G_hat_plus = R_G + sqrt(ln(2/delta)/(2n)), D_hat_plus = D + sqrt(ln(2/delta)/(2n)).
       Returns the bound on R_MV; the plot shows 1 - bound as an upper curve on accuracy.
       """
   ```

   This **diagnostic** (not a certified guarantee) helps relate empirical ( \widehat{R}_G ) and ( \widehat{D} ) to the observed ensemble accuracy.

---

## Extending & Customizing

* **Heterogeneous voters**: mix MLP/CNN/LSTM (any `nn.Module`) by supplying a list of builders.
* **Custom initialization**: pass a per-voter `InitDistribution` with `custom_init(module)` to reproduce specific parametrizations.
* **Custom aggregation**: provide a callable `f: Tensor[V,N,C] -> Tensor[N,C]` to implement, e.g., temperature-scaled logit averaging, weighted votes, or class-dependent weights.
* **Preprocessing**: attach any sklearn transformer (e.g., `StandardScaler`, `PCA`) via `sklearn_transformer` in the estimator; it will be fit and applied internally.
* **New datasets**: add a loader in the experiment script and plug a new builder for voters.
* **Alternative bounds**: replace `heuristic_bound` with your preferred closed-form or data-dependent bound.

---

## Limitations & Notes

* **Independence**: theory often assumes weak dependence or independence across voters; sharing parameters or data augmentations can reduce diversity.
* **Calibration**: `"mean_logits"` vs `"mean_probs"` can behave differently when voters are miscalibrated; consider temperature scaling if needed.
* **Non-determinism on GPU**: we disable some nondeterministic cuDNN paths, but certain layers/ops may still be nondeterministic on specific hardware/driver stacks.
* **Validation size**: with very small datasets, set `val_split=0` to avoid overly small validation sets; or increase data.

---

## Reproducibility Checklist

* Fix `random_state` in the estimator and **also** pass `--seeds ...` to the runner.
* Keep `num_workers=0` for deterministic data loading.
* Log environment: the runner prints Python/torch/sklearn versions and CUDA availability.
* All outputs (CSV + plots) are written under `--out_dir` with stable file names.
