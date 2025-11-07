"""Configuration dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DatasetConfig:
    name: str
    root: str | None = None
    batch_size: int = 128
    val_batch_size: int | None = None
    num_workers: int = 0


@dataclass
class ModelConfig:
    width: int = 512
    depth: int = 2
    sigma_grid: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5])


@dataclass
class EvalConfig:
    ensemble_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    attack_eps: List[float] = field(default_factory=lambda: [0.05, 0.1])


@dataclass
class ExperimentConfig:
    seed: int = 0
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "results"

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentConfig":
        dataset = DatasetConfig(**data.get("dataset", {}))
        model = ModelConfig(**data.get("model", {}))
        evaluation = EvalConfig(**data.get("evaluation", {}))
        return cls(
            seed=data.get("seed", 0),
            dataset=dataset,
            model=model,
            evaluation=evaluation,
            output_dir=data.get("output_dir", "results"),
        )
