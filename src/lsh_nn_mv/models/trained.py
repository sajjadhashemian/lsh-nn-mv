"""Trained neural baselines."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.seed import set_all_seeds


class TrainedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int = 256,
        depth: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        set_all_seeds(seed)
        layers = []
        in_features = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def predict(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
    ) -> Dict[str, float]:
        device = torch.device(device)
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": []}
        for _ in tqdm(range(epochs), desc="TrainedMLP", leave=False):
            self.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.view(xb.size(0), -1)
                optim.zero_grad()
                loss = criterion(self.forward(xb), yb)
                loss.backward()
                optim.step()
                running_loss += loss.item() * xb.size(0)
            history["train_loss"].append(running_loss / len(train_loader.dataset))
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        xb = xb.view(xb.size(0), -1)
                        logits = self.forward(xb)
                        val_loss += criterion(logits, yb).item() * xb.size(0)
                history["val_loss"].append(val_loss / len(val_loader.dataset))
        return history
