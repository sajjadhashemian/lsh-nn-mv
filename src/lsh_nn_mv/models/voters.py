"""Randomly initialised neural network voters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import nn, Tensor

from ..utils.seed import set_all_seeds


class _RandomInitMixin:
    def _apply_random_init(self, sigma: float) -> None:
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    module.weight.normal_(mean=0.0, std=sigma)
                    if module.bias is not None:
                        module.bias.zero_()


class MLPVoter(nn.Module, _RandomInitMixin):
    """Random MLP voter."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int = 512,
        depth: int = 2,
        sigma: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            set_all_seeds(seed)

        layers = []
        in_features = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)
        self._apply_random_init(sigma)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.view(x.size(0), -1))

    def hash(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


class ConvVoter(nn.Module, _RandomInitMixin):
    """Random convolutional voter for image data."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        sigma: float = 1.0,
        seed: int | None = None,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        if seed is not None:
            set_all_seeds(seed)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        flattened = None
        # Determine flatten dimension using dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            flattened = self.net(dummy).shape[1]
        self.head = nn.Linear(flattened, num_classes)
        self._apply_random_init(sigma)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.net(x)
        return self.head(feats)

    def hash(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


@dataclass
class VoterFactory:
    ctor: Callable[..., nn.Module]
    args: Tuple
    kwargs: dict

    def build(self, seed: int | None = None) -> nn.Module:
        if seed is not None:
            kwargs = dict(self.kwargs)
            kwargs["seed"] = seed
        else:
            kwargs = self.kwargs
        return self.ctor(*self.args, **kwargs)
