"""Majority vote ensemble over random voters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List

import torch
from torch import Tensor

from .voters import VoterFactory
from ..utils.seed import set_all_seeds


class LSHEnsemble(torch.nn.Module):
    def __init__(
        self,
        voter_ctor: Callable[..., torch.nn.Module] | VoterFactory,
        n_voters: int,
        voter_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_voters = n_voters
        self.voters: List[torch.nn.Module] = []
        set_all_seeds(seed)

        if isinstance(voter_ctor, VoterFactory):
            for i in range(n_voters):
                voter = voter_ctor.build(seed + i)
                self.voters.append(voter.to(self.device))
        else:
            voter_kwargs = voter_kwargs or {}
            for i in range(n_voters):
                kwargs = dict(voter_kwargs)
                kwargs.setdefault("seed", seed + i)
                voter = voter_ctor(**kwargs)
                self.voters.append(voter.to(self.device))

    def forward(self, x: Tensor) -> Tensor:
        votes = [voter.hash(x.to(self.device)) for voter in self.voters]
        stacked = torch.stack(votes, dim=0)
        return stacked

    def predict_proba(self, x: Tensor, num_classes: int) -> Tensor:
        votes = self.forward(x)
        one_hot = torch.nn.functional.one_hot(votes, num_classes=num_classes).float()
        probs = one_hot.mean(dim=0)
        return probs

    def predict(self, x: Tensor, num_classes: int) -> Tensor:
        probs = self.predict_proba(x, num_classes)
        return torch.argmax(probs, dim=-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "n_voters": self.n_voters,
            "state_dicts": [voter.state_dict() for voter in self.voters],
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        voter_ctor: Callable[..., torch.nn.Module],
        voter_kwargs: dict,
        device: str | torch.device = "cpu",
    ) -> "LSHEnsemble":
        data = torch.load(path, map_location=device)
        ensemble = cls(voter_ctor, data["n_voters"], voter_kwargs, device=device)
        for voter, state_dict in zip(ensemble.voters, data["state_dicts"]):
            voter.load_state_dict(state_dict)
        return ensemble

    def to(self, device: str | torch.device):  # type: ignore[override]
        self.device = torch.device(device)
        for voter in self.voters:
            voter.to(self.device)
        return self

    def eval(self):  # type: ignore[override]
        for voter in self.voters:
            voter.eval()
        return self

    def train(self, mode: bool = True):  # type: ignore[override]
        for voter in self.voters:
            voter.train(mode)
        return self

    def __iter__(self) -> Iterable[torch.nn.Module]:
        return iter(self.voters)
