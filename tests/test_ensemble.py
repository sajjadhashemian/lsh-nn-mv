import pytest

torch = pytest.importorskip("torch")

from lsh_nn_mv.models.ensemble import LSHEnsemble
from lsh_nn_mv.models.voters import MLPVoter


def test_majority_vote_matches_manual():
    def ctor(seed: int):
        return MLPVoter(input_dim=2, num_classes=2, width=4, depth=1, sigma=1.0, seed=seed)

    ensemble = LSHEnsemble(voter_ctor=ctor, n_voters=3, voter_kwargs={}, device="cpu", seed=0)
    x = torch.randn(5, 2)
    votes = ensemble.forward(x)
    majority = torch.mode(votes, dim=0).values
    preds = ensemble.predict(x, num_classes=2)
    assert torch.all(preds == majority)
