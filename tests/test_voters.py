import pytest

torch = pytest.importorskip("torch")

from lsh_nn_mv.models.voters import MLPVoter


def test_hash_outputs_indices():
    voter = MLPVoter(input_dim=4, num_classes=3, width=8, depth=1, sigma=0.5, seed=0)
    x = torch.randn(5, 4)
    hashes = voter.hash(x)
    assert hashes.shape == (5,)
    assert hashes.dtype == torch.long
    assert torch.all((hashes >= 0) & (hashes < 3))
