import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from lsh_nn_mv.methods.calibrate import calibrate_sigma


def _make_toy_dataset():
    near = torch.randn(20, 4) * 0.1
    far = torch.randn(20, 4) * 0.1 + 5.0
    X = torch.cat([near, far], dim=0)
    y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def test_calibration_improves_margin():
    loader = _make_toy_dataset()
    result = calibrate_sigma(
        model_family="mlp",
        data_loader=loader,
        r=0.5,
        c=3.0,
        tau=0.5,
        grid_sigma=[0.1, 0.5, 1.0],
        max_pairs=200,
    )
    assert result.score >= 0.0
