import pytest

torch = pytest.importorskip("torch")

from lsh_nn_mv.methods.adversarial import fgsm, pgd


def simple_model(x: torch.Tensor) -> torch.Tensor:
    logits = torch.stack([x[:, 0], -x[:, 0]], dim=1)
    return logits


def test_fgsm_changes_prediction():
    x = torch.tensor([[0.5, 0.0]], requires_grad=False)
    y = torch.tensor([0])
    clean_pred = simple_model(x).argmax(dim=-1)
    adv = fgsm(simple_model, x, y, eps=1.0)
    adv_pred = simple_model(adv).argmax(dim=-1)
    assert clean_pred.item() != adv_pred.item()


def test_pgd_changes_prediction():
    x = torch.tensor([[0.5, 0.0]], requires_grad=False)
    y = torch.tensor([0])
    adv = pgd(simple_model, x, y, eps=1.0, steps=3, step_size=0.5)
    adv_pred = simple_model(adv).argmax(dim=-1)
    assert adv_pred.item() == 1
