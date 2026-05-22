import torch
from src.losses import MRLLoss


def _fake_logits(dims, batch=4, classes=10):
    return {m: torch.randn(batch, classes, requires_grad=True) for m in dims}


def test_returns_scalar_and_per_dim_breakdown():
    dims = [8, 16, 32]
    loss_fn = MRLLoss(nesting_dims=dims, dim_weights=None, label_smoothing=0.0)
    logits = _fake_logits(dims)
    targets = torch.randint(0, 10, (4,))
    total, parts = loss_fn(logits, targets)
    assert total.ndim == 0
    assert set(parts.keys()) == set(dims)
    assert all(isinstance(v, float) for v in parts.values())


def test_total_equals_weighted_sum_of_parts():
    dims = [8, 16]
    weights = [1.0, 3.0]
    loss_fn = MRLLoss(nesting_dims=dims, dim_weights=weights, label_smoothing=0.0)
    logits = _fake_logits(dims)
    targets = torch.randint(0, 10, (4,))
    total, parts = loss_fn(logits, targets)
    expected = weights[0] * parts[8] + weights[1] * parts[16]
    assert abs(total.item() - expected) < 1e-4


def test_loss_is_differentiable():
    dims = [8, 16]
    loss_fn = MRLLoss(nesting_dims=dims, dim_weights=None, label_smoothing=0.1)
    logits = _fake_logits(dims)
    targets = torch.randint(0, 10, (4,))
    total, _ = loss_fn(logits, targets)
    total.backward()
    assert logits[8].grad is not None


def test_missing_dim_in_logits_raises():
    import pytest
    loss_fn = MRLLoss(nesting_dims=[8, 16], dim_weights=None, label_smoothing=0.0)
    with pytest.raises(KeyError):
        loss_fn({8: torch.randn(4, 10)}, torch.randint(0, 10, (4,)))
