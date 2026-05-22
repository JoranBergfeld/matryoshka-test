import numpy as np
import torch

from src.eval import pca_truncated_accuracy, evaluate_mrl_per_dim


def test_pca_truncated_accuracy_returns_value_per_dim():
    rng = np.random.RandomState(0)
    feats = np.vstack([rng.randn(50, 2048) + 5, rng.randn(50, 2048) - 5]).astype("float32")
    labels = np.array([0] * 50 + [1] * 50)
    acc = pca_truncated_accuracy(feats, labels, feats, labels, dims=[2, 8])
    assert set(acc.keys()) == {2, 8}
    assert acc[8] >= 90.0


def test_evaluate_mrl_per_dim_shapes():
    class FakeModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.nesting = [4, 8]
        def forward(self, x):
            b = x.shape[0]
            return {4: torch.randn(b, 3), 8: torch.randn(b, 3)}
    loader = [(torch.randn(5, 3, 4, 4), torch.randint(0, 3, (5,))) for _ in range(2)]
    out = FakeModel()
    res = evaluate_mrl_per_dim(out, loader, dims=[4, 8], device="cpu", ks=(1,))
    assert set(res.keys()) == {4, 8}
    assert 1 in res[4]
