import torch
from src.models import MRLHead, BaselineResNet50, MRLResNet50, build_model


def test_mrl_head_returns_logits_per_dim():
    head = MRLHead(embedding_dim=2048, num_classes=1000, nesting_dims=[8, 16, 2048])
    z = torch.randn(2, 2048)
    out = head(z)
    assert set(out.keys()) == {8, 16, 2048}
    for m, logits in out.items():
        assert logits.shape == (2, 1000)


def test_mrl_head_uses_weight_prefix():
    head = MRLHead(embedding_dim=16, num_classes=5, nesting_dims=[4, 16])
    z = torch.randn(3, 16)
    out = head(z)
    W = head.classifier.weight  # (5, 16)
    b = head.classifier.bias    # (5,)
    expected_4 = z[:, :4] @ W[:, :4].T + b
    assert torch.allclose(out[4], expected_4, atol=1e-5)


def test_mrl_head_rejects_bad_nesting_dims():
    import pytest
    with pytest.raises(ValueError):
        MRLHead(embedding_dim=16, num_classes=5, nesting_dims=[4, 32])  # 32 > 16


def test_baseline_forward_and_embed_shapes():
    model = BaselineResNet50(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    assert model(x).shape == (2, 10)
    assert model.embed(x).shape == (2, 2048)


def test_baseline_forward_equals_fc_of_embed():
    model = BaselineResNet50(num_classes=10).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        feats = model.embed(x)
        manual = model.fc(feats)
    assert torch.allclose(logits, manual, atol=1e-5)


def test_mrl_resnet_forward_and_embed():
    model = MRLResNet50(num_classes=10, nesting_dims=[8, 16, 2048])
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert set(out.keys()) == {8, 16, 2048}
    assert out[8].shape == (2, 10)
    assert model.embed(x).shape == (2, 2048)


def test_build_model_factory():
    import pytest
    assert isinstance(build_model("baseline", num_classes=10), BaselineResNet50)
    assert isinstance(build_model("mrl", num_classes=10, nesting_dims=[8, 2048]), MRLResNet50)
    with pytest.raises(ValueError):
        build_model("nonsense", num_classes=10)
