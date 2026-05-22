"""ResNet50 backbone with two interchangeable heads: stock linear (baseline) and
Matryoshka-Efficient nested head (MRL). Backbone identical across arms."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class MRLHead(nn.Module):
    """MRL-E head: one shared Linear(embedding_dim, num_classes). For nesting dim
    m, logits_m = z[:, :m] @ W[:, :m].T + b, with W and b shared across all m."""

    def __init__(self, embedding_dim: int, num_classes: int, nesting_dims: list[int]) -> None:
        super().__init__()
        dims = sorted(nesting_dims)
        if any(d <= 0 for d in dims):
            raise ValueError("nesting dims must be positive")
        if max(dims) > embedding_dim:
            raise ValueError(f"largest nesting dim {max(dims)} exceeds embedding_dim {embedding_dim}")
        self.nesting_dims = dims
        self.embedding_dim = embedding_dim
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        W = self.classifier.weight  # (num_classes, embedding_dim)
        b = self.classifier.bias
        return {m: z[:, :m] @ W[:, :m].T + b for m in self.nesting_dims}


def _resnet50_backbone(num_classes: int) -> nn.Module:
    return torchvision.models.resnet50(weights=None, num_classes=num_classes)


class BaselineResNet50(nn.Module):
    """Stock resnet50(weights=None). embed() returns the pre-fc 2048-d feature via
    an Identity-swap: model.fc is held aside and applied explicitly in forward()."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        net = _resnet50_backbone(num_classes)
        self.fc = net.fc                 # real classifier, applied manually
        net.fc = nn.Identity()           # backbone now outputs the 2048-d feature
        self.backbone = net

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)          # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))


class MRLResNet50(nn.Module):
    """Same backbone as baseline; fc replaced by MRLHead."""

    def __init__(self, num_classes: int = 1000, nesting_dims: list[int] | None = None) -> None:
        super().__init__()
        if nesting_dims is None:
            nesting_dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        net = _resnet50_backbone(num_classes)
        net.fc = nn.Identity()
        self.backbone = net
        self.head = MRLHead(embedding_dim=2048, num_classes=num_classes, nesting_dims=nesting_dims)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        return self.head(self.embed(x))


def build_model(model_kind: str, num_classes: int, nesting_dims: list[int] | None = None) -> nn.Module:
    if model_kind == "baseline":
        return BaselineResNet50(num_classes=num_classes)
    if model_kind == "mrl":
        return MRLResNet50(num_classes=num_classes, nesting_dims=nesting_dims)
    raise ValueError(f"unknown model_kind: {model_kind!r}")
