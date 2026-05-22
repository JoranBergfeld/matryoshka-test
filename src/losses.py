"""Matryoshka representation learning loss."""
from __future__ import annotations

import torch
import torch.nn as nn


class MRLLoss(nn.Module):
    """Weighted sum of cross-entropy over all nesting dims:
        L = sum_m w_m * CE(logits_m, y)
    with label smoothing. forward returns (total_loss, {m: ce_value}) where the
    per-dim values are detached floats for logging."""

    def __init__(
        self,
        nesting_dims: list[int],
        dim_weights: list[float] | None = None,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.nesting_dims = list(nesting_dims)
        if dim_weights is None:
            dim_weights = [1.0] * len(self.nesting_dims)
        if len(dim_weights) != len(self.nesting_dims):
            raise ValueError("dim_weights must match nesting_dims length")
        self.dim_weights = dim_weights
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self, logits_by_dim: dict[int, torch.Tensor], targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, float]]:
        total = None
        parts: dict[int, float] = {}
        for m, w in zip(self.nesting_dims, self.dim_weights):
            ce_m = self.ce(logits_by_dim[m], targets)  # KeyError if dim missing
            parts[m] = ce_m.item()
            term = w * ce_m
            total = term if total is None else total + term
        return total, parts
