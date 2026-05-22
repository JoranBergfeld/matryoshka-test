"""Metric helpers (top-k accuracy, gradient norm) and a jsonl appender."""
from __future__ import annotations

import json
from typing import Iterable

import torch


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor,
                  ks: tuple[int, ...] = (1, 5)) -> dict[int, float]:
    """Top-k accuracy in percent for each k. logits (B, C), targets (B,)."""
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
    correct = pred.eq(targets.view(-1, 1))
    out: dict[int, float] = {}
    batch = targets.size(0)
    for k in ks:
        kk = min(k, maxk)
        out[k] = correct[:, :kk].any(dim=1).float().sum().item() * 100.0 / batch
    return out


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm over all parameter gradients (0.0 if none have grads)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


class JsonlWriter:
    """Append one JSON object per line. Flushes each write so an interrupted run
    keeps every completed epoch's record."""

    def __init__(self, path: str) -> None:
        self.path = path

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
