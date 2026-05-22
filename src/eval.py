"""Validation: standard top-k, MRL per-dim accuracy, and PCA-truncated baseline."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.metrics import accuracy_topk


@torch.no_grad()
def evaluate_mrl_per_dim(model, loader, dims: list[int], device: str,
                         ks: tuple[int, ...] = (1, 5)) -> dict[int, dict[int, float]]:
    """Top-k accuracy at each MRL nesting dim. Returns {dim: {k: pct}}."""
    model.eval()
    correct = {m: {k: 0 for k in ks} for m in dims}
    total = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        bs = y.size(0); total += bs
        for m in dims:
            acc = accuracy_topk(out[m], y, ks=ks)
            for k in ks:
                correct[m][k] += acc[k] * bs / 100.0
    return {m: {k: correct[m][k] * 100.0 / total for k in ks} for m in dims}


@torch.no_grad()
def extract_features(model, loader, device: str, max_samples: int | None = None):
    """Collect (features, labels) using model.embed(). Caps at max_samples."""
    model.eval()
    feats, labels = [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        f = model.embed(x).cpu().numpy()
        feats.append(f); labels.append(y.numpy())
        n += len(y)
        if max_samples is not None and n >= max_samples:
            break
    return np.concatenate(feats)[:max_samples], np.concatenate(labels)[:max_samples]


def pca_truncated_accuracy(train_feats, train_labels, val_feats, val_labels,
                           dims: list[int]) -> dict[int, float]:
    """For each dim m: PCA-project to m dims, fit a logistic-regression probe on
    train, return top-1 percent on val. This is the PCA-truncated baseline curve."""
    out: dict[int, float] = {}
    max_dim = min(max(dims), train_feats.shape[1], train_feats.shape[0])
    pca = PCA(n_components=max_dim, random_state=0).fit(train_feats)
    train_proj_full = pca.transform(train_feats)
    val_proj_full = pca.transform(val_feats)
    for m in dims:
        mm = min(m, max_dim)
        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        clf.fit(train_proj_full[:, :mm], train_labels)
        preds = clf.predict(val_proj_full[:, :mm])
        out[m] = float((preds == val_labels).mean() * 100.0)
    return out
