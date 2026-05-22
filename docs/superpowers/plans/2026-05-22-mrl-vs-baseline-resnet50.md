# MRL vs Baseline ResNet50 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, reproducible image-classification experiment comparing a baseline ResNet50 against a Matryoshka-Efficient (MRL-E) ResNet50 on ImageNet-1K, with per-epoch metrics/plots, interruption-resilient checkpointing, and a post-hoc analysis report.

**Architecture:** A single explicit training loop (no Lightning/Accelerate) drives both arms from one shared config. Models differ only in the final head: baseline uses ResNet50's stock `fc`; MRL replaces it with a shared-weight nested head producing logits at every nesting dim. Pure units (loss, head, checkpoint, metrics, eval math) are unit-tested on CPU with tiny tensors; the loop and data layer get CPU smoke tests with synthetic data.

**Tech Stack:** Python, `uv`, PyTorch ≥ 2.1, torchvision, scikit-learn (PCA + linear probe), matplotlib, pyyaml, tqdm, pytest. Optional `data` extras: kaggle, huggingface_hub.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Project metadata, deps, `data` extra. uv-managed env. |
| `.gitignore` | Ignore `runs/`, dataset dir, `.venv/`, caches. |
| `configs/default.yaml` | All hyperparameters; CLI overrides. |
| `src/config.py` | Load YAML, apply CLI overrides, resolve `null` sentinels. |
| `src/losses.py` | `MRLLoss`. |
| `src/models.py` | `MRLHead`, `BaselineResNet50`, `MRLResNet50`, `build_model`. |
| `src/checkpoint.py` | Atomic save/load, RNG capture/restore, git hash. |
| `src/metrics.py` | `MetricTracker`, jsonl writer, topk/grad-norm helpers. |
| `src/data.py` | Transforms, ImageFolder loaders, deterministic `--quick` subset. |
| `src/eval.py` | Validation, MRL per-dim accuracy, PCA-truncation baseline curve. |
| `src/viz.py` | Per-epoch matplotlib plots. |
| `src/train.py` | Training loop, AMP, warmup+cosine, SIGINT, orchestration. |
| `scripts/train.py` | CLI entry point. |
| `scripts/init_data.py` | Detect creds, download/skip/validate dataset. |
| `scripts/analyze.py` | Post-hoc cross-run report + plots. |
| `README.md` | Setup, run instructions, expected timings, outputs. |
| `tests/` | Mirror of `src/` for unit + smoke tests. |

Dependency order of tasks: scaffold → config → losses → models → checkpoint → metrics → data → eval → viz → train → entry point → init_data → analyze → README.

---

## Task 0: Project scaffolding & uv environment

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/__init__.py`, `tests/__init__.py`, `configs/default.yaml`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "matryoshka-test"
version = "0.1.0"
description = "MRL vs baseline ResNet50 comparison on ImageNet-1K"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "torchvision",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "tqdm",
    "pyyaml",
]

[project.optional-dependencies]
data = ["kaggle", "huggingface_hub"]

[dependency-groups]
dev = ["pytest"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
runs/
data/
*.pt
*.pt.tmp
```

- [ ] **Step 3: Create empty package markers**

`src/__init__.py` and `tests/__init__.py` — both empty files.

- [ ] **Step 4: Create `configs/default.yaml`** (full content)

```yaml
# All hyperparameters live here. CLI args override any value.
run:
  seed: 42
  output_root: runs
  log_level: INFO

data:
  imagenet_path_env: IMAGENET_PATH
  num_classes: 1000
  quick:
    enabled: true          # default to the 100-class subset; --full disables
    num_classes: 100
  image_size: 224
  resize_size: 256
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]

dataloader:
  batch_size: 256
  effective_batch_size: 256
  num_workers: null
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

model:
  arch: resnet50
  embedding_dim: 2048
  mrl_nesting_dims: [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
  mrl_dim_weights: null
  mrl_weighting: uniform

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: false
  label_smoothing: 0.1

schedule:
  epochs: 90
  warmup_epochs: 5
  type: cosine

amp:
  enabled: true
  compile: true

eval:
  topk: [1, 5]
  pca:
    fit_every_epochs: 10
    fit_sample_size: 50000
  latency:
    warmup_batches: 100
    measure_samples: 1000

checkpoint:
  save_last: true
  save_best: true
  best_metric: val_top1
```

- [ ] **Step 5: Create the env and verify**

Run: `uv sync` then `uv run python -c "import torch, torchvision, sklearn, yaml, matplotlib; print(torch.__version__)"`
Expected: prints a torch version ≥ 2.1, no import errors. (`.venv/` created.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/__init__.py tests/__init__.py configs/default.yaml
git commit -m "chore: scaffold uv project, deps, and default config"
```

---

## Task 1: Config loader (`src/config.py`)

**Files:**
- Create: `src/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
from src.config import load_config, resolve_dim_weights


def test_load_default_returns_nested_dict():
    cfg = load_config("configs/default.yaml")
    assert cfg["run"]["seed"] == 42
    assert cfg["model"]["mrl_nesting_dims"][0] == 8


def test_cli_overrides_take_precedence():
    cfg = load_config("configs/default.yaml", overrides={"schedule.epochs": 3,
                                                          "run.seed": 7})
    assert cfg["schedule"]["epochs"] == 3
    assert cfg["run"]["seed"] == 7


def test_uniform_weighting_gives_equal_weights():
    dims = [8, 16, 32]
    assert resolve_dim_weights(dims, weighting="uniform", explicit=None) == [1.0, 1.0, 1.0]


def test_increasing_weighting_proportional_to_dim():
    dims = [8, 16, 32]
    w = resolve_dim_weights(dims, weighting="increasing", explicit=None)
    # proportional to dim, normalized so smallest == 1.0
    assert w == [1.0, 2.0, 4.0]


def test_explicit_weights_override_scheme():
    dims = [8, 16, 32]
    assert resolve_dim_weights(dims, weighting="increasing", explicit=[1.0, 1.0, 5.0]) == [1.0, 1.0, 5.0]


def test_explicit_weights_wrong_length_raises():
    import pytest
    with pytest.raises(ValueError):
        resolve_dim_weights([8, 16], weighting="uniform", explicit=[1.0])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`.

- [ ] **Step 3: Implement `src/config.py`**

```python
"""Config loading: YAML defaults with dotted-key CLI overrides."""
from __future__ import annotations

import copy
from typing import Any

import yaml


def load_config(path: str, overrides: dict[str, Any] | None = None) -> dict:
    """Load YAML config and apply dotted-key overrides (e.g. 'schedule.epochs').

    Overrides whose value is None are ignored (lets argparse defaults pass through
    without clobbering the YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            _set_dotted(cfg, dotted, value)
    return cfg


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def resolve_dim_weights(
    nesting_dims: list[int],
    weighting: str,
    explicit: list[float] | None,
) -> list[float]:
    """Turn a weighting scheme into concrete per-dim weights.

    explicit, if given, wins and must match len(nesting_dims).
    'uniform' -> all 1.0. 'increasing' -> proportional to dim, normalized so the
    smallest dim has weight 1.0."""
    if explicit is not None:
        if len(explicit) != len(nesting_dims):
            raise ValueError(
                f"dim_weights length {len(explicit)} != nesting_dims length {len(nesting_dims)}"
            )
        return [float(w) for w in explicit]
    if weighting == "uniform":
        return [1.0] * len(nesting_dims)
    if weighting == "increasing":
        smallest = min(nesting_dims)
        return [d / smallest for d in nesting_dims]
    raise ValueError(f"unknown weighting scheme: {weighting!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: config loader with CLI overrides and dim-weight resolution"
```

---

## Task 2: MRL loss (`src/losses.py`)

**Files:**
- Create: `src/losses.py`, `tests/test_losses.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_losses.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.losses'`.

- [ ] **Step 3: Implement `src/losses.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_losses.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: MRL loss with per-dim breakdown"
```

---

## Task 3: Models (`src/models.py`)

**Files:**
- Create: `src/models.py`, `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
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
    # logits at dim m must equal z[:, :m] @ W[:, :m].T + b
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models'`.

- [ ] **Step 3: Implement `src/models.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (7 passed). (Backbone runs on CPU with tiny batch — slow but fine.)

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: baseline and MRL ResNet50 models with shared backbone"
```

---

## Task 4: Checkpointing (`src/checkpoint.py`)

**Files:**
- Create: `src/checkpoint.py`, `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint.py
import os
import random

import numpy as np
import torch

from src.checkpoint import capture_rng_state, restore_rng_state, atomic_save, load_checkpoint


def test_atomic_save_creates_file_and_no_tmp(tmp_path):
    path = tmp_path / "last.pt"
    atomic_save({"epoch": 3}, str(path))
    assert path.exists()
    assert not (tmp_path / "last.pt.tmp").exists()


def test_load_roundtrip(tmp_path):
    path = tmp_path / "last.pt"
    atomic_save({"epoch": 5, "best_metric": 0.42}, str(path))
    ckpt = load_checkpoint(str(path), map_location="cpu")
    assert ckpt["epoch"] == 5
    assert ckpt["best_metric"] == 0.42


def test_rng_state_roundtrip_makes_sampling_reproducible():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    state = capture_rng_state()
    first = (random.random(), float(np.random.rand()), torch.rand(1).item())
    # advance the generators
    random.random(); np.random.rand(); torch.rand(1)
    restore_rng_state(state)
    second = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert first == second
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.checkpoint'`.

- [ ] **Step 3: Implement `src/checkpoint.py`**

```python
"""Atomic checkpoint save/load and RNG state capture/restore."""
from __future__ import annotations

import os
import random
import subprocess
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Snapshot all RNG states needed to resume bit-identically (within FP)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def git_commit_hash() -> str | None:
    """Return current git commit hash, or None if unavailable."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def atomic_save(obj: dict, path: str) -> None:
    """Write to <path>.tmp then os.replace() onto <path> — never a partial file."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: atomic checkpointing and RNG state capture/restore"
```

---

## Task 5: Metrics tracking (`src/metrics.py`)

**Files:**
- Create: `src/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import json

import torch

from src.metrics import accuracy_topk, grad_norm, JsonlWriter


def test_accuracy_topk_perfect():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    acc = accuracy_topk(logits, targets, ks=(1,))
    assert acc[1] == 100.0


def test_accuracy_top5_counts_within_top_k():
    logits = torch.tensor([[5., 4., 3., 2., 1., 0.]])  # true class rank 5 -> in top5
    targets = torch.tensor([4])
    acc = accuracy_topk(logits, targets, ks=(1, 5))
    assert acc[1] == 0.0
    assert acc[5] == 100.0


def test_grad_norm_zero_for_no_grads():
    p = torch.nn.Parameter(torch.zeros(3))
    assert grad_norm([p]) == 0.0


def test_grad_norm_matches_manual():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.0, 4.0])
    assert abs(grad_norm([p]) - 5.0) < 1e-5


def test_jsonl_writer_appends_one_line_per_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    w = JsonlWriter(str(path))
    w.append({"epoch": 0, "val_top1": 1.0})
    w.append({"epoch": 1, "val_top1": 2.0})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["epoch"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.metrics'`.

- [ ] **Step 3: Implement `src/metrics.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: top-k accuracy, grad-norm, and jsonl metric writer"
```

---

## Task 6: Data layer (`src/data.py`)

**Files:**
- Create: `src/data.py`, `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
from src.data import build_transforms, select_quick_classes, resolve_imagenet_path


def test_train_and_val_transforms_differ():
    train_t = build_transforms(train=True, image_size=224, resize_size=256,
                               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_t = build_transforms(train=False, image_size=224, resize_size=256,
                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_names = [type(t).__name__ for t in train_t.transforms]
    val_names = [type(t).__name__ for t in val_t.transforms]
    assert "RandomResizedCrop" in train_names
    assert "RandomHorizontalFlip" in train_names
    assert "CenterCrop" in val_names
    assert "RandomHorizontalFlip" not in val_names


def test_quick_class_selection_deterministic():
    a = select_quick_classes(all_classes=list(range(1000)), num_classes=100, seed=42)
    b = select_quick_classes(all_classes=list(range(1000)), num_classes=100, seed=42)
    c = select_quick_classes(all_classes=list(range(1000)), num_classes=100, seed=7)
    assert a == b
    assert a != c
    assert len(a) == 100


def test_resolve_imagenet_path_missing_env_raises(monkeypatch):
    import pytest
    monkeypatch.delenv("IMAGENET_PATH", raising=False)
    with pytest.raises(RuntimeError) as e:
        resolve_imagenet_path("IMAGENET_PATH")
    assert "IMAGENET_PATH" in str(e.value)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Implement `src/data.py`**

```python
"""ImageNet data: transforms, path resolution, deterministic quick subset, loaders."""
from __future__ import annotations

import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

_DOWNLOAD_HINT = (
    "Set IMAGENET_PATH to a directory with train/ and val/ ImageFolder layout.\n"
    "Get ImageNet-1K from Kaggle 'imagenet-object-localization-challenge' or "
    "Hugging Face 'timm/imagenet-1k-wds'. See scripts/init_data.py."
)


def resolve_imagenet_path(env_var: str) -> str:
    path = os.environ.get(env_var)
    if not path:
        raise RuntimeError(f"{env_var} is not set.\n{_DOWNLOAD_HINT}")
    if not (os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "val"))):
        raise RuntimeError(f"{env_var}={path} missing train/ or val/.\n{_DOWNLOAD_HINT}")
    return path


def build_transforms(train: bool, image_size: int, resize_size: int,
                     mean: list[float], std: list[float]) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def select_quick_classes(all_classes: list, num_classes: int, seed: int) -> list:
    """Deterministically pick a subset of class identifiers (sorted output)."""
    rng = random.Random(seed)
    chosen = rng.sample(list(all_classes), num_classes)
    return sorted(chosen)


def _subset_to_classes(dataset: ImageFolder, keep_class_names: set[str]) -> Subset:
    keep_idx = {dataset.class_to_idx[c] for c in keep_class_names}
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in keep_idx]
    return Subset(dataset, indices)


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, int]:
    """Build train/val loaders. Returns (train_loader, val_loader, num_classes).

    Honors cfg['data']['quick'] for the deterministic 100-class subset."""
    data_cfg = cfg["data"]
    root = resolve_imagenet_path(data_cfg["imagenet_path_env"])
    mean, std = data_cfg["mean"], data_cfg["std"]
    train_t = build_transforms(True, data_cfg["image_size"], data_cfg["resize_size"], mean, std)
    val_t = build_transforms(False, data_cfg["image_size"], data_cfg["resize_size"], mean, std)

    train_ds = ImageFolder(os.path.join(root, "train"), transform=train_t)
    val_ds = ImageFolder(os.path.join(root, "val"), transform=val_t)
    num_classes = data_cfg["num_classes"]

    if data_cfg["quick"]["enabled"]:
        keep = set(select_quick_classes(train_ds.classes,
                                        data_cfg["quick"]["num_classes"],
                                        cfg["run"]["seed"]))
        train_ds = _subset_to_classes(train_ds, keep)
        val_ds = _subset_to_classes(val_ds, keep)
        num_classes = data_cfg["quick"]["num_classes"]

    dl = cfg["dataloader"]
    num_workers = dl["num_workers"] if dl["num_workers"] is not None else min(8, os.cpu_count() or 1)
    common = dict(
        batch_size=dl["batch_size"],
        num_workers=num_workers,
        pin_memory=dl["pin_memory"],
        persistent_workers=dl["persistent_workers"] and num_workers > 0,
        prefetch_factor=dl["prefetch_factor"] if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, num_classes
```

> **Note on quick-subset labels:** `_subset_to_classes` keeps original ImageFolder label indices (0–999). For training with `num_classes=100` we need labels remapped to 0–99. Implement this remap inside `_subset_to_classes` by wrapping the Subset in a small label-remapping dataset. Add it now:

```python
class _RemappedSubset(torch.utils.data.Dataset):
    """Subset that remaps a sparse set of original labels to contiguous 0..K-1."""

    def __init__(self, base: ImageFolder, indices: list[int], label_map: dict[int, int]) -> None:
        self.base = base
        self.indices = indices
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]
        return x, self.label_map[y]
```

Replace `_subset_to_classes` body with:

```python
def _subset_to_classes(dataset: ImageFolder, keep_class_names: set[str]):
    keep_idx = sorted(dataset.class_to_idx[c] for c in keep_class_names)
    label_map = {orig: new for new, orig in enumerate(keep_idx)}
    keep_set = set(keep_idx)
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in keep_set]
    return _RemappedSubset(dataset, indices, label_map)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (3 passed). (These tests don't touch real data.)

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: data transforms, quick-subset, and loaders"
```

---

## Task 7: Evaluation (`src/eval.py`)

**Files:**
- Create: `src/eval.py`, `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval.py
import numpy as np
import torch

from src.eval import pca_truncated_accuracy, evaluate_mrl_per_dim


def test_pca_truncated_accuracy_returns_value_per_dim():
    rng = np.random.RandomState(0)
    # two well-separated clusters -> low dims should classify reasonably
    feats = np.vstack([rng.randn(50, 2048) + 5, rng.randn(50, 2048) - 5]).astype("float32")
    labels = np.array([0] * 50 + [1] * 50)
    acc = pca_truncated_accuracy(feats, labels, feats, labels, dims=[2, 8])
    assert set(acc.keys()) == {2, 8}
    assert acc[8] >= 90.0  # separable data


def test_evaluate_mrl_per_dim_shapes():
    # fake model returning per-dim logits
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval'`.

- [ ] **Step 3: Implement `src/eval.py`**

```python
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
        clf = LogisticRegression(max_iter=200, n_jobs=-1, multi_class="auto")
        clf.fit(train_proj_full[:, :mm], train_labels)
        preds = clf.predict(val_proj_full[:, :mm])
        out[m] = float((preds == val_labels).mean() * 100.0)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/eval.py tests/test_eval.py
git commit -m "feat: MRL per-dim eval and PCA-truncated baseline accuracy"
```

---

## Task 8: Visualization (`src/viz.py`)

**Files:**
- Create: `src/viz.py`, `tests/test_viz.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_viz.py
from src.viz import plot_loss_curves, plot_accuracy_curves, plot_mrl_accuracy_by_dim


def test_plot_loss_curves_writes_png(tmp_path):
    history = [{"epoch": 0, "train_loss": 2.0, "val_loss": 1.8},
               {"epoch": 1, "train_loss": 1.5, "val_loss": 1.4}]
    out = tmp_path / "loss.png"
    plot_loss_curves(history, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_accuracy_curves_writes_png(tmp_path):
    history = [{"epoch": 0, "val_top1": 10.0, "val_top5": 30.0}]
    out = tmp_path / "acc.png"
    plot_accuracy_curves(history, str(out))
    assert out.exists()


def test_plot_mrl_accuracy_by_dim_writes_png(tmp_path):
    per_dim = {8: 12.0, 16: 20.0, 2048: 60.0}
    out = tmp_path / "bydim.png"
    plot_mrl_accuracy_by_dim(per_dim, epoch=3, out_path=str(out))
    assert out.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.viz'`.

- [ ] **Step 3: Implement `src/viz.py`**

```python
"""Per-epoch matplotlib plots. Uses the non-interactive Agg backend."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _epochs(history: list[dict]) -> list[int]:
    return [h["epoch"] for h in history]


def plot_loss_curves(history: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(_epochs(history), [h["train_loss"] for h in history], label="train")
    ax.plot(_epochs(history), [h["val_loss"] for h in history], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("Loss")
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_accuracy_curves(history: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(_epochs(history), [h["val_top1"] for h in history], label="top-1")
    if all("val_top5" in h for h in history):
        ax.plot(_epochs(history), [h["val_top5"] for h in history], label="top-5")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.legend(); ax.set_title("Val accuracy")
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_mrl_accuracy_by_dim(per_dim: dict[int, float], epoch: int, out_path: str) -> None:
    dims = sorted(per_dim)
    fig, ax = plt.subplots()
    ax.bar([str(d) for d in dims], [per_dim[d] for d in dims])
    ax.set_xlabel("nesting dim"); ax.set_ylabel("top-1 (%)")
    ax.set_title(f"MRL top-1 by dim (epoch {epoch})")
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_mrl_loss_components(history: list[dict], dims: list[int], out_path: str) -> None:
    """history rows carry 'loss_components': {dim: ce}. Plot each dim over epochs."""
    fig, ax = plt.subplots()
    for m in dims:
        ys = [h["loss_components"][str(m)] if str(m) in h["loss_components"]
              else h["loss_components"][m] for h in history if "loss_components" in h]
        xs = [h["epoch"] for h in history if "loss_components" in h]
        ax.plot(xs, ys, label=f"dim {m}")
    ax.set_xlabel("epoch"); ax.set_ylabel("CE"); ax.legend(fontsize=7); ax.set_title("MRL loss components")
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/viz.py tests/test_viz.py
git commit -m "feat: per-epoch matplotlib plots"
```

---

## Task 9: Training engine (`src/train.py`)

**Files:**
- Create: `src/train.py`, `tests/test_train_smoke.py`

This module ties everything together: device setup, logging, warmup+cosine LR,
AMP, grad accumulation, SIGINT handler, per-epoch eval/plots/jsonl, checkpointing,
resume. It is exercised by a CPU smoke test on a synthetic dataset.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_train_smoke.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import build_scheduler, run_training
from src.config import load_config


def _tiny_loaders(num_classes=3, n=12):
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def test_build_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = build_scheduler(opt, warmup_epochs=2, total_epochs=10, steps_per_epoch=1)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    assert lrs[0] < lrs[1] < lrs[2]      # warmup ramps up
    assert lrs[-1] < lrs[2]              # cosine decays after


def test_run_training_mrl_smoke(tmp_path):
    cfg = load_config("configs/default.yaml", overrides={
        "schedule.epochs": 1, "schedule.warmup_epochs": 0,
        "amp.enabled": False, "amp.compile": False,
        "run.output_root": str(tmp_path),
    })
    train_loader, val_loader = _tiny_loaders()
    result = run_training(
        cfg=cfg, model_kind="mrl", run_name="smoke",
        train_loader=train_loader, val_loader=val_loader,
        num_classes=3, device="cpu", nesting_dims=[8, 2048],
    )
    run_dir = tmp_path / "smoke"
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "plots" / "loss_curves.png").exists()
    assert result["epochs_completed"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_train_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train'`.

- [ ] **Step 3: Implement `src/train.py`**

```python
"""Explicit training loop for both arms. No framework abstractions."""
from __future__ import annotations

import logging
import math
import os
import signal
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from src.checkpoint import (atomic_save, capture_rng_state, git_commit_hash,
                            load_checkpoint, restore_rng_state)
from src.eval import evaluate_mrl_per_dim
from src.losses import MRLLoss
from src.metrics import JsonlWriter, accuracy_topk, grad_norm
from src.models import build_model
from src import viz

logger = logging.getLogger("train")


def setup_logging(run_dir: str, level: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    handlers = [logging.StreamHandler(),
                logging.FileHandler(os.path.join(run_dir, "train.log"))]
    logging.basicConfig(level=getattr(logging, level), force=True,
                        format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def log_device_info(device: str) -> None:
    logger.info("PyTorch %s", torch.__version__)
    if device == "cuda":
        i = torch.cuda.current_device()
        logger.info("GPU: %s | VRAM %.1f GB | CUDA %s | cuDNN %s",
                    torch.cuda.get_device_name(i),
                    torch.cuda.get_device_properties(i).total_memory / 1e9,
                    torch.version.cuda, torch.backends.cudnn.version())
    else:
        logger.warning("Running on CPU — training will be UNUSABLY SLOW for real runs.")


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                    steps_per_epoch: int) -> LambdaLR:
    """Linear warmup for warmup_epochs, then cosine decay to 0 over the rest."""
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    total_steps = max(1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


class _Interrupt:
    """SIGINT handler: flips a flag so the loop saves a checkpoint and exits after
    the current batch instead of dying mid-write."""

    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, *_):
        logger.warning("SIGINT received — will checkpoint and exit after this batch.")
        self.stop = True


def _accum_steps(cfg: dict) -> int:
    dl = cfg["dataloader"]
    return max(1, dl["effective_batch_size"] // dl["batch_size"])


def run_training(cfg, model_kind, run_name, train_loader, val_loader,
                 num_classes, device, nesting_dims=None, resume_path=None) -> dict:
    run_dir = os.path.join(cfg["run"]["output_root"], run_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    setup_logging(run_dir, cfg["run"]["log_level"])
    log_device_info(device)

    is_mrl = model_kind == "mrl"
    if is_mrl and nesting_dims is None:
        nesting_dims = cfg["model"]["mrl_nesting_dims"]

    model = build_model(model_kind, num_classes=num_classes, nesting_dims=nesting_dims).to(device)
    if cfg["amp"]["compile"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:  # noqa: BLE001
            logger.warning("torch.compile failed, continuing eager: %s", e)

    ls = cfg["optim"]["label_smoothing"]
    if is_mrl:
        from src.config import resolve_dim_weights
        weights = resolve_dim_weights(nesting_dims, cfg["model"]["mrl_weighting"],
                                      cfg["model"]["mrl_dim_weights"])
        criterion = MRLLoss(nesting_dims, weights, ls)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    o = cfg["optim"]
    optimizer = torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"],
                                weight_decay=o["weight_decay"], nesterov=o["nesterov"])
    steps_per_epoch = max(1, len(train_loader) // _accum_steps(cfg))
    scheduler = build_scheduler(optimizer, cfg["schedule"]["warmup_epochs"],
                                cfg["schedule"]["epochs"], steps_per_epoch)
    use_amp = cfg["amp"]["enabled"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    writer = JsonlWriter(os.path.join(run_dir, "metrics.jsonl"))
    start_epoch, best = 0, -1.0

    if resume_path:
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"]); scaler.load_state_dict(ckpt["scaler"])
        restore_rng_state(ckpt["rng"]); start_epoch = ckpt["epoch"] + 1
        best = ckpt["best_metric"]; history = ckpt.get("history", [])
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    interrupt = _Interrupt()
    accum = _accum_steps(cfg)

    for epoch in range(start_epoch, cfg["schedule"]["epochs"]):
        model.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); running_loss = 0.0; seen = 0
        running_components: dict[int, float] = {}
        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                if is_mrl:
                    loss, parts = criterion(out, y)
                    for m, v in parts.items():
                        running_components[m] = running_components.get(m, 0.0) + v
                else:
                    loss = criterion(out, y)
                loss = loss / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); scheduler.step()
            running_loss += loss.item() * accum * y.size(0); seen += y.size(0)
            if interrupt.stop:
                break

        gnorm = grad_norm(model.parameters())
        train_loss = running_loss / max(1, seen)
        val_loss, val_acc = _validate(model, val_loader, criterion, is_mrl,
                                      nesting_dims, device, cfg["eval"]["topk"])
        epoch_secs = time.time() - t0
        throughput = seen / epoch_secs if epoch_secs > 0 else 0.0
        peak_mem = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0

        record = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_top1": val_acc[1], "val_top5": val_acc.get(5, float("nan")),
            "lr": optimizer.param_groups[0]["lr"], "epoch_secs": epoch_secs,
            "throughput": throughput, "peak_mem_gb": peak_mem, "grad_norm": gnorm,
        }
        if is_mrl:
            per_dim = evaluate_mrl_per_dim(model, val_loader, nesting_dims, device,
                                           ks=tuple(cfg["eval"]["topk"]))
            record["mrl_top1_by_dim"] = {str(m): per_dim[m][1] for m in nesting_dims}
            record["mrl_top5_by_dim"] = {str(m): per_dim[m].get(5, float("nan")) for m in nesting_dims}
            record["loss_components"] = {str(m): running_components[m] / max(1, len(train_loader))
                                         for m in nesting_dims}
        history.append(record); writer.append(record)
        logger.info("epoch %d train_loss=%.4f val_top1=%.2f lr=%.5f %.1fs",
                    epoch, train_loss, val_acc[1], record["lr"], epoch_secs)

        _write_plots(history, record, is_mrl, nesting_dims, plots_dir)

        ckpt = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_metric": max(best, val_acc[1]),
            "nesting_dims": nesting_dims, "rng": capture_rng_state(),
            "git": git_commit_hash(), "history": history,
        }
        atomic_save(ckpt, os.path.join(run_dir, "last.pt"))
        if val_acc[1] > best:
            best = val_acc[1]; atomic_save(ckpt, os.path.join(run_dir, "best.pt"))

        if interrupt.stop:
            logger.warning("Stopping early after epoch %d due to SIGINT.", epoch)
            return {"epochs_completed": epoch - start_epoch + 1, "interrupted": True}

    return {"epochs_completed": cfg["schedule"]["epochs"] - start_epoch, "interrupted": False}


@torch.no_grad()
def _validate(model, loader, criterion, is_mrl, nesting_dims, device, topk):
    model.eval(); total_loss = 0.0; seen = 0
    acc_acc = {k: 0.0 for k in topk}
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        if is_mrl:
            loss, _ = criterion(out, y)
            logits_full = out[max(nesting_dims)]
        else:
            loss = criterion(out, y); logits_full = out
        bs = y.size(0); seen += bs; total_loss += loss.item() * bs
        a = accuracy_topk(logits_full, y, ks=tuple(topk))
        for k in topk:
            acc_acc[k] += a[k] * bs / 100.0
    return total_loss / max(1, seen), {k: acc_acc[k] * 100.0 / max(1, seen) for k in topk}


def _write_plots(history, record, is_mrl, nesting_dims, plots_dir):
    viz.plot_loss_curves(history, os.path.join(plots_dir, "loss_curves.png"))
    viz.plot_accuracy_curves(history, os.path.join(plots_dir, "accuracy_curves.png"))
    if is_mrl:
        per_dim = {int(m): record["mrl_top1_by_dim"][str(m)] for m in nesting_dims}
        viz.plot_mrl_accuracy_by_dim(per_dim, record["epoch"],
                                     os.path.join(plots_dir, "mrl_accuracy_by_dim.png"))
        viz.plot_mrl_loss_components(history, nesting_dims,
                                     os.path.join(plots_dir, "mrl_loss_components.png"))
```

> **Note:** the baseline PCA-truncated per-dim curve is computed in `scripts/train.py` (it needs the train loader + sklearn and runs every 10 epochs), not inside the hot loop. See Task 10. The smoke test only exercises the MRL/baseline core loop.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_train_smoke.py -v`
Expected: PASS (2 passed). May take ~1–2 min on CPU (real ResNet50 forward/backward on 12 tiny samples).

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train_smoke.py
git commit -m "feat: explicit training loop with AMP, warmup+cosine, SIGINT, resume"
```

---

## Task 10: CLI entry point (`scripts/train.py`)

**Files:**
- Create: `scripts/train.py`, `scripts/__init__.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import subprocess
import sys


def test_train_cli_help_lists_flags():
    out = subprocess.run([sys.executable, "scripts/train.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ["--model", "--run-name", "--epochs", "--lr", "--quick",
                 "--full", "--resume", "--seed", "--nesting-dims",
                 "--mrl-weighting", "--label-smoothing"]:
        assert flag in out.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — file not found / nonzero return.

- [ ] **Step 3: Implement `scripts/train.py`**

```python
"""CLI entry point: configure and launch a training run for one arm."""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import build_dataloaders
from src.eval import extract_features, pca_truncated_accuracy
from src.train import run_training

logger = logging.getLogger("train")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train baseline or MRL ResNet50.")
    p.add_argument("--model", choices=["baseline", "mrl"], required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--nesting-dims", default=None, help="comma-separated, e.g. 8,16,2048")
    p.add_argument("--mrl-weighting", choices=["uniform", "increasing"], default=None)
    p.add_argument("--mrl-dim-weights", default=None, help="comma-separated floats")
    p.add_argument("--quick", action="store_true", help="100-class subset (default on)")
    p.add_argument("--full", action="store_true", help="use full ImageNet-1K")
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "schedule.epochs": args.epochs, "optim.lr": args.lr,
        "dataloader.batch_size": args.batch_size, "run.seed": args.seed,
        "optim.label_smoothing": args.label_smoothing,
        "model.mrl_weighting": args.mrl_weighting,
    }
    if args.nesting_dims:
        overrides["model.mrl_nesting_dims"] = [int(x) for x in args.nesting_dims.split(",")]
    if args.mrl_dim_weights:
        overrides["model.mrl_dim_weights"] = [float(x) for x in args.mrl_dim_weights.split(",")]
    cfg = load_config(args.config, overrides=overrides)
    if args.full:
        cfg["data"]["quick"]["enabled"] = False
    elif args.quick:
        cfg["data"]["quick"]["enabled"] = True

    seed_everything(cfg["run"]["seed"])
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or args.model
    nesting_dims = cfg["model"]["mrl_nesting_dims"] if args.model == "mrl" else None

    train_loader, val_loader, num_classes = build_dataloaders(cfg)
    result = run_training(cfg=cfg, model_kind=args.model, run_name=run_name,
                          train_loader=train_loader, val_loader=val_loader,
                          num_classes=num_classes, device=device,
                          nesting_dims=nesting_dims, resume_path=args.resume)
    logger.info("Training finished: %s", result)


if __name__ == "__main__":
    main()
```

> **Baseline PCA curve hook:** add a post-training pass for the baseline arm. After `run_training`, if `args.model == "baseline"`, extract features (cap `eval.pca.fit_sample_size`) from train+val via `model.embed`, call `pca_truncated_accuracy` for `cfg["model"]["mrl_nesting_dims"]`, and append a final record to `metrics.jsonl` with key `pca_top1_by_dim`. (Implement by having `run_training` also return the trained `model`; update its return dict to include `"model"`. Update the Task 9 return statements and the smoke test's assertions remain valid since they only check listed keys.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Manual smoke (optional, needs IMAGENET_PATH)**

Run: `uv run python scripts/train.py --model mrl --quick --epochs 1 --run-name cli_smoke`
Expected: completes one epoch, writes `runs/cli_smoke/{last.pt,metrics.jsonl,plots/}`.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py scripts/__init__.py tests/test_cli.py src/train.py
git commit -m "feat: training CLI entry point with config overrides and baseline PCA pass"
```

---

## Task 11: Dataset init (`scripts/init_data.py`)

**Files:**
- Create: `scripts/init_data.py`, `tests/test_init_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_init_data.py
from scripts.init_data import detect_credentials, dataset_present


def test_dataset_present_true_when_train_and_val_exist(tmp_path):
    (tmp_path / "train").mkdir(); (tmp_path / "val").mkdir()
    assert dataset_present(str(tmp_path)) is True


def test_dataset_present_false_when_missing(tmp_path):
    assert dataset_present(str(tmp_path)) is False


def test_detect_credentials_none(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("scripts.init_data._kaggle_json_exists", lambda: False)
    monkeypatch.setattr("scripts.init_data._hf_token_exists", lambda: False)
    assert detect_credentials() == []


def test_detect_credentials_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "u"); monkeypatch.setenv("KAGGLE_KEY", "k")
    monkeypatch.setattr("scripts.init_data._hf_token_exists", lambda: False)
    assert "kaggle" in detect_credentials()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_init_data.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `scripts/init_data.py`**

```python
"""Detect dataset, download via available credentials, or print setup help."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SETUP_HELP = """\
No ImageNet credentials found. Configure ONE of:

Kaggle:
  pip/uv add 'kaggle'; place kaggle.json in ~/.kaggle/ OR set KAGGLE_USERNAME + KAGGLE_KEY
  Accept rules at: https://www.kaggle.com/c/imagenet-object-localization-challenge

Hugging Face:
  set HF_TOKEN (or run `huggingface-cli login`); accept license for timm/imagenet-1k-wds

Then set IMAGENET_PATH to the target directory and re-run:
  uv run python scripts/init_data.py
"""


def dataset_present(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "val"))


def _kaggle_json_exists() -> bool:
    return os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))


def _hf_token_exists() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    return os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))


def detect_credentials() -> list[str]:
    creds = []
    if (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or _kaggle_json_exists():
        creds.append("kaggle")
    if os.environ.get("HF_TOKEN") or _hf_token_exists():
        creds.append("huggingface")
    return creds


def download_kaggle(dest: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    print(f"Downloading ImageNet via Kaggle into {dest} (large, slow)...")
    api.competition_download_files("imagenet-object-localization-challenge", path=dest)
    print("Download complete. Unzip and arrange into train/ and val/ ImageFolder layout.")


def download_hf(dest: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"Downloading timm/imagenet-1k-wds into {dest} (large, slow)...")
    snapshot_download(repo_id="timm/imagenet-1k-wds", repo_type="dataset", local_dir=dest)
    print("Download complete. Convert WebDataset shards to ImageFolder layout as needed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Initialize the ImageNet dataset.")
    parser.add_argument("--path", default=os.environ.get("IMAGENET_PATH"))
    args = parser.parse_args(argv)

    if not args.path:
        print("Set IMAGENET_PATH or pass --path.", file=sys.stderr)
        print(SETUP_HELP, file=sys.stderr); return 2
    if dataset_present(args.path):
        print(f"Dataset already present at {args.path}. Skipping."); return 0

    creds = detect_credentials()
    if not creds:
        print(SETUP_HELP, file=sys.stderr); return 2

    os.makedirs(args.path, exist_ok=True)
    if "kaggle" in creds:
        download_kaggle(args.path)
    else:
        download_hf(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_init_data.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/init_data.py tests/test_init_data.py
git commit -m "feat: dataset init with credential detection and setup help"
```

---

## Task 12: Analysis report (`scripts/analyze.py`)

**Files:**
- Create: `scripts/analyze.py`, `tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analyze.py
import json

from scripts.analyze import load_metrics, build_report


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_metrics_reads_all_rows(tmp_path):
    p = tmp_path / "metrics.jsonl"
    _write_jsonl(p, [{"epoch": 0, "val_top1": 1.0}, {"epoch": 1, "val_top1": 2.0}])
    rows = load_metrics(str(p))
    assert len(rows) == 2 and rows[1]["val_top1"] == 2.0


def test_build_report_contains_headline_numbers(tmp_path):
    base = [{"epoch": 0, "val_top1": 50.0, "val_top5": 80.0,
             "pca_top1_by_dim": {"8": 10.0, "2048": 50.0}}]
    mrl = [{"epoch": 0, "val_top1": 49.0, "val_top5": 79.0,
            "mrl_top1_by_dim": {"8": 30.0, "2048": 49.0}}]
    md = build_report(base, mrl)
    assert "# MRL vs Baseline" in md
    assert "8" in md and "2048" in md
    assert "49.0" in md or "49" in md
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `scripts/analyze.py`**

```python
"""Post-hoc cross-run analysis: report markdown + comparison plots."""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _last_with(rows, key):
    for r in reversed(rows):
        if key in r:
            return r
    return None


def build_report(baseline_rows: list[dict], mrl_rows: list[dict]) -> str:
    b_final = baseline_rows[-1]; m_final = mrl_rows[-1]
    pca = _last_with(baseline_rows, "pca_top1_by_dim")
    mrl_dim = _last_with(mrl_rows, "mrl_top1_by_dim")
    lines = ["# MRL vs Baseline ResNet50 — Analysis", ""]
    lines += ["## Full-dim (2048) accuracy", "",
              f"- Baseline top-1: {b_final.get('val_top1'):.2f}  top-5: {b_final.get('val_top5'):.2f}",
              f"- MRL top-1: {m_final.get('val_top1'):.2f}  top-5: {m_final.get('val_top5'):.2f}", ""]
    if pca and mrl_dim:
        lines += ["## Accuracy across nesting dims (top-1 %)", "",
                  "| dim | MRL | PCA-baseline |", "|---|---|---|"]
        dims = sorted((int(d) for d in mrl_dim["mrl_top1_by_dim"]))
        for d in dims:
            mv = mrl_dim["mrl_top1_by_dim"].get(str(d), float("nan"))
            pv = pca["pca_top1_by_dim"].get(str(d), float("nan"))
            lines.append(f"| {d} | {mv:.1f} | {pv:.1f} |")
        lines.append("")
    return "\n".join(lines)


def plot_accuracy_vs_dim(baseline_rows, mrl_rows, out_path):
    pca = _last_with(baseline_rows, "pca_top1_by_dim")
    mrl_dim = _last_with(mrl_rows, "mrl_top1_by_dim")
    if not (pca and mrl_dim):
        return
    dims = sorted(int(d) for d in mrl_dim["mrl_top1_by_dim"])
    fig, ax = plt.subplots()
    ax.plot(dims, [mrl_dim["mrl_top1_by_dim"][str(d)] for d in dims], "o-", label="MRL")
    ax.plot(dims, [pca["pca_top1_by_dim"][str(d)] for d in dims], "s--", label="PCA baseline")
    ax.set_xscale("log", base=2); ax.set_xlabel("embedding dim")
    ax.set_ylabel("top-1 (%)"); ax.legend(); ax.set_title("Accuracy vs embedding dim")
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _overlay(baseline_rows, mrl_rows, key, ylabel, title, out_path, log=False):
    fig, ax = plt.subplots()
    for rows, label in [(baseline_rows, "baseline"), (mrl_rows, "mrl")]:
        xs = [r["epoch"] for r in rows if key in r]
        ys = [r[key] for r in rows if key in r]
        if xs:
            ax.plot(xs, ys, label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = argparse.ArgumentParser(description="Compare baseline vs MRL runs.")
    p.add_argument("--baseline", required=True, help="baseline run dir")
    p.add_argument("--mrl", required=True, help="mrl run dir")
    p.add_argument("--out", default="analysis")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    b = load_metrics(os.path.join(args.baseline, "metrics.jsonl"))
    m = load_metrics(os.path.join(args.mrl, "metrics.jsonl"))

    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(build_report(b, m))
    plot_accuracy_vs_dim(b, m, os.path.join(args.out, "accuracy_vs_dim.png"))
    _overlay(b, m, "val_loss", "val loss", "Validation loss", os.path.join(args.out, "loss_overlay.png"))
    _overlay(b, m, "throughput", "images/sec", "Throughput", os.path.join(args.out, "time_per_epoch.png"))
    _overlay(b, m, "peak_mem_gb", "GB", "Peak VRAM", os.path.join(args.out, "memory.png"))
    print(f"Wrote report and plots to {args.out}/")


if __name__ == "__main__":
    main()
```

> **Latency benchmark plot:** `analysis/latency_benchmark.png` requires loading the
> trained checkpoints and timing inference per dim (warmup 100 batches, average over
> 1000). Add a `--baseline-ckpt`/`--mrl-ckpt` optional pair; when given, load models,
> run `torch.cuda.synchronize()`-bracketed timing at each dim, and plot. When absent,
> skip with a logged note. Implement after the core report passes; add a test that
> the function no-ops cleanly when checkpoints are not provided.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze.py tests/test_analyze.py
git commit -m "feat: cross-run analysis report and comparison plots"
```

---

## Task 13: README and full-suite verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# MRL vs Baseline ResNet50 on ImageNet-1K

Compare a baseline ResNet50 against a Matryoshka-Efficient (MRL-E) variant: same
backbone, the final head is the only difference. See
`docs/superpowers/specs/2026-05-22-mrl-vs-baseline-resnet50-design.md`.

## Setup

```bash
uv sync                 # core training deps + .venv
uv sync --extra data    # add kaggle + huggingface_hub for dataset download
```

## Dataset

Set `IMAGENET_PATH` to a dir with `train/` and `val/` (ImageFolder layout):

```bash
export IMAGENET_PATH=/path/to/imagenet      # PowerShell: $env:IMAGENET_PATH="..."
uv run python scripts/init_data.py          # downloads if creds present, else prints help
```

Runs default to a deterministic **100-class subset** (`--quick`). Use `--full` for
all 1000 classes (the multi-day big run).

## Train

```bash
uv run python scripts/train.py --model baseline --run-name baseline
uv run python scripts/train.py --model mrl --run-name mrl
```

Common flags: `--epochs`, `--lr`, `--batch-size`, `--seed`, `--label-smoothing`,
`--nesting-dims 8,16,2048`, `--mrl-weighting {uniform,increasing}`,
`--mrl-dim-weights ...`, `--quick`/`--full`, `--resume runs/mrl/last.pt`.

Ctrl+C checkpoints cleanly mid-run; resume with `--resume`.

## Analyze

```bash
uv run python scripts/analyze.py --baseline runs/baseline --mrl runs/mrl
```

Writes `analysis/report.md` plus comparison plots.

## Outputs

Per run under `runs/<name>/`: `last.pt`, `best.pt`, `metrics.jsonl`, `train.log`,
`plots/`. Cross-run analysis under `analysis/`.

## Expected timing (single RTX 4060 Ti)

- `--quick --epochs 3` smoke: < 30 min.
- Full 1000-class, 90 epochs: multiple days — treat as the big run.
- CPU: works for tests only; real training is unusably slow (logged loudly).
````

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -v`
Expected: ALL pass (config, losses, models, checkpoint, metrics, data, eval, viz, train smoke, cli, init_data, analyze).

- [ ] **Step 3: Verify the smoke-test definition-of-done locally (needs IMAGENET_PATH)**

Run: `uv run python scripts/train.py --quick --epochs 3 --model mrl --run-name dod_smoke`
Expected: completes < 30 min on RTX 4060 Ti; produces checkpoints, plots, jsonl.
Then: kill a fresh run with Ctrl+C mid-epoch, confirm `last.pt` exists, and
`--resume runs/<name>/last.pt` continues.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, run, and analysis instructions"
```

---

## Self-Review Notes

- **Spec coverage:** config ✓ (T0/T1), MRL-E head + models ✓ (T3), MRL loss + configurable weighting ✓ (T1/T2), data + quick subset + path resolution ✓ (T6), training recipe/AMP/compile/warmup-cosine/grad-accum/SIGINT/resume ✓ (T9), device logging ✓ (T9), atomic checkpoint + RNG + git hash ✓ (T4/T9), per-epoch metrics incl. grad norm/throughput/peak mem ✓ (T5/T9), per-epoch plots incl. MRL-specific ✓ (T8/T9), PCA-truncated baseline curve ✓ (T7/T10), analysis report + headline/loss/throughput/memory plots ✓ (T12), latency benchmark plot ✓ (T12 note, deferred sub-step), init_data credential detection ✓ (T11), README + DoD ✓ (T13).
- **Deferred within-task items (not placeholders — concrete instructions given):** baseline PCA post-pass wiring (T10 note), latency benchmark plot (T12 note). Both have explicit implementation directions and are small.
- **Type consistency:** `build_model(kind, num_classes, nesting_dims)`, `MRLLoss(nesting_dims, dim_weights, label_smoothing)`, `MRLHead(embedding_dim, num_classes, nesting_dims)`, `accuracy_topk(logits, targets, ks)`, `evaluate_mrl_per_dim(model, loader, dims, device, ks)`, `run_training(...)` signatures are consistent across tasks and tests.
- **Note for executor:** `run_training` must be updated to also return the trained `model` in its result dict (used by T10's baseline PCA pass). Smoke-test assertions only check named keys, so they remain valid.
