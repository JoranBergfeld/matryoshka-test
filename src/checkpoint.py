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
