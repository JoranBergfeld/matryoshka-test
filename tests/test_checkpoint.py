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
    random.random(); np.random.rand(); torch.rand(1)
    restore_rng_state(state)
    second = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert first == second
