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
