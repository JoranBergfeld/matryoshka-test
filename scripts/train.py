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
from src.metrics import JsonlWriter
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


def _run_baseline_pca_pass(cfg, run_name, model, train_loader, val_loader, device):
    """Baseline has no nested head, so derive the per-dim accuracy curve post-hoc:
    extract embeddings, then PCA-truncate + linear-probe at each nesting dim.
    Appends one record with key 'pca_top1_by_dim' to the run's metrics.jsonl."""
    cap = cfg["eval"]["pca"]["fit_sample_size"]
    dims = cfg["model"]["mrl_nesting_dims"]
    logger.info("Baseline PCA-truncation pass over up to %d samples...", cap)
    train_feats, train_labels = extract_features(model, train_loader, device, max_samples=cap)
    val_feats, val_labels = extract_features(model, val_loader, device, max_samples=cap)
    pca_acc = pca_truncated_accuracy(train_feats, train_labels, val_feats, val_labels, dims)
    writer = JsonlWriter(os.path.join(cfg["run"]["output_root"], run_name, "metrics.jsonl"))
    writer.append({"epoch": cfg["schedule"]["epochs"] - 1,
                   "pca_top1_by_dim": {str(d): pca_acc[d] for d in dims}})
    logger.info("PCA-truncation accuracy by dim: %s", pca_acc)


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
    logger.info("Training finished: %s", {k: v for k, v in result.items() if k != "model"})

    if args.model == "baseline":
        _run_baseline_pca_pass(cfg, run_name, result["model"], train_loader, val_loader, device)


if __name__ == "__main__":
    main()
