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
    """Linear warmup for warmup_epochs, then cosine decay to 0 over the rest.

    Warmup uses (step+1)/(warmup_steps+1) so the last warmup step is strictly below
    the cosine peak, giving a monotonically increasing ramp into the peak."""
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    total_steps = max(1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
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
            return {"epochs_completed": epoch - start_epoch + 1, "interrupted": True,
                    "model": model}

    return {"epochs_completed": cfg["schedule"]["epochs"] - start_epoch,
            "interrupted": False, "model": model}


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
