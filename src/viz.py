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
