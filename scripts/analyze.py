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
    b_final = _last_with(baseline_rows, "val_top1")
    m_final = _last_with(mrl_rows, "val_top1")
    pca = _last_with(baseline_rows, "pca_top1_by_dim")
    mrl_dim = _last_with(mrl_rows, "mrl_top1_by_dim")
    lines = ["# MRL vs Baseline ResNet50 — Analysis", ""]
    if b_final and m_final:
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


def plot_latency_benchmark(baseline_ckpt, mrl_ckpt, out_path):
    """Inference latency per sample at each dim, both arms. Deferred: not yet
    implemented. Returns False (clean skip) regardless of inputs for now."""
    if not (baseline_ckpt and mrl_ckpt):
        print("Latency benchmark skipped (checkpoints not provided).")
        return False
    print("Latency benchmark not yet implemented — skipping (deferred feature).")
    return False


def main(argv=None):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = argparse.ArgumentParser(description="Compare baseline vs MRL runs.")
    p.add_argument("--baseline", required=True, help="baseline run dir")
    p.add_argument("--mrl", required=True, help="mrl run dir")
    p.add_argument("--out", default="analysis")
    p.add_argument("--baseline-ckpt", default=None)
    p.add_argument("--mrl-ckpt", default=None)
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
    plot_latency_benchmark(args.baseline_ckpt, args.mrl_ckpt,
                           os.path.join(args.out, "latency_benchmark.png"))
    print(f"Wrote report and plots to {args.out}/")


if __name__ == "__main__":
    main()
