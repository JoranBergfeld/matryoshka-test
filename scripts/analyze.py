"""Post-hoc cross-run analysis: report markdown + comparison plots."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

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
    ax.plot(dims, [mrl_dim["mrl_top1_by_dim"].get(str(d), float("nan")) for d in dims],
            "o-", label="MRL")
    ax.plot(dims, [pca["pca_top1_by_dim"].get(str(d), float("nan")) for d in dims],
            "s--", label="PCA baseline")
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


def _time_per_sample(fn, batch_size, warmup_batches, measure_samples, device):
    """Microseconds per sample for `fn` (one call processes `batch_size` rows).
    Warms up `warmup_batches` calls, then averages over enough calls to cover
    `measure_samples`, bracketing the timed region with cuda.synchronize()."""
    for _ in range(warmup_batches):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    iters = max(1, -(-measure_samples // batch_size))  # ceil division
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / (iters * batch_size) * 1e6  # microseconds/sample


@torch.no_grad()
def benchmark_latency_by_dim(dims, embedding_dim=2048, num_classes=1000,
                             batch_size=256, warmup_batches=100,
                             measure_samples=1000, device=None):
    """Head-only inference latency (microseconds/sample) at each embedding dim,
    for both arms. Latency is weight-independent, so random tensors stand in for a
    trained model (no checkpoint or dataset needed).

    MRL classifies a dim-m prefix with one matmul against the shared classifier
    (z[:, :m] @ W[:, :m].T). The baseline has no nested head, so to use dim m it
    must project the full embedding down (the PCA-equivalent 2048->m matmul) and
    then run an m->classes probe. Returns ({dim: us_mrl}, {dim: us_baseline})."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dims = sorted(dims)
    embedding_dim = max(embedding_dim, max(dims))
    z = torch.randn(batch_size, embedding_dim, device=device)
    weight = torch.randn(num_classes, embedding_dim, device=device)
    bias = torch.randn(num_classes, device=device)
    mrl, base = {}, {}
    for m in dims:
        w_m = weight[:, :m]
        proj = torch.randn(embedding_dim, m, device=device)  # PCA-equivalent projection
        probe = torch.randn(m, num_classes, device=device)   # linear probe
        mrl[m] = _time_per_sample(lambda w_m=w_m, m=m: z[:, :m] @ w_m.T + bias,
                                  batch_size, warmup_batches, measure_samples, device)
        base[m] = _time_per_sample(lambda proj=proj, probe=probe: (z @ proj) @ probe,
                                   batch_size, warmup_batches, measure_samples, device)
    return mrl, base


def plot_latency_benchmark(dims, out_path, embedding_dim=2048, num_classes=1000,
                           batch_size=256, warmup_batches=100, measure_samples=1000,
                           device=None):
    """Benchmark head-only per-dim latency and write the comparison plot. Returns
    the (mrl, baseline) timing dicts, or False if `dims` is empty (clean skip)."""
    if not dims:
        print("Latency benchmark skipped (no nesting dims found).")
        return False
    mrl, base = benchmark_latency_by_dim(dims, embedding_dim, num_classes, batch_size,
                                         warmup_batches, measure_samples, device)
    sd = sorted(dims)
    fig, ax = plt.subplots()
    ax.plot(sd, [mrl[d] for d in sd], "o-", label="MRL (prefix slice)")
    ax.plot(sd, [base[d] for d in sd], "s--", label="baseline (project + probe)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("embedding dim"); ax.set_ylabel("latency per sample (µs)")
    ax.set_title("Head-only inference latency vs embedding dim"); ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return mrl, base


def main(argv=None):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = argparse.ArgumentParser(description="Compare baseline vs MRL runs.")
    p.add_argument("--baseline", required=True, help="baseline run dir")
    p.add_argument("--mrl", required=True, help="mrl run dir")
    p.add_argument("--out", default="analysis")
    p.add_argument("--no-latency", action="store_true",
                   help="skip the head-only latency benchmark")
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
    if not args.no_latency:
        mrl_dim = _last_with(m, "mrl_top1_by_dim")
        dims = sorted(int(d) for d in mrl_dim["mrl_top1_by_dim"]) if mrl_dim else []
        plot_latency_benchmark(dims, os.path.join(args.out, "latency_benchmark.png"))
    print(f"Wrote report and plots to {args.out}/")


if __name__ == "__main__":
    main()
