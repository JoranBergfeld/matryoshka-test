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


def test_build_report_handles_separate_pca_row(tmp_path):
    # Real baseline output: epoch rows, then a separate PCA-only row (no val_top1)
    base = [
        {"epoch": 0, "val_top1": 50.0, "val_top5": 80.0},
        {"epoch": 0, "pca_top1_by_dim": {"8": 10.0, "2048": 50.0}},
    ]
    mrl = [{"epoch": 0, "val_top1": 49.0, "val_top5": 79.0,
            "mrl_top1_by_dim": {"8": 30.0, "2048": 49.0}}]
    md = build_report(base, mrl)  # must not raise
    assert "Baseline top-1: 50.00" in md
    assert "MRL top-1: 49.00" in md
    assert "| 8 |" in md


def test_benchmark_latency_returns_positive_value_per_dim():
    from scripts.analyze import benchmark_latency_by_dim
    mrl, base = benchmark_latency_by_dim([8, 64, 2048], batch_size=8,
                                         warmup_batches=2, measure_samples=16,
                                         device="cpu")
    assert set(mrl) == {8, 64, 2048}
    assert set(base) == {8, 64, 2048}
    assert all(v > 0 for v in mrl.values())
    assert all(v > 0 for v in base.values())


def test_plot_latency_benchmark_writes_png(tmp_path):
    from scripts.analyze import plot_latency_benchmark
    out = tmp_path / "lat.png"
    res = plot_latency_benchmark([8, 2048], str(out), batch_size=8,
                                 warmup_batches=2, measure_samples=16, device="cpu")
    assert out.exists() and out.stat().st_size > 0
    assert res is not False


def test_plot_latency_benchmark_empty_dims_skips(tmp_path):
    from scripts.analyze import plot_latency_benchmark
    assert plot_latency_benchmark([], str(tmp_path / "x.png"), device="cpu") is False
