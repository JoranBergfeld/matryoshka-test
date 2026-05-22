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

## Development

```bash
uv run pytest            # full test suite
```
