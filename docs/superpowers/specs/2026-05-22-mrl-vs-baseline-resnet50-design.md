# MRL vs Baseline ResNet50 Comparison on ImageNet-1K

**Date:** 2026-05-22
**Status:** Approved design (pending spec review)

## Goal

Implement and rigorously compare a baseline ResNet50 against a Matryoshka
Representation Learning (MRL) variant on image classification. The experiment
answers two questions:

1. At full embedding dim (2048), does MRL hurt top-1/top-5 accuracy vs baseline?
2. Across nested dims {8, 16, …, 2048}, how does MRL's accuracy curve compare to
   PCA-truncated baseline embeddings?

Both arms train **our own** ResNet50 from scratch (random init, `weights=None`).
"ResNet50" is only the standard architecture blueprint from torchvision; every
weight is learned during our training loop. The arms differ in exactly one place:
the final classification head.

## Background (given, not re-derived)

MRL trains a single backbone but applies classification at multiple nested
embedding dimensions simultaneously:

    Loss = Σ over m ∈ M of w_m · CE(W_m · z[:m], y)

We use the **MRL-E (efficient)** variant: one shared classifier weight matrix
`W ∈ R^{1000×2048}` and one shared bias `b ∈ R^{1000}`. For dim `m`,
`logits_m = z[:, :m] @ W[:, :m].T + b`. The convolutional backbone is untouched —
only the head changes. Default weights `w_m` are equal across dims.

## Key clarifications (resolved during brainstorming)

- **Dataset scale:** Default to the `--quick` 100-class subset (deterministic by
  seed) for practical runs. Full ImageNet-1K is the opt-in "big run" via `--full`.
- **Dataset download:** `scripts/init_data.py` detects whichever of Kaggle /
  Hugging Face credentials are present and uses that; if neither, it prints
  copy-pasteable setup instructions and exits cleanly (no crash). If the dataset
  already exists and validates, it skips.
- **Baseline `embed()`:** ResNet50's final layer is `model.fc` (2048→1000). For
  the PCA comparison we need the 2048-d feature *before* `fc`. We swap `model.fc`
  for `nn.Identity()`, keep a reference to the real FC, and apply it ourselves for
  logits. `embed(x)` returns the 2048-d feature; `forward(x)` applies the real FC.
- **Baseline loss:** plain `CrossEntropyLoss(label_smoothing)`, single term. The
  configurable-loss surface (dim weighting schemes) applies only to the MRL arm.
  Label smoothing is held identical across both arms by design.
- **PCA per-dim baseline accuracy:** for each dim `m`, fit PCA→`m` on a fixed 50k
  train-embedding subset, train a linear probe (logistic regression) on the `m`-d
  projection, evaluate. Fit every 10 epochs (cached in checkpoint) to save compute.

## Dataset

ImageNet-1K via `IMAGENET_PATH` env var (standard `train/` + `val/` ImageFolder
layout). If unset, raise a clear error pointing to Kaggle
`imagenet-object-localization-challenge` and HF `timm/imagenet-1k-wds`.

`--quick` (default on) subsets to 100 deterministic classes for ~10x faster
iteration. `--full` opts into all 1000 classes.

Preprocessing (clean comparison — no heavy augmentation):
- Train: RandomResizedCrop(224), RandomHorizontalFlip, normalize (ImageNet stats)
- Val: Resize(256), CenterCrop(224), normalize

## Models (`src/models.py`)

- `BaselineResNet50`: torchvision `resnet50(weights=None)`, default 2048→1000 FC.
  `embed()` returns pre-FC 2048-d feature (Identity-swap approach).
- `MRLResNet50`: same backbone, FC replaced by `MRLHead`. `forward()` returns
  `{m: logits_m}`; `embed()` returns the full 2048-d feature.
- `MRLHead`: shared `Linear(2048, 1000)`; `forward(z)` returns `{m: z[:,:m] @
  W[:,:m].T + b}` for each nesting dim.
- `build_model(kind, ...)`: factory for `'baseline' | 'mrl'`.

## Loss (`src/losses.py`)

- `MRLLoss(nesting_dims, dim_weights, label_smoothing)`: weighted sum of CE over
  all dims with label smoothing. `forward()` returns `(total_loss, {m: ce_value})`
  for per-dim logging. Equal weights if `dim_weights` is None.

### Configurable loss surface (CLI overrides config)

| CLI flag | Config key | Default | Effect |
|---|---|---|---|
| `--label-smoothing` | `optim.label_smoothing` | `0.1` | Both arms, held identical. |
| `--mrl-dim-weights` | `model.mrl_dim_weights` | `null` (equal) | Explicit per-dim weights. |
| `--mrl-weighting` | `model.mrl_weighting` | `uniform` | `uniform` or `increasing` scheme; ignored if explicit weights given. |
| `--nesting-dims` | `model.mrl_nesting_dims` | `[8,…,2048]` | Which dims to train. |

## Training recipe (identical across arms)

- SGD, momentum 0.9, weight decay 1e-4, nesterov=False
- LR 0.1, cosine schedule, 5-epoch linear warmup
- Batch size 256 (gradient accumulation to effective 256 if VRAM < 16GB,
  auto-detected and logged loudly; config exposes `batch_size` +
  `effective_batch_size` explicitly)
- Epochs 90 (CLI configurable)
- Label smoothing 0.1
- Mixed precision: `torch.amp.autocast('cuda')` + `GradScaler`
- Seed 42 (CLI configurable): seed torch, numpy, python random, CUDA
- `torch.compile()` if PyTorch ≥ 2.0, wrapped in try/except with loud fallback log
- **Both arms must share seed, DataLoader settings, and all hyperparameters** —
  the experiment is invalid otherwise.

## Device handling

- `device = cuda if available else cpu`
- Startup log: GPU name, VRAM total, CUDA version, cuDNN version, PyTorch version
- `torch.backends.cudnn.benchmark = True`
- DataLoader: `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`,
  `num_workers=min(8, os.cpu_count())`
- CPU fallback works but logs a prominent "training will be unusably slow" warning

## Checkpointing (interruption-resilient)

After every epoch, save to `runs/<run_name>/`:
- `last.pt` (latest), `best.pt` (highest val top-1 so far)

Contents: model state_dict, optimizer state, scheduler state, GradScaler state,
epoch number, best metric, MRL nesting dims, RNG states (torch CPU + CUDA, numpy,
python random), cached PCA (baseline), git commit hash if available.

**Atomic writes:** write `last.pt.tmp`, then `os.replace()` → `last.pt`.

`--resume <path>` fully restores RNG state; training continues bit-identically
(within FP nondeterminism).

`SIGINT` handler: on Ctrl+C finish the current batch, save a checkpoint, exit
cleanly, log clearly.

## Per-epoch outputs

To `runs/<run_name>/plots/` (overwrite each epoch):
- `loss_curves.png`: train + val loss vs epoch
- `accuracy_curves.png`: top-1 and top-5 vs epoch
- MRL only: `mrl_accuracy_by_dim.png` (bar chart, current epoch)
- MRL only: `mrl_loss_components.png` (per-dim CE over time)

Append one JSON line per epoch to `runs/<run_name>/metrics.jsonl`
(machine-parseable). matplotlib for plots.

### Metrics per epoch

Train loss (epoch mean), val loss; top-1/top-5 on val; MRL: top-1/top-5 at every
nesting dim; baseline: top-1 at each dim via PCA truncation (fit every 10 epochs
on fixed 50k subset, cached); epoch wall-clock seconds; train throughput
(images/sec); current LR; peak GPU memory (reset each epoch); mean gradient norm.

## Final analysis (`scripts/analyze.py`)

Reads `metrics.jsonl` from both run dirs; produces:
- `analysis/report.md`: summary table + key findings
- `analysis/accuracy_vs_dim.png`: headline — MRL vs PCA-truncated baseline across
  dims, log-scale x-axis
- `analysis/loss_overlay.png`: train + val loss, both arms overlaid
- `analysis/time_per_epoch.png`: throughput comparison
- `analysis/memory.png`: peak VRAM over time
- `analysis/latency_benchmark.png`: inference latency per sample at each dim, both
  arms (warmup 100 batches, average over 1000)

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore                # ignores runs/ and the dataset dir
├── configs/
│   └── default.yaml          # all hyperparameters; CLI overrides
├── src/
│   ├── __init__.py
│   ├── data.py               # loaders, transforms, --quick subset logic
│   ├── models.py             # BaselineResNet50, MRLResNet50, MRLHead
│   ├── losses.py             # MRLLoss
│   ├── train.py              # training loop, AMP, checkpointing, SIGINT
│   ├── eval.py               # validation, PCA truncation, per-dim accuracy
│   ├── checkpoint.py         # atomic save/load, RNG state handling
│   ├── viz.py                # per-epoch plotting
│   └── metrics.py            # metric tracking + jsonl writer
├── scripts/
│   ├── init_data.py          # detect creds, download/skip, validate
│   ├── train.py              # entry: --model {baseline,mrl}
│   └── analyze.py            # post-hoc analysis + report
└── runs/                     # gitignored
```

## Dependencies (`requirements.txt`)

`torch>=2.1`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`,
`pyyaml`, plus `kaggle` and `huggingface_hub` for `init_data.py`. No Lightning,
no Accelerate, no timm — explicit, readable training loop.

## Logging

Python `logging` with a file handler → `runs/<run_name>/train.log`. No `print`.

## Anti-patterns to avoid

- No framework abstractions hiding the training loop — every line visible.
- No silent AMP / `torch.compile` fallback — log loudly.
- No PCA every epoch — every 10 epochs.
- No differing seeds, batch sizes, or augmentation between arms.
- No `print` for logging.
- No over-engineering with abstract base classes — two model classes, one loop.

## Definition of done

- `python scripts/train.py --model baseline --run-name baseline` trains, produces
  checkpoints + plots + jsonl.
- `python scripts/train.py --model mrl --run-name mrl` does the same for MRL.
- Ctrl+C mid-epoch produces a valid checkpoint; `--resume runs/mrl/last.pt`
  continues correctly.
- `python scripts/analyze.py --baseline runs/baseline --mrl runs/mrl` produces the
  comparison report.
- `python scripts/train.py --quick --epochs 3 --model mrl` completes a full smoke
  test in under 30 minutes on a single RTX 4060 Ti.
- README documents env setup, `IMAGENET_PATH` expectations, how to run each arm,
  expected wall-clock per epoch on RTX 4060 Ti, where outputs land.
