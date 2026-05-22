import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import build_scheduler, run_training
from src.config import load_config


def _tiny_loaders(num_classes=3, n=12):
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def test_build_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = build_scheduler(opt, warmup_epochs=2, total_epochs=10, steps_per_epoch=1)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    assert lrs[0] < lrs[1] < lrs[2]      # warmup ramps up
    assert lrs[-1] < lrs[2]              # cosine decays after


def test_run_training_mrl_smoke(tmp_path):
    cfg = load_config("configs/default.yaml", overrides={
        "schedule.epochs": 1, "schedule.warmup_epochs": 0,
        "amp.enabled": False, "amp.compile": False,
        "run.output_root": str(tmp_path),
    })
    train_loader, val_loader = _tiny_loaders()
    result = run_training(
        cfg=cfg, model_kind="mrl", run_name="smoke",
        train_loader=train_loader, val_loader=val_loader,
        num_classes=3, device="cpu", nesting_dims=[8, 2048],
    )
    run_dir = tmp_path / "smoke"
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "plots" / "loss_curves.png").exists()
    assert result["epochs_completed"] == 1


def test_resume_tolerates_orig_mod_prefix(tmp_path):
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    from src.models import build_model
    # a checkpoint saved from a compiled model has _orig_mod. prefixed keys
    model = build_model("mrl", num_classes=3, nesting_dims=[8, 2048])
    compiled_like = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    consume_prefix_in_state_dict_if_present(compiled_like, "_orig_mod.")
    # after stripping, an eager model must accept the keys
    fresh = build_model("mrl", num_classes=3, nesting_dims=[8, 2048])
    fresh.load_state_dict(compiled_like, strict=True)
    assert set(compiled_like.keys()) == set(model.state_dict().keys())
