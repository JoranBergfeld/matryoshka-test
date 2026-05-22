from src.viz import plot_loss_curves, plot_accuracy_curves, plot_mrl_accuracy_by_dim


def test_plot_loss_curves_writes_png(tmp_path):
    history = [{"epoch": 0, "train_loss": 2.0, "val_loss": 1.8},
               {"epoch": 1, "train_loss": 1.5, "val_loss": 1.4}]
    out = tmp_path / "loss.png"
    plot_loss_curves(history, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_accuracy_curves_writes_png(tmp_path):
    history = [{"epoch": 0, "val_top1": 10.0, "val_top5": 30.0}]
    out = tmp_path / "acc.png"
    plot_accuracy_curves(history, str(out))
    assert out.exists()


def test_plot_mrl_accuracy_by_dim_writes_png(tmp_path):
    per_dim = {8: 12.0, 16: 20.0, 2048: 60.0}
    out = tmp_path / "bydim.png"
    plot_mrl_accuracy_by_dim(per_dim, epoch=3, out_path=str(out))
    assert out.exists()
