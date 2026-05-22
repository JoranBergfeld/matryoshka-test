import subprocess
import sys


def test_train_cli_help_lists_flags():
    out = subprocess.run([sys.executable, "scripts/train.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ["--model", "--run-name", "--epochs", "--lr", "--quick",
                 "--full", "--resume", "--seed", "--nesting-dims",
                 "--mrl-weighting", "--label-smoothing"]:
        assert flag in out.stdout
