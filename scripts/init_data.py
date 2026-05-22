"""Detect dataset, download via available credentials, or print setup help."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SETUP_HELP = """\
No ImageNet credentials found. Configure ONE of:

Kaggle:
  pip/uv add 'kaggle'; place kaggle.json in ~/.kaggle/ OR set KAGGLE_USERNAME + KAGGLE_KEY
  Accept rules at: https://www.kaggle.com/c/imagenet-object-localization-challenge

Hugging Face:
  set HF_TOKEN (or run `huggingface-cli login`); accept license for timm/imagenet-1k-wds

Then set IMAGENET_PATH to the target directory and re-run:
  uv run python scripts/init_data.py
"""


def dataset_present(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "val"))


def _kaggle_json_exists() -> bool:
    return os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))


def _hf_token_exists() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    return os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))


def detect_credentials() -> list[str]:
    creds = []
    if (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or _kaggle_json_exists():
        creds.append("kaggle")
    if os.environ.get("HF_TOKEN") or _hf_token_exists():
        creds.append("huggingface")
    return creds


def download_kaggle(dest: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    print(f"Downloading ImageNet via Kaggle into {dest} (large, slow)...")
    api.competition_download_files("imagenet-object-localization-challenge", path=dest)
    print("Download complete. Unzip and arrange into train/ and val/ ImageFolder layout.")


def download_hf(dest: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"Downloading timm/imagenet-1k-wds into {dest} (large, slow)...")
    snapshot_download(repo_id="timm/imagenet-1k-wds", repo_type="dataset", local_dir=dest)
    print("Download complete. Convert WebDataset shards to ImageFolder layout as needed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Initialize the ImageNet dataset.")
    parser.add_argument("--path", default=os.environ.get("IMAGENET_PATH"))
    args = parser.parse_args(argv)

    if not args.path:
        print("Set IMAGENET_PATH or pass --path.", file=sys.stderr)
        print(SETUP_HELP, file=sys.stderr); return 2
    if dataset_present(args.path):
        print(f"Dataset already present at {args.path}. Skipping."); return 0

    creds = detect_credentials()
    if not creds:
        print(SETUP_HELP, file=sys.stderr); return 2

    os.makedirs(args.path, exist_ok=True)
    if "kaggle" in creds:
        download_kaggle(args.path)
    else:
        download_hf(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
