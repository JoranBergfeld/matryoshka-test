"""ImageNet data: transforms, path resolution, deterministic quick subset, loaders."""
from __future__ import annotations

import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

_DOWNLOAD_HINT = (
    "Set IMAGENET_PATH to a directory with train/ and val/ ImageFolder layout.\n"
    "Get ImageNet-1K from Kaggle 'imagenet-object-localization-challenge' or "
    "Hugging Face 'timm/imagenet-1k-wds'. See scripts/init_data.py."
)


def resolve_imagenet_path(env_var: str) -> str:
    path = os.environ.get(env_var)
    if not path:
        raise RuntimeError(f"{env_var} is not set.\n{_DOWNLOAD_HINT}")
    if not (os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "val"))):
        raise RuntimeError(f"{env_var}={path} missing train/ or val/.\n{_DOWNLOAD_HINT}")
    return path


def build_transforms(train: bool, image_size: int, resize_size: int,
                     mean: list[float], std: list[float]) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def select_quick_classes(all_classes: list, num_classes: int, seed: int) -> list:
    """Deterministically pick a subset of class identifiers (sorted output)."""
    rng = random.Random(seed)
    chosen = rng.sample(list(all_classes), num_classes)
    return sorted(chosen)


class _RemappedSubset(torch.utils.data.Dataset):
    """Subset that remaps a sparse set of original labels to contiguous 0..K-1."""

    def __init__(self, base: ImageFolder, indices: list[int], label_map: dict[int, int]) -> None:
        self.base = base
        self.indices = indices
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]
        return x, self.label_map[y]


def _subset_to_classes(dataset: ImageFolder, keep_class_names: set[str]):
    keep_idx = sorted(dataset.class_to_idx[c] for c in keep_class_names)
    label_map = {orig: new for new, orig in enumerate(keep_idx)}
    keep_set = set(keep_idx)
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in keep_set]
    return _RemappedSubset(dataset, indices, label_map)


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, int]:
    """Build train/val loaders. Returns (train_loader, val_loader, num_classes).

    Honors cfg['data']['quick'] for the deterministic 100-class subset."""
    data_cfg = cfg["data"]
    root = resolve_imagenet_path(data_cfg["imagenet_path_env"])
    mean, std = data_cfg["mean"], data_cfg["std"]
    train_t = build_transforms(True, data_cfg["image_size"], data_cfg["resize_size"], mean, std)
    val_t = build_transforms(False, data_cfg["image_size"], data_cfg["resize_size"], mean, std)

    train_ds = ImageFolder(os.path.join(root, "train"), transform=train_t)
    val_ds = ImageFolder(os.path.join(root, "val"), transform=val_t)
    num_classes = data_cfg["num_classes"]

    if data_cfg["quick"]["enabled"]:
        keep = set(select_quick_classes(train_ds.classes,
                                        data_cfg["quick"]["num_classes"],
                                        cfg["run"]["seed"]))
        train_ds = _subset_to_classes(train_ds, keep)
        val_ds = _subset_to_classes(val_ds, keep)
        num_classes = data_cfg["quick"]["num_classes"]

    dl = cfg["dataloader"]
    num_workers = dl["num_workers"] if dl["num_workers"] is not None else min(8, os.cpu_count() or 1)
    common = dict(
        batch_size=dl["batch_size"],
        num_workers=num_workers,
        pin_memory=dl["pin_memory"],
        persistent_workers=dl["persistent_workers"] and num_workers > 0,
        prefetch_factor=dl["prefetch_factor"] if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, num_classes
