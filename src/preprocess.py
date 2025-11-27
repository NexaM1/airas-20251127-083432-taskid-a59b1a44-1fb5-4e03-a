"""Dataset loading + augmentation pipeline.
All external files are cached inside `.cache/` to satisfy the specification.
Supports ImageNet-64 via HuggingFace."""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

os.environ.setdefault("HF_HOME", ".cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", ".cache/huggingface/datasets")


# ---------------------------------------------------------------------------
# Transformations -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _transform_pipeline(cfg):
    aug = getattr(cfg.dataset, "augmentations", {}) or {}
    tfms = []
    if aug.get("random_crop", False):
        tfms.append(transforms.RandomCrop(cfg.dataset.resolution, padding=4))
    if aug.get("horizontal_flip", False):
        tfms.append(transforms.RandomHorizontalFlip())
    tfms += [
        transforms.Resize(cfg.dataset.resolution, antialias=True),
        transforms.ToTensor(),
    ]
    return transforms.Compose(tfms)


# ---------------------------------------------------------------------------
# Dataset wrappers ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _load_imagenet64(cfg, split: str):
    hf_path = cfg.dataset.source.replace("hf://", "")

    # Try to load the requested split, fallback to train if not available
    try:
        split_name = cfg.dataset.splits[split]
        ds = load_dataset(
            hf_path,
            split=split_name,
            streaming=True,
            cache_dir=".cache/",
        )
    except ValueError:
        # If the requested split doesn't exist, use train split for both
        print(f"Warning: Split '{cfg.dataset.splits[split]}' not available, using 'train' split instead")
        ds = load_dataset(
            hf_path,
            split="train",
            streaming=True,
            cache_dir=".cache/",
        )

    tfm = _transform_pipeline(cfg)

    class _Wrap(torch.utils.data.IterableDataset):
        def __iter__(self):
            for item in ds:
                yield tfm(item["image"])

    return _Wrap()


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, *, for_optuna: bool = False) -> Tuple[DataLoader, DataLoader]:
    if cfg.dataset.name.lower() != "imagenet64":
        raise ValueError("Only ImageNet64 is implemented in this codebase.")

    train_ds = _load_imagenet64(cfg, "train")
    val_ds = _load_imagenet64(cfg, "val")

    bs = int(cfg.training.batch_size)
    if cfg.mode == "trial" or for_optuna:
        bs = min(8, bs)

    loader_kwargs = dict(batch_size=bs, num_workers=int(cfg.training.num_workers), pin_memory=True)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    return train_loader, val_loader