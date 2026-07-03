"""ONE augmentation pipeline for both models (protocol requirement).

Train: RandomResizedCrop(224, scale=(0.5, 1.0)) + HorizontalFlip +
light ColorJitter(0.2, 0.2, 0.2, hue=0). Deliberately modest; heavy
augmentation (mixup/cutmix/randaug) is excluded from the study as a confound.

Eval: Resize(shorter side 256) -> CenterCrop(224), identical for both models,
fully deterministic (no TTA, no randomness).

The ONLY per-model difference is the normalization constants, taken from each
model's own timm pretrained config — a declared, controlled difference: each
backbone must see inputs in the statistics it was pre-trained with, otherwise
we would be handicapping one model at the input layer.
"""
from functools import lru_cache

import timm
from torchvision import transforms as T


@lru_cache(maxsize=None)
def norm_constants(timm_name: str):
    try:
        cfg = timm.get_pretrained_cfg(timm_name)
        return tuple(cfg.mean), tuple(cfg.std)
    except Exception:
        # fallback: resolve through an uninitialised model instance
        from timm.data import resolve_data_config
        cfg = resolve_data_config({}, model=timm.create_model(timm_name, pretrained=False))
        return tuple(cfg["mean"]), tuple(cfg["std"])


def train_tfms(timm_name: str, resolution: int = 224):
    mean, std = norm_constants(timm_name)
    return T.Compose([
        T.RandomResizedCrop(resolution, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_tfms(timm_name: str, resolution: int = 224):
    mean, std = norm_constants(timm_name)
    return T.Compose([
        T.Resize(int(resolution * 256 / 224)),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
