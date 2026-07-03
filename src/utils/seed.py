"""Determinism helpers.

seed_everything(seed) is the FIRST call of every entry point (make_splits,
train, evaluate) so a rerun reproduces the same numbers: same batch order,
same augmentations, same init.

Why each line exists:
- cudnn.deterministic=True forces deterministic conv kernels (slightly slower,
  but "a rerun must reproduce the same numbers" is a protocol requirement).
- cudnn.benchmark=False stops cuDNN from autotuning kernels per input shape,
  which introduces run-to-run variation.
- DataLoader worker processes need their own seeding on top of this — see
  build_loader in src/data/datasets.py (generator + worker_init_fn).
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
