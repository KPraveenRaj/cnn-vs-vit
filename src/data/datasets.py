"""Dataset = one committed split CSV -> (image tensor, label). Nothing more.

All experiment structure (which images, which fraction, which seed) was
decided once in make_splits.py and frozen in the committed CSVs; this class
just reads them. build_loader adds seeded shuffling so that batch order is
reproducible for a given seed (part of the determinism contract).
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# A handful of Caltech-256 jpgs are truncated; refusing to load them would
# effectively (and silently) edit the frozen splits, so tolerate truncation.
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[2]


class CsvImageDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.paths = df["filepath"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(REPO_ROOT / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_loader(csv_path, transform, batch_size, shuffle, seed, num_workers=8):
    ds = CsvImageDataset(csv_path, transform)
    gen = torch.Generator()
    gen.manual_seed(seed)

    def worker_init(worker_id):  # each worker gets its own derived seed
        s = seed * 1000 + worker_id
        np.random.seed(s)
        random.seed(s)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        worker_init_fn=worker_init,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
