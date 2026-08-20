"""Deterministic on-the-fly input corruptions: Gaussian noise, Gaussian blur, JPEG.

Three corruption families x 5 severities, applied to the 224x224 RGB crop in
[0, 1] before normalization (see src/data/transforms.py for why that insertion
point is the only defensible one). Corrupted images are NEVER written to disk:
they are regenerated on demand, which keeps the ~30k-image dataset from turning
into 15 copies of itself and makes the severity ladder a property of the code
rather than of a directory nobody can audit.

Severity constants follow the ImageNet-C convention (Hendrycks & Dietterich,
2019) so the numbers are comparable to published robustness work:

    gaussian_noise  sigma   0.08  0.12  0.18  0.26  0.38   (on [0, 1] pixels)
    gaussian_blur   sigma   1     2     3     4     6      (pixels)
    jpeg            quality 25    18    15    10    7      (libjpeg quality)

Determinism is the load-bearing property here
---------------------------------------------
The comparison is only controlled if ResNet-50 and ViT-B/16 are scored on the
SAME corrupted pixels. So the noise for a given (corruption, severity, image)
is drawn from a seed derived from those three things alone -- via crc32, not
Python's hash(), which is salted per process. Consequences:
  - both models see identical inputs, in the same order, always;
  - a rerun months later reproduces the numbers exactly;
  - severity 3 for image 700 is the same picture no matter which run produced
    it, so per-image error overlap stays meaningful under corruption too.
Blur and JPEG are deterministic by construction and ignore the seed.
"""
import io
import zlib

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from src.data.datasets import REPO_ROOT

SEVERITIES = (1, 2, 3, 4, 5)

CORRUPTIONS = {
    "gaussian_noise": (0.08, 0.12, 0.18, 0.26, 0.38),
    "gaussian_blur": (1.0, 2.0, 3.0, 4.0, 6.0),
    "jpeg": (25, 18, 15, 10, 7),
}


def _seed_for(corruption: str, severity: int, index: int) -> int:
    """Stable across processes and machines, unlike hash() on a str."""
    return zlib.crc32(f"{corruption}|{severity}|{index}".encode()) & 0xFFFFFFFF


def corrupt(img: Image.Image, corruption: str, severity: int, index: int) -> Image.Image:
    """Apply one corruption at one severity to a 224x224 RGB PIL image."""
    if corruption not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {corruption!r}; have {sorted(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    c = CORRUPTIONS[corruption][severity - 1]

    if corruption == "gaussian_noise":
        rng = np.random.default_rng(_seed_for(corruption, severity, index))
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.clip(x + rng.normal(0.0, c, x.shape).astype(np.float32), 0.0, 1.0)
        return Image.fromarray((x * 255.0).round().astype(np.uint8))

    if corruption == "gaussian_blur":
        x = np.asarray(img, dtype=np.float32) / 255.0
        # sigma applies to the two spatial axes only; channels stay independent.
        x = np.clip(gaussian_filter(x, sigma=(c, c, 0)), 0.0, 1.0)
        return Image.fromarray((x * 255.0).round().astype(np.uint8))

    # jpeg: round-trip through the real encoder so we get true block artifacts
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(c))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class CorruptedCsvImageDataset(Dataset):
    """Split CSV -> (corrupted, normalized tensor, label).

    Composes: load -> geometric eval transform (PIL) -> corruption (PIL)
    -> ToTensor + per-model Normalize. `index` is the row index in the CSV,
    which is frozen, so it is a stable image identity across models and runs.
    """

    def __init__(self, csv_path, geometric_tfms, tensor_tfms,
                 corruption: str, severity: int):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.paths = df["filepath"].tolist()
        self.labels = df["label"].tolist()
        self.geometric = geometric_tfms
        self.tensor = tensor_tfms
        self.corruption = corruption
        self.severity = severity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(REPO_ROOT / self.paths[i]).convert("RGB")
        img = self.geometric(img)
        img = corrupt(img, self.corruption, self.severity, i)
        return self.tensor(img), self.labels[i]
