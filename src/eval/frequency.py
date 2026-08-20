"""FFT frequency-sensitivity probes: ideal low/high-pass sweeps + band-limited noise.

This is the mechanism half of the study. Corruption curves say *that* the two
families degrade differently; these curves are meant to say *why* -- by asking
which spatial-frequency content each model's accuracy actually depends on.

Two probes, both checkpoint-agnostic (they transform inputs, never weights):

  (a) ideal low-pass / high-pass sweep over cutoff radius
      "How much accuracy survives if the model may only see frequencies below
      (resp. above) radius r?"  -> accuracy-vs-cutoff curves.

  (b) band-limited fixed-energy noise sweep
      "Perturbation energy is held constant and slid from DC to Nyquist; where
      does it hurt most?" -> accuracy-vs-band curves, i.e. a 1-D Fourier
      sensitivity profile in the spirit of Yin et al. (NeurIPS 2019).

Both operate on the 224x224 RGB crop in [0, 1], before normalization, for the
reasons in src/data/transforms.py.

---------------------------------------------------------------------------
The math, because this has to be defensible in the viva
---------------------------------------------------------------------------
Let x be one HxW image channel and X = F{x} its 2-D DFT. torch.fft.fft2 returns
X with DC at index [0, 0] and frequencies wrapping at the Nyquist edge, which
makes "distance from DC" awkward. fftshift moves DC to the array centre
[n//2, n//2], after which the frequency index along each axis relative to DC is

    u_i = i - n//2,     u in [-n/2, n/2 - 1]        (n = 224 -> -112 .. 111)

and the radial frequency of bin (i, j), in units of DFT bins, is the Euclidean

    r_ij = sqrt(u_i^2 + u_j^2).

Radius is reported both in bins and normalized by the axis Nyquist n/2 = 112,
so r_norm = 1.0 is Nyquist along an axis. Because the DFT grid is square while
r is radial, the corners reach r_norm = sqrt(2) ~ 1.414 -- which is exactly why
the sweep's top cutoff is 159 bins rather than 112: the outermost populated
bin sits at the corner, r = sqrt(112^2 + 112^2) = 158.39, so 159 is the first
integer radius that covers every bin and makes the low-pass mask the identity.
That gives the sweep a free correctness anchor (see _self_test below).

An IDEAL filter is a binary mask on that radius:

    low-pass(r_c):   M = 1 if r <= r_c else 0
    high-pass(r_c):  M = 1 if r >  r_c else 0

so M_lp(r_c) + M_hp(r_c) = 1 everywhere: the two masks are exact complements,
and filtering is linear, hence lowpass(x, r) + highpass(x, r) == x up to
floating point. That identity is asserted in the self-test and is the cheapest
possible proof the implementation is right.

Filtering is then multiplication in the frequency domain:

    y = Re{ F^-1 { ifftshift( fftshift(F{x}) * M ) } }

Re{} is taken because x is real, so its spectrum is Hermitian-symmetric and a
radially symmetric (hence Hermitian-symmetric) mask preserves that symmetry --
the inverse transform is real up to ~1e-7 numerical residue.

Two honest caveats, both stated in the report rather than hidden:
  - Ideal (brick-wall) filters ring: a sharp cutoff in frequency is a sinc in
    space, so low-pass images show Gibbs halos. This is deliberate and standard
    for this analysis (Park & Kim 2022 use the same construction); a Butterworth
    or Gaussian rolloff would trade ringing for an unclear cutoff, and the
    cutoff is the independent variable of the whole experiment.
  - Clamping back to [0, 1] after filtering is a mild nonlinearity, but the
    alternative -- feeding out-of-range pixels -- would be a different and
    less physical kind of distribution shift.

For (b), noise must be band-limited AND of fixed energy so that band index is
the only thing varying across the sweep. White Gaussian noise is generated in
image space, transformed, masked to an annulus r_lo <= r < r_hi, transformed
back, and then RESCALED so its spatial RMS equals a fixed target. Rescaling
after masking is the crucial step: an annulus near DC contains far fewer bins
than one near Nyquist, so without renormalization the "high frequency" bands
would simply carry more energy and the curve would measure bin count instead of
model sensitivity.
"""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.datasets import REPO_ROOT

# Cutoff radii in DFT bins for the ideal-filter sweep. 158 > 112*sqrt(2) so the
# low-pass mask there covers every populated bin => identity (correctness anchor).
# Corner radius is sqrt(112^2 + 112^2) = 158.39, so 159 is the first integer past it.
CUTOFFS_BINS = (4, 8, 12, 16, 24, 32, 48, 72, 112, 159)

# Annuli [r_lo, r_hi) in DFT bins for the band-limited noise sweep: contiguous,
# covering DC through the corner, widening with radius because the interesting
# structure is concentrated at low frequency.
NOISE_BANDS = ((0, 8), (8, 16), (16, 32), (32, 56), (56, 88), (88, 159))

# Fixed spatial RMS of the injected noise, in [0, 1] pixel units. Held constant
# across every band -- that constancy is what makes the sweep interpretable.
# 0.10 sits just above ImageNet-C gaussian_noise severity 1 (0.08): enough to
# separate the models, not enough to floor both at chance.
NOISE_RMS = 0.10

NYQUIST_BINS = 112.0  # n/2 for n = 224; used to report normalized radius


@lru_cache(maxsize=None)
def radial_bin_grid(n: int) -> torch.Tensor:
    """Radial frequency of every bin, in DFT bins, in fftshifted layout."""
    u = torch.arange(n, dtype=torch.float32) - n // 2
    return torch.sqrt(u[:, None] ** 2 + u[None, :] ** 2)


@lru_cache(maxsize=None)
def ideal_mask(n: int, cutoff_bins: float, mode: str) -> torch.Tensor:
    """Binary fftshifted mask. mode='low' keeps r <= cutoff, 'high' keeps r > cutoff."""
    r = radial_bin_grid(n)
    if mode == "low":
        return (r <= cutoff_bins).to(torch.float32)
    if mode == "high":
        return (r > cutoff_bins).to(torch.float32)
    raise ValueError(f"mode must be 'low' or 'high', got {mode!r}")


def fft_filter(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply an fftshifted frequency mask to a (C, H, W) image in [0, 1]."""
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    y = torch.fft.ifft2(torch.fft.ifftshift(X * mask, dim=(-2, -1))).real
    return y.clamp(0.0, 1.0)


def band_limited_noise(shape, r_lo: float, r_hi: float, rms: float,
                       seed: int) -> torch.Tensor:
    """(C, H, W) real noise whose spectral support is the annulus [r_lo, r_hi).

    Energy is normalized AFTER band-limiting so every band carries the same
    spatial RMS regardless of how many DFT bins its annulus contains.
    """
    c, h, w = shape
    g = torch.Generator().manual_seed(int(seed))
    white = torch.randn(c, h, w, generator=g)

    r = radial_bin_grid(h)
    mask = ((r >= r_lo) & (r < r_hi)).to(torch.float32)
    X = torch.fft.fftshift(torch.fft.fft2(white), dim=(-2, -1))
    noise = torch.fft.ifft2(torch.fft.ifftshift(X * mask, dim=(-2, -1))).real

    cur = noise.pow(2).mean().sqrt()
    if cur < 1e-12:  # empty annulus; degenerate but do not divide by zero
        return torch.zeros_like(noise)
    return noise * (rms / cur)


def add_band_noise(x: torch.Tensor, r_lo: float, r_hi: float, rms: float,
                   seed: int) -> torch.Tensor:
    return (x + band_limited_noise(tuple(x.shape), r_lo, r_hi, rms, seed)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# Dataset wrappers: same contract as CorruptedCsvImageDataset.
# --------------------------------------------------------------------------
class _FreqDatasetBase(Dataset):
    def __init__(self, csv_path, geometric_tfms, tensor_tfms):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.paths = df["filepath"].tolist()
        self.labels = df["label"].tolist()
        self.geometric = geometric_tfms
        self.tensor = tensor_tfms
        # ToTensor+Normalize expects a PIL image; the frequency operators need a
        # float tensor. Split the composed transform so we can insert between.
        self._to_tensor = tensor_tfms.transforms[0]
        self._normalize = tensor_tfms.transforms[1]

    def __len__(self):
        return len(self.paths)

    def _load_unit(self, i) -> torch.Tensor:
        """Image as a (C, H, W) float tensor in [0, 1], pre-normalization."""
        img = Image.open(REPO_ROOT / self.paths[i]).convert("RGB")
        return self._to_tensor(self.geometric(img))


class FilteredCsvImageDataset(_FreqDatasetBase):
    """Ideal low- or high-pass filtered images at one cutoff radius."""

    def __init__(self, csv_path, geometric_tfms, tensor_tfms, cutoff_bins, mode,
                 resolution=224):
        super().__init__(csv_path, geometric_tfms, tensor_tfms)
        self.mask = ideal_mask(resolution, float(cutoff_bins), mode)

    def __getitem__(self, i):
        return self._normalize(fft_filter(self._load_unit(i), self.mask)), self.labels[i]


class BandNoiseCsvImageDataset(_FreqDatasetBase):
    """Images plus fixed-energy noise confined to one frequency annulus."""

    def __init__(self, csv_path, geometric_tfms, tensor_tfms, r_lo, r_hi,
                 rms=NOISE_RMS):
        super().__init__(csv_path, geometric_tfms, tensor_tfms)
        self.r_lo, self.r_hi, self.rms = float(r_lo), float(r_hi), float(rms)

    def __getitem__(self, i):
        import zlib
        # Same crc32 seeding discipline as corruptions.py: the noise for a given
        # (band, image) is identical for both models and across reruns.
        seed = zlib.crc32(f"band|{self.r_lo}|{self.r_hi}|{i}".encode()) & 0xFFFFFFFF
        x = add_band_noise(self._load_unit(i), self.r_lo, self.r_hi, self.rms, seed)
        return self._normalize(x), self.labels[i]


# --------------------------------------------------------------------------
# Self-test: `python -m src.eval.frequency` -- cheap, no data or GPU needed.
# These are the properties the derivation above promises; if one fails, the
# curves this module produces are not measuring what the report claims.
# --------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    n = 224
    x = torch.rand(3, n, n)

    lp = ideal_mask(n, 159, "low")
    assert lp.min() == 1.0, "low-pass at r=159 must cover every populated bin"
    err = (fft_filter(x, lp) - x).abs().max().item()
    assert err < 1e-5, f"identity anchor failed: max abs err {err:.2e}"
    print(f"  identity at full radius      max|err| = {err:.2e}   OK")

    # complementarity: lowpass + highpass == original (linearity of the DFT).
    # Compare pre-clamp so the clamp in fft_filter does not mask a real error.
    for rc in (8, 32, 112):
        X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        parts = [torch.fft.ifft2(torch.fft.ifftshift(X * ideal_mask(n, rc, m),
                                                     dim=(-2, -1))).real
                 for m in ("low", "high")]
        e = (parts[0] + parts[1] - x).abs().max().item()
        assert e < 1e-4, f"complementarity failed at r={rc}: {e:.2e}"
        print(f"  lowpass+highpass == x  r={rc:<4d}  max|err| = {e:.2e}   OK")

    # band noise: fixed RMS regardless of annulus width, and really band-limited
    for r_lo, r_hi in NOISE_BANDS:
        nz = band_limited_noise((3, n, n), r_lo, r_hi, NOISE_RMS, seed=1234)
        got = nz.pow(2).mean().sqrt().item()
        assert abs(got - NOISE_RMS) < 1e-5, f"band [{r_lo},{r_hi}) rms {got:.5f}"
        N = torch.fft.fftshift(torch.fft.fft2(nz), dim=(-2, -1)).abs()
        outside = N[:, (radial_bin_grid(n) < r_lo) | (radial_bin_grid(n) >= r_hi)]
        assert outside.max().item() < 1e-3, "energy leaked outside the annulus"
        print(f"  band [{r_lo:3d},{r_hi:3d})  rms = {got:.5f}  leak = "
              f"{outside.max().item():.2e}   OK")

    print("\nfrequency.py self-test passed.")


if __name__ == "__main__":
    _self_test()
