"""Generate every qualitative figure the report and decks need, reproducibly.

Rationale: the quantitative figures (plots.py) prove what happened; these show
what the experiment actually DID to the pixels. A reader who has never seen an
ideal high-pass image cannot judge a high-pass accuracy curve, and an examiner
asking "what does severity 5 actually look like?" deserves a picture rather than
a constant from a table.

Everything here is deterministic: one fixed demo image chosen by index from the
frozen test split, fixed seeds, fixed layout. Re-running overwrites byte-similar
files, so assets can be regenerated at any time and never drift from the code
that defines the experiment. Nothing is hand-made in an image editor.

Every asset is logged to MANIFEST.csv with a caption and the module that defines
its semantics, so the report can cite figures by filename and a reader can trace
each one back to the code that produced it.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.transforms import eval_geometric_tfms, train_tfms
from src.eval.corruptions import CORRUPTIONS, SEVERITIES, corrupt
from src.eval.frequency import (CUTOFFS_BINS, NOISE_BANDS, NOISE_RMS,
                                add_band_noise, fft_filter, ideal_mask,
                                radial_bin_grid)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_INDEX = 1500          # fixed row of the frozen test split; stable forever
ACCENT = "#1F4E79"
MANIFEST = []


def _log(fname, title, caption, source):
    MANIFEST.append({"file": fname, "title": title, "caption": caption,
                     "defined_by": source})


def _save(fig, out_dir, fname, title, caption, source):
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / fname.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    _log(fname, title, caption, source)
    print(f"  {fname}")


def _grid(images, titles, ncols, figsize, suptitle=None, row_labels=None):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=8, color="#262626")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if row_labels:
        for r, lab in enumerate(row_labels):
            axes[r * ncols].set_ylabel(lab)
            axes[r * ncols].axis("on")
            axes[r * ncols].set_xticks([])
            axes[r * ncols].set_yticks([])
            for s in axes[r * ncols].spines.values():
                s.set_visible(False)
            axes[r * ncols].set_ylabel(lab, fontsize=9, color=ACCENT,
                                       fontweight="bold", rotation=90, labelpad=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color=ACCENT, fontweight="bold")
    fig.tight_layout(h_pad=1.8)
    return fig


def _to_np(t):
    """(C,H,W) float tensor in [0,1] -> HxWx3 uint8-ish array for imshow."""
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_caltech256.yaml")
    ap.add_argument("--out", default="results/figures/assets")
    args = ap.parse_args()

    from src.utils.config import load_yaml
    cfg = load_yaml(REPO_ROOT / args.data)
    splits = REPO_ROOT / cfg["splits_dir"]
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(splits / "test.csv")
    geo = eval_geometric_tfms(224)
    demo_row = test.iloc[DEMO_INDEX]
    demo_pil = geo(Image.open(REPO_ROOT / demo_row["filepath"]).convert("RGB"))
    demo_t = torch.from_numpy(np.asarray(demo_pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    demo_name = demo_row["class_name"]
    print(f"[assets] demo image: {demo_row['filepath']} ({demo_name})\n")

    # 1. dataset samples ----------------------------------------------------
    rng = np.random.default_rng(0)
    pick = rng.choice(len(test), 16, replace=False)
    imgs = [geo(Image.open(REPO_ROOT / test.iloc[i]["filepath"]).convert("RGB")) for i in pick]
    titles = [test.iloc[i]["class_name"].split(".", 1)[-1][:18] for i in pick]
    _save(_grid(imgs, titles, 8, (16, 4.6),
                "Caltech-256 — frozen test split, 16 random samples"),
          out_dir, "dataset_samples.png", "Caltech-256 samples",
          "Sixteen images drawn at random (seed 0) from the frozen test split, "
          "after the deterministic evaluation transform Resize(256)+CenterCrop(224).",
          "src/data/transforms.py")

    # 2. class distribution -------------------------------------------------
    counts = test.groupby("class_name").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(range(len(counts)), counts.values, color=ACCENT, width=1.0)
    ax.set_xlabel("class rank"); ax.set_ylabel("test images")
    ax.set_title(f"Caltech-256 is long-tailed — {len(counts)} classes, "
                 f"{counts.max()} to {counts.min()} test images per class",
                 fontsize=11, color=ACCENT, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir, "class_distribution.png", "Class imbalance",
          f"Per-class test-set counts, ranked. The tail is why macro-F1 and "
          f"worst-decile class accuracy are reported alongside top-1.",
          "src/data/make_splits.py")

    # 3. augmentation -------------------------------------------------------
    torch.manual_seed(0)
    tf = train_tfms("resnet50.a1_in1k", 224)
    src = Image.open(REPO_ROOT / demo_row["filepath"]).convert("RGB")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    augs = [_to_np(tf(src) * std + mean) for _ in range(7)]
    _save(_grid([np.asarray(demo_pil)] + augs,
                ["original (eval transform)"] + [f"train aug #{i+1}" for i in range(7)],
                4, (11, 6),
                "One modest augmentation pipeline, applied identically to both models"),
          out_dir, "augmentation_samples.png", "Training augmentation",
          "RandomResizedCrop(224, scale 0.5-1.0) + horizontal flip + light colour "
          "jitter (0.2/0.2/0.2, hue 0). Deliberately modest: mixup/cutmix/RandAugment "
          "are excluded as confounds. De-normalised for display.",
          "src/data/transforms.py")

    # 4. corruption ladder --------------------------------------------------
    imgs, titles, rows = [], [], []
    for fam in CORRUPTIONS:
        rows.append(fam.replace("_", " "))
        imgs.append(np.asarray(demo_pil)); titles.append("clean")
        for s in SEVERITIES:
            imgs.append(np.asarray(corrupt(demo_pil, fam, s, DEMO_INDEX)))
            titles.append(f"severity {s}  ({CORRUPTIONS[fam][s-1]})")
    _save(_grid(imgs, titles, 6, (15, 9.0),
                "Corruption battery — ImageNet-C severities, applied on the fly",
                row_labels=rows),
          out_dir, "corruption_ladder.png", "Corruption severities",
          "Three corruption families at five ImageNet-C severities. Applied to the "
          "224x224 crop in [0,1] before normalisation, seeded per (image, severity) "
          "so both architectures score on byte-identical inputs. Never written to disk.",
          "src/eval/corruptions.py")

    # 5. low/high-pass ladders ---------------------------------------------
    for mode, tag in (("low", "lowpass"), ("high", "highpass")):
        imgs = [np.asarray(demo_pil)]
        titles = ["original"]
        for rc in CUTOFFS_BINS:
            imgs.append(_to_np(fft_filter(demo_t, ideal_mask(224, float(rc), mode))))
            titles.append(f"r = {rc} bins  ({rc/112:.2f}$\\times$Nyq)")
        word = "below" if mode == "low" else "above"
        _save(_grid(imgs, titles, 6, (15, 6.8),
                    f"Ideal {tag} sweep — only frequencies {word} the cutoff survive"),
              out_dir, f"frequency_{tag}_ladder.png", f"Ideal {tag} sweep",
              f"Ideal (brick-wall) {tag} filtering at each swept cutoff radius. "
              f"Radius is in DFT bins; 112 bins = Nyquist along an axis, and 159 "
              f"covers the corner so the low-pass mask there is the identity. "
              f"Ringing is the expected Gibbs artefact of a sharp spectral cutoff.",
              "src/eval/frequency.py")

    # 6. band-limited noise -------------------------------------------------
    imgs = [np.asarray(demo_pil)]; titles = ["original"]
    for lo, hi in NOISE_BANDS:
        imgs.append(_to_np(add_band_noise(demo_t, lo, hi, NOISE_RMS, seed=lo * 1000 + hi)))
        titles.append(f"band [{lo}, {hi}) bins")
    _save(_grid(imgs, titles, 4, (11, 7.2),
                f"Band-limited noise at fixed energy (RMS = {NOISE_RMS}) — "
                f"same energy, slid from DC to Nyquist"),
          out_dir, "band_noise_ladder.png", "Band-limited noise sweep",
          f"Gaussian noise confined to one frequency annulus and rescaled to a "
          f"fixed spatial RMS of {NOISE_RMS}. Holding energy constant is what makes "
          f"band index the only variable, so the resulting curve measures model "
          f"sensitivity rather than how many DFT bins an annulus happens to contain.",
          "src/eval/frequency.py")

    # 7. spectrum + masks ---------------------------------------------------
    spec = torch.fft.fftshift(torch.fft.fft2(demo_t.mean(0))).abs()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    axes[0].imshow(np.asarray(demo_pil)); axes[0].set_title("image", fontsize=9)
    axes[1].imshow(torch.log1p(spec).numpy(), cmap="magma")
    axes[1].set_title("log |FFT| (DC centred)", fontsize=9)
    for ax, (rc, mode) in zip(axes[2:], [(16, "low"), (16, "high"), (48, "low")]):
        ax.imshow(ideal_mask(224, float(rc), mode).numpy(), cmap="gray")
        ax.set_title(f"{mode}-pass mask, r={rc}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("How the frequency probes are built", fontsize=13,
                 color=ACCENT, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fft_construction.png", "FFT probe construction",
          "The demo image, its centred log-magnitude spectrum, and three of the "
          "binary radial masks the sweep multiplies it by. Low- and high-pass masks "
          "at the same radius are exact complements, which the module's self-test "
          "asserts (lowpass + highpass reconstructs the original to 4e-07).",
          "src/eval/frequency.py")

    # 8. nested fractions ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fracs, shades = [100, 50, 25, 10], ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    for f, c in zip(fracs, shades):
        n = len(pd.read_csv(splits / f"train_f{f}_s0.csv"))
        ax.barh([0], [n], color=c, height=0.55, label=f"f{f}  ({n:,} images)")
    ax.set_yticks([]); ax.set_xlabel("training images (seed 0)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    ax.set_title("Data fractions are per-class and NESTED: f10 $\\subset$ f25 "
                 "$\\subset$ f50 $\\subset$ f100", fontsize=11, color=ACCENT,
                 fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    _save(fig, out_dir, "fraction_nesting.png", "Nested data fractions",
          "Each smaller fraction is a strict per-class subset of the larger one, so "
          "the data-efficiency curve varies only quantity, never composition. Three "
          "independent seed nestings give the error bands.",
          "src/data/make_splits.py")

    pd.DataFrame(MANIFEST).to_csv(out_dir / "MANIFEST.csv", index=False)
    print(f"\n[assets] {len(MANIFEST)} assets (png + pdf) -> {out_dir.relative_to(REPO_ROOT)}")
    print(f"[assets] captions logged to {(out_dir / 'MANIFEST.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
