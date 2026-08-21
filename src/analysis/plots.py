"""Every quantitative figure in the report, generated from results/tables/.

House style (one place, applied to all figures)
----------------------------------------------
Colour encodes MODEL IDENTITY and nothing else -- it is categorical, assigned in
fixed order, and never reused for regime, fraction, or severity. Those are
carried by linestyle, facet, or x-position instead, so that adding a model never
repaints an existing one and a reader never has to ask what a colour means in
this particular panel.

The two hues are slots 1 and 2 of a pre-validated colourblind-safe categorical
palette (blue #2a78d6 / orange #eb6834): the standard blue-orange opposition,
which is the most robust two-series pair under all common CVD types. They are
used unchanged rather than re-derived, because a palette that has been validated
as a set should not be edited by eye.

Other rules, all from the same principle -- the ink should be the data:
  - seed variation is a translucent band (mean +/- 1 SD over the 3 seeds), never
    3 separate lines, so the eye reads one trend with an uncertainty, which is
    what the experiment actually measured;
  - grid and spines are recessive grey, text is ink-coloured (never the series
    colour), and a legend is always present once there are two series;
  - vector PDF for the report alongside PNG for the slides, from one call.

Figures are emitted only if the tables they need exist, so this is safe to run
mid-matrix -- which is when it is most useful.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_COLOR = {"resnet50": "#2a78d6", "vit_b16": "#eb6834"}
MODEL_LABEL = {"resnet50": "ResNet-50 (CNN)", "vit_b16": "ViT-B/16 (Transformer)"}
REGIME_STYLE = {"fullft": ("-", "o"), "linprobe": ("--", "s")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d9d6"
MANIFEST = []

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def _save(fig, out_dir, name, title, caption, source):
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    MANIFEST.append({"file": f"{name}.png", "title": title, "caption": caption,
                     "generated_from": source})
    print(f"  {name}.png / .pdf")


def _band(ax, g, xcol, ycol, color, ls, marker, label):
    """mean +/- 1 SD over seeds. One line, one band -- never one line per seed."""
    a = g.groupby(xcol)[ycol].agg(["mean", "std", "count"]).reset_index()
    a["std"] = a["std"].fillna(0.0)
    ax.plot(a[xcol], a["mean"], ls, marker=marker, color=color, label=label,
            linewidth=2.0, markersize=6, zorder=3)
    if (a["count"] > 1).any():
        ax.fill_between(a[xcol], a["mean"] - a["std"], a["mean"] + a["std"],
                        color=color, alpha=0.16, linewidth=0, zorder=2)
    return a


def fig_data_efficiency(master, out_dir):
    d = master.dropna(subset=["test_top1"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for regime in ("fullft", "linprobe"):
        for model in sorted(d["model_name"].unique()):
            g = d[(d["model_name"] == model) & (d["regime"] == regime)]
            if g.empty:
                continue
            ls, mk = REGIME_STYLE[regime]
            lab = f"{MODEL_LABEL[model]}" + ("" if regime == "fullft" else ", linear probe")
            _band(ax, g, "fraction", "test_top1", MODEL_COLOR[model], ls, mk, lab)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "training data per class (%, log scale)", "frozen-test top-1",
           "Data efficiency under a controlled transfer protocol")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_dir, "fig_data_efficiency", "Data-efficiency curves",
          "Frozen-test top-1 against the nested per-class training fraction. Line is "
          "the mean over 3 seeds, band is +/-1 SD. Solid = full fine-tuning, "
          "dashed = linear probe on frozen features.", "results/tables/master.csv")


def fig_corruption(master, long, out_dir):
    d = long[long["curve"].str.startswith("corruption:")] if not long.empty else pd.DataFrame()
    if d.empty:
        return
    fams = sorted(d["curve"].unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(4.6 * len(fams), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        g0 = d[(d["curve"] == fam) & (d["regime"] == "fullft") & (d["fraction"] == 100)]
        for model in sorted(g0["model_name"].unique()):
            _band(ax, g0[g0["model_name"] == model], "x", "top1",
                  MODEL_COLOR[model], "-", "o", MODEL_LABEL[model])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax, "ImageNet-C severity", "top-1" if ax is axes[0] else None,
               fam.split(":")[1].replace("_", " "))
        ax.set_xticks([1, 2, 3, 4, 5])
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Corruption robustness at f100 (full fine-tuning)",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "fig_corruption", "Corruption degradation curves",
          "Top-1 against corruption severity for each family, at 100% training data. "
          "Mean over 3 seeds, band +/-1 SD. Corruptions are applied deterministically "
          "on the fly, identically for both architectures.", "results/tables/curves_long.csv")


def fig_frequency(long, out_dir):
    d = long[long["curve"].isin(["frequency:lowpass", "frequency:highpass"])] \
        if not long.empty else pd.DataFrame()
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, curve, t in zip(axes, ["frequency:lowpass", "frequency:highpass"],
                            ["Ideal low-pass: accuracy vs cutoff",
                             "Ideal high-pass: accuracy vs cutoff"]):
        g0 = d[(d["curve"] == curve) & (d["regime"] == "fullft") & (d["fraction"] == 100)]
        for model in sorted(g0["model_name"].unique()):
            _band(ax, g0[g0["model_name"] == model], "x_norm", "top1",
                  MODEL_COLOR[model], "-", "o", MODEL_LABEL[model])
        ax.axvline(1.0, color=GRID, linestyle=":", linewidth=1.2, zorder=1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax, "cutoff radius (x Nyquist)", "top-1" if ax is axes[0] else None, t)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Frequency sensitivity — which spatial frequencies each family relies on",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "fig_frequency", "Frequency-response curves",
          "Accuracy when only frequencies below (left) or above (right) the cutoff "
          "radius survive an ideal filter. Dotted line marks Nyquist along an axis. "
          "f100, full fine-tuning, mean over 3 seeds.", "results/tables/curves_long.csv")


def fig_band_noise(long, out_dir):
    d = long[long["curve"] == "frequency:band_noise"] if not long.empty else pd.DataFrame()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    g0 = d[(d["regime"] == "fullft") & (d["fraction"] == 100)]
    for model in sorted(g0["model_name"].unique()):
        _band(ax, g0[g0["model_name"] == model], "x_norm", "top1",
              MODEL_COLOR[model], "-", "o", MODEL_LABEL[model])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "centre of noise band (x Nyquist)", "top-1",
           "Where does fixed-energy noise hurt most?")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_dir, "fig_band_noise", "Band-limited noise sensitivity",
          "Top-1 under additive noise confined to one frequency annulus and rescaled "
          "to a constant spatial RMS, so band position is the only variable. "
          "Lower means more vulnerable at that frequency.", "results/tables/curves_long.csv")


def fig_frequency_interaction(long, out_dir):
    """The novelty figure: does frequency reliance SHIFT with data fraction?"""
    d = long[(long["curve"] == "frequency:lowpass") & (long["regime"] == "fullft")] \
        if not long.empty else pd.DataFrame()
    if d.empty or d["fraction"].nunique() < 2:
        return
    fracs = sorted(d["fraction"].unique())
    fig, axes = plt.subplots(1, len(fracs), figsize=(3.4 * len(fracs), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fracs):
        g0 = d[d["fraction"] == f]
        for model in sorted(g0["model_name"].unique()):
            _band(ax, g0[g0["model_name"] == model], "x_norm", "top1",
                  MODEL_COLOR[model], "-", "o", MODEL_LABEL[model])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax, "cutoff (x Nyq)", "top-1" if ax is axes[0] else None, f"f{f}")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Frequency reliance x data fraction — does the low-data regime change "
                 "WHICH frequencies a family depends on?",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "fig_frequency_interaction", "Fraction x frequency interaction",
          "Low-pass accuracy-vs-cutoff curves, one panel per training fraction. The "
          "contribution of this study is the interaction: whether the gap between the "
          "two families at a given cutoff widens or narrows as data shrinks.",
          "results/tables/curves_long.csv")


def fig_calibration(out_dir, tables):
    cal_p, rel_p = tables / "calibration.csv", tables / "reliability.csv"
    if not cal_p.exists():
        return
    cal, rel = pd.read_csv(cal_p), pd.read_csv(rel_p)
    cal = cal[cal["regime"] == "fullft"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1.4, zorder=1)
    r = rel[(rel["regime"] == "fullft") & (rel["fraction"] == 100) & (rel["count"] > 0)]
    for model in sorted(r["model_name"].unique()):
        g = r[r["model_name"] == model].groupby("bin").agg(
            confidence=("confidence", "mean"), accuracy=("accuracy", "mean")).reset_index()
        ax.plot(g["confidence"], g["accuracy"], "-o", color=MODEL_COLOR[model],
                linewidth=2.0, markersize=6, label=MODEL_LABEL[model], zorder=3)
    _style(ax, "mean confidence in bin", "observed accuracy",
           "Reliability diagram (f100)")
    ax.text(0.04, 0.92, "above the line = under-confident", fontsize=8, color=INK2)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    for model in sorted(cal["model_name"].unique()):
        _band(ax, cal[cal["model_name"] == model], "fraction", "ece",
              MODEL_COLOR[model], "-", "o", MODEL_LABEL[model])
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style(ax, "training data per class (%)", "ECE (15 bins)",
           "Miscalibration vs data fraction")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig_calibration", "Calibration",
          "Left: reliability diagram at f100 -- points below the diagonal are "
          "over-confident. Right: expected calibration error against training "
          "fraction, mean over seeds.", "results/tables/calibration.csv")


def fig_error_overlap(out_dir, tables):
    p = tables / "error_overlap.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d["regime"] == "fullft"]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    agg = d.groupby("fraction")[["both_correct", "only_a", "only_b", "both_wrong"]].mean()
    bottom = np.zeros(len(agg))
    # Which model is "a" and which is "b" comes from the data, not from an
    # assumption about alphabetical order — a hard-coded pair of names silently
    # mislabels the chart the moment a third model or a rename appears.
    ma = MODEL_LABEL.get(d["model_a"].iloc[0], d["model_a"].iloc[0]).split(" (")[0]
    mb = MODEL_LABEL.get(d["model_b"].iloc[0], d["model_b"].iloc[0]).split(" (")[0]
    # Sequential shades of one neutral ramp: these are parts of a whole, not
    # model identities, so they must not borrow the categorical model hues.
    for col, c, lab in zip(["both_correct", "only_a", "only_b", "both_wrong"],
                           ["#c6dbef", "#6baed6", "#2171b5", "#08306b"],
                           ["both correct", f"only {ma}", f"only {mb}", "both wrong"]):
        # A thin surface-coloured edge separates adjacent segments, so a stacked
        # bar reads as parts rather than as one gradient.
        ax.bar(agg.index.astype(str), agg[col], bottom=bottom, color=c, label=lab,
               width=0.62, edgecolor="white", linewidth=1.5)
        bottom += agg[col].to_numpy()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "training data per class (%)", "share of test set", "Per-image outcome split")
    # Below the axes, not over the data: at low fractions the "both correct" band
    # reaches high enough that an inset legend lands on top of it.
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              columnspacing=1.0, handlelength=1.4)

    ax = axes[1]
    # These two series are measures, not model identities, so they are labelled
    # directly and carry no model colour semantics.
    for col, c, lab in (("kappa", "#2a78d6", "Cohen's kappa (correctness)"),
                        ("same_wrong_pred", "#eb6834", "same wrong label | both wrong")):
        a = d.groupby("fraction")[col].agg(["mean", "std"]).reset_index()
        a["std"] = a["std"].fillna(0)
        ax.plot(a["fraction"], a["mean"], "-o", color=c, linewidth=2.0,
                markersize=6, label=lab)
        ax.fill_between(a["fraction"], a["mean"] - a["std"], a["mean"] + a["std"],
                        color=c, alpha=0.16, linewidth=0)
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style(ax, "training data per class (%)", "agreement",
           "Are the two families making the same mistakes?")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "fig_error_overlap", "Error overlap",
          "Left: how the test set splits by which model was right. Right: kappa "
          "corrects agreement for chance; 'same wrong label' isolates shared "
          "inductive bias from merely shared difficulty.", "results/tables/error_overlap.csv")


def fig_deployment(out_dir, tables):
    p = tables / "deployment.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    metrics = [("params_m", "parameters (M)"), ("gmacs", "compute (GMACs)"),
               ("peak_vram_mb", "peak train VRAM (MB)"),
               ("train_imgs_per_sec", "training throughput (img/s)")]
    metrics = [(c, l) for c, l in metrics if c in d.columns and d[c].notna().any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (col, lab) in zip(axes, metrics):
        for i, (_, r) in enumerate(d.iterrows()):
            ax.bar(i, r[col], color=MODEL_COLOR.get(r["model_name"], "#888"), width=0.6)
            ax.text(i, r[col], f"{r[col]:,.0f}" if r[col] >= 100 else f"{r[col]:.2f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([MODEL_LABEL.get(m, m).split(" (")[0] for m in d["model_name"]],
                           fontsize=8)
        ax.margins(y=0.18)
        _style(ax, None, None, lab)
    fig.suptitle("Deployment cost", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "fig_deployment", "Deployment cost",
          "Static cost (parameters, GMACs at 224) and cost measured on the real "
          "workload (peak training VRAM, training throughput on an RTX 4060 8 GB).",
          "results/tables/deployment.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()
    tables = REPO_ROOT / args.out_root / "tables"
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(tables / "master.csv")
    lp = tables / "curves_long.csv"
    long = pd.read_csv(lp) if lp.exists() else pd.DataFrame()

    print("[plots] generating:")
    fig_data_efficiency(master, out_dir)
    fig_corruption(master, long, out_dir)
    fig_frequency(long, out_dir)
    fig_band_noise(long, out_dir)
    fig_frequency_interaction(long, out_dir)
    fig_calibration(out_dir, tables)
    fig_error_overlap(out_dir, tables)
    fig_deployment(out_dir, tables)

    if MANIFEST:
        man = out_dir / "MANIFEST.csv"
        pd.DataFrame(MANIFEST).to_csv(man, index=False)
        print(f"\n[plots] {len(MANIFEST)} figures -> {out_dir.relative_to(REPO_ROOT)}")
        print(f"[plots] captions logged to {man.relative_to(REPO_ROOT)}")
    else:
        print("  (nothing yet -- run the eval battery / more models first)")


if __name__ == "__main__":
    main()
