"""Does Food-101 replicate the Caltech-256 findings? One row per claim.

Food-101 is the confirmation dataset. Its job is not to produce new headline
numbers but to answer a single question about each Caltech finding: does it
point the same way on a second, larger, finer-grained dataset?

Two things this deliberately does NOT do:

  - It does not test significance. Food-101 runs ONE seed by design, so there is
    no spread to test against. Direction and magnitude are reported; the n=1
    caveat is attached to every row and must survive into the report.
  - It does not treat disagreement as failure. A finding that holds on
    Caltech-256 and not on Food-101 is a finding about dataset dependence, not
    evidence that the Caltech measurement was wrong. The Caltech result stands
    on its own internal validity (frozen split, three seeds, one protocol);
    replication speaks to generality, which is a different claim.

Comparisons are matched: Food-101 only has f25 and f100, so the Caltech side is
read at f25 and f100 too — including the spectral-profile shift, which is
computed f25 -> f100 on BOTH datasets rather than reusing the f10 -> f100
number from the headline figure.

Output: results/tables/replication.csv, plus a readable summary on stdout.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LO, HI = 25, 100          # the two fractions Food-101 has


def _mean_top1(master, dataset, model, frac, regime="fullft"):
    d = master[(master.dataset == dataset) & (master.model_name == model)
               & (master.fraction == frac) & (master.regime == regime)]
    d = d[d.test_top1.notna()]
    return (float(d.test_top1.mean()), len(d)) if len(d) else (np.nan, 0)


def _mean_col(master, dataset, model, frac, col):
    if col not in master.columns:
        return np.nan
    d = master[(master.dataset == dataset) & (master.model_name == model)
               & (master.fraction == frac) & (master.regime == "fullft")]
    d = d[d[col].notna()]
    return float(d[col].mean()) if len(d) else np.nan


def _band_profile(dataset, model, frac):
    """Mean relative retention per band over available seeds, or None."""
    rows, ks = [], None
    for seed in (0, 1, 2):
        p = (REPO_ROOT / "results" / "runs"
             / f"{model}_{dataset}_f{frac}_s{seed}_fullft" / "eval_results.json")
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        b = r.get("frequency", {}).get("band_noise", {})
        if not b or "clean" not in r:
            continue
        ks = sorted(b, key=lambda x: int(x.split("-")[0]))
        rows.append([b[k]["top1"] / r["clean"]["top1"] for k in ks])
    return (np.array(rows).mean(0), ks) if rows else (None, None)


def _shift(dataset, model):
    lo, ks = _band_profile(dataset, model, LO)
    hi, _ = _band_profile(dataset, model, HI)
    if lo is None or hi is None:
        return np.nan
    return float(np.abs(hi - lo).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()
    tables = REPO_ROOT / args.out_root / "tables"
    m = pd.read_csv(tables / "master.csv")

    claims = []

    def add(claim, cal, food, agrees, note=""):
        claims.append({"claim": claim, "caltech256": cal, "food101": food,
                       "agrees": agrees, "note": note})

    # 1-2. accuracy ordering at each shared fraction ------------------------
    for fr in (LO, HI):
        gaps = {}
        for ds in ("caltech256", "food101"):
            r, nr = _mean_top1(m, ds, "resnet50", fr)
            v, nv = _mean_top1(m, ds, "vit_b16", fr)
            gaps[ds] = (v - r) * 100 if (nr and nv) else np.nan
        c, f_ = gaps["caltech256"], gaps["food101"]
        add(f"ViT-B/16 beats ResNet-50 at {fr}% data",
            f"{c:+.2f} pp" if c == c else "—",
            f"{f_:+.2f} pp" if f_ == f_ else "pending",
            _same_sign(c, f_))

    # 3. does the gap narrow as data grows? ---------------------------------
    trend = {}
    for ds in ("caltech256", "food101"):
        g = []
        for fr in (LO, HI):
            r, nr = _mean_top1(m, ds, "resnet50", fr)
            v, nv = _mean_top1(m, ds, "vit_b16", fr)
            g.append((v - r) * 100 if (nr and nv) else np.nan)
        trend[ds] = g[1] - g[0]
    c, f_ = trend["caltech256"], trend["food101"]
    add(f"Gap narrows from {LO}% to {HI}% data",
        f"{c:+.2f} pp change" if c == c else "—",
        f"{f_:+.2f} pp change" if f_ == f_ else "pending",
        _same_sign(c, f_))

    # 4. corruption robustness ----------------------------------------------
    rel = {}
    for ds in ("caltech256", "food101"):
        r = _mean_col(m, ds, "resnet50", HI, "corr_rel_drop")
        v = _mean_col(m, ds, "vit_b16", HI, "corr_rel_drop")
        rel[ds] = (r, v)
    c = rel["caltech256"][0] - rel["caltech256"][1]
    f_ = rel["food101"][0] - rel["food101"][1]
    add("ViT-B/16 degrades less under corruption",
        f"{rel['caltech256'][0]*100:.1f}% vs {rel['caltech256'][1]*100:.1f}% drop"
        if c == c else "—",
        f"{rel['food101'][0]*100:.1f}% vs {rel['food101'][1]*100:.1f}% drop"
        if f_ == f_ else "pending",
        _same_sign(c, f_))

    # 5. low-pass AUC --------------------------------------------------------
    auc = {}
    for ds in ("caltech256", "food101"):
        auc[ds] = (_mean_col(m, ds, "resnet50", HI, "freq_lp_auc"),
                   _mean_col(m, ds, "vit_b16", HI, "freq_lp_auc"))
    c = auc["caltech256"][1] - auc["caltech256"][0]
    f_ = auc["food101"][1] - auc["food101"][0]
    add("ViT-B/16 retains more accuracy under low-pass filtering",
        f"{auc['caltech256'][1]:.3f} vs {auc['caltech256'][0]:.3f}" if c == c else "—",
        f"{auc['food101'][1]:.3f} vs {auc['food101'][0]:.3f}" if f_ == f_ else "pending",
        _same_sign(c, f_))

    # 6. THE CONTRIBUTION: whose spectral profile moves more with data? ------
    sh = {ds: (_shift(ds, "resnet50"), _shift(ds, "vit_b16"))
          for ds in ("caltech256", "food101")}
    c = sh["caltech256"][0] - sh["caltech256"][1]
    f_ = sh["food101"][0] - sh["food101"][1]
    add(f"ResNet's spectral profile shifts MORE than ViT's ({LO}%->{HI}%)",
        f"{sh['caltech256'][0]:.3f} vs {sh['caltech256'][1]:.3f}" if c == c else "—",
        f"{sh['food101'][0]:.3f} vs {sh['food101'][1]:.3f}" if f_ == f_ else "pending",
        _same_sign(c, f_), "THE CONTRIBUTION")

    df = pd.DataFrame(claims)
    df.to_csv(tables / "replication.csv", index=False)

    print("  Does Food-101 replicate the Caltech-256 findings?")
    print("  (Food-101 is ONE seed by design — direction only, no significance test)\n")
    w = max(len(c["claim"]) for c in claims)
    print(f"  {'claim'.ljust(w)}  {'Caltech-256':<26} {'Food-101':<26} verdict")
    print("  " + "-" * (w + 62))
    for c in claims:
        v = {True: "REPLICATES", False: "DIFFERS", None: "pending"}[c["agrees"]]
        star = "  <- " + c["note"] if c["note"] else ""
        print(f"  {c['claim'].ljust(w)}  {str(c['caltech256']):<26} "
              f"{str(c['food101']):<26} {v}{star}")
    done = [c for c in claims if c["agrees"] is not None]
    if done:
        n = sum(1 for c in done if c["agrees"])
        print(f"\n  {n} of {len(done)} testable claims replicate.")
    print(f"\n  -> {(tables / 'replication.csv').relative_to(REPO_ROOT)}")


def _same_sign(a, b):
    """None if either side is missing; else whether they point the same way."""
    if a != a or b != b:
        return None
    return bool((a > 0) == (b > 0))


if __name__ == "__main__":
    main()
