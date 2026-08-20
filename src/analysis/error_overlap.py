"""Do the two families fail on the SAME images? CPU-only, from prediction tables.

Accuracy says how often each model is right. It cannot say whether a CNN and a
ViT are two views of one problem or two different problems. This compares
per-image outcomes at matched (dataset, fraction, seed) -- the only fair
pairing, since both models then saw identical training data.

For each matched pair, on the frozen test set:

    both_correct / only_a / only_b / both_wrong    the 2x2 outcome table
    agreement            fraction of images where correctness matches
    kappa                Cohen's kappa on the correctness variable, i.e.
                         agreement corrected for what chance alone would give
                         two models of these accuracies. Near 0 means they are
                         effectively independent; near 1 means interchangeable.
    same_wrong_pred      among images BOTH get wrong, how often they emit the
                         SAME wrong label. This is the sharpest of the four:
                         shared wrong answers indicate a shared inductive bias
                         or genuinely ambiguous ground truth, whereas
                         independent wrong answers indicate different failure
                         modes -- and therefore that an ensemble would help.
    oracle_top1          accuracy of a hypothetical picker that is right when
                         EITHER model is. The headroom an ensemble could reach,
                         and the honest upper bound on "these models are
                         complementary".

Why kappa and not raw agreement: two models that are both 88% accurate agree
~79% of the time by chance alone. Raw agreement would look impressive and mean
nothing. kappa subtracts exactly that baseline.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """kappa on two boolean correctness vectors."""
    n = len(a)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()
    out_root = REPO_ROOT / args.out_root

    tables = {}
    for run_dir in sorted(p for p in (out_root / "runs").iterdir() if p.is_dir()):
        pq = run_dir / "predictions.parquet"
        if not pq.exists() or "sweep" in run_dir.name:
            continue
        parts = run_dir.name.split("_")
        key = (parts[1], int(parts[2][1:]), int(parts[3][1:]), parts[4])  # ds, frac, seed, regime
        tables.setdefault(key, {})[parts[0]] = pd.read_parquet(pq)

    rows = []
    for (ds, frac, seed, regime), by_model in sorted(tables.items()):
        for ma, mb in itertools.combinations(sorted(by_model), 2):
            da, db = by_model[ma], by_model[mb]
            if not da["filepath"].equals(db["filepath"]):
                # Both come from the frozen test CSV in order; if this ever trips,
                # the two tables are not row-aligned and the comparison is void.
                raise SystemExit(f"row misalignment between {ma} and {mb} at "
                                 f"{ds} f{frac} s{seed} {regime}")
            ca = da["correct"].to_numpy(dtype=bool)
            cb = db["correct"].to_numpy(dtype=bool)
            both_wrong = ~ca & ~cb
            same_wrong = (da["pred"].to_numpy()[both_wrong]
                          == db["pred"].to_numpy()[both_wrong])
            rows.append({
                "dataset": ds, "fraction": frac, "seed": seed, "regime": regime,
                "model_a": ma, "model_b": mb, "n": len(ca),
                "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                "both_correct": float((ca & cb).mean()),
                "only_a": float((ca & ~cb).mean()),
                "only_b": float((~ca & cb).mean()),
                "both_wrong": float(both_wrong.mean()),
                "agreement": float((ca == cb).mean()),
                "kappa": _cohens_kappa(ca, cb),
                "same_wrong_pred": float(same_wrong.mean()) if both_wrong.sum() else np.nan,
                "oracle_top1": float((ca | cb).mean()),
            })

    if not rows:
        raise SystemExit("no matched model pairs found -- need >=2 models evaluated "
                         "at the same (fraction, seed, regime)")
    tdir = out_root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["regime", "fraction", "seed"])
    df.to_csv(tdir / "error_overlap.csv", index=False)
    print(f"[overlap] {len(df)} matched pairs -> tables/error_overlap.csv\n")
    print(df[["regime", "fraction", "seed", "model_a", "model_b", "acc_a", "acc_b",
              "both_wrong", "kappa", "same_wrong_pred", "oracle_top1"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
