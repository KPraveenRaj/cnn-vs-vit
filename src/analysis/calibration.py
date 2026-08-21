"""Calibration from saved prediction tables. CPU-only, no checkpoints needed.

A model can be more accurate and still be worse to deploy if its confidence
lies. This reads results/runs/<run_id>/predictions.parquet -- which evaluate.py
writes for every checkpoint -- and asks whether stated confidence matches
observed accuracy.

Expected Calibration Error, the committed metric:

    ECE = sum_b (n_b / N) * | acc(b) - conf(b) |

over B equal-width bins of the top-1 confidence. It is the average gap between
"how sure the model said it was" and "how often it was right", weighted by how
many predictions land in each bin. MCE is the same gap at its worst bin.

Equal-WIDTH bins (not equal-mass) are the standard choice (Guo et al., 2017) and
are what makes ECE comparable to published numbers; the cost is that highly
accurate models pile most mass into the top bin, so bin counts are reported
alongside so a reader can see that.

Sign convention: overconfidence_gap = mean(confidence) - accuracy. Positive
means the model overstates itself, which is the usual failure direction for
networks trained with cross-entropy, and the direction that matters when a
downstream system thresholds on confidence.

Outputs
  results/tables/calibration.csv    one row per run
  results/tables/reliability.csv    per-bin data, long format, for the diagrams
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.runid import parse_run_id

REPO_ROOT = Path(__file__).resolve().parents[2]


def calibration_stats(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so confidence exactly 1.0 lands in the top bin, not out of range
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows, ece, mce, n = [], 0.0, 0.0, len(conf)
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin": b, "bin_lo": edges[b], "bin_hi": edges[b + 1],
                         "count": 0, "accuracy": np.nan, "confidence": np.nan, "gap": np.nan})
            continue
        acc, cf = float(correct[m].mean()), float(conf[m].mean())
        gap = abs(acc - cf)
        ece += cnt / n * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "bin_lo": edges[b], "bin_hi": edges[b + 1],
                     "count": cnt, "accuracy": acc, "confidence": cf, "gap": gap})
    return ece, mce, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    out_root = REPO_ROOT / args.out_root
    summary, rel = [], []
    for run_dir in sorted(p for p in (out_root / "runs").iterdir() if p.is_dir()):
        pq = run_dir / "predictions.parquet"
        if not pq.exists() or "sweep" in run_dir.name:
            continue
        df = pd.read_parquet(pq)
        conf = df["confidence"].to_numpy(dtype=float)
        correct = df["correct"].to_numpy(dtype=bool)
        ece, mce, rows = calibration_stats(conf, correct, args.n_bins)

        r = parse_run_id(run_dir.name)
        meta = {"run_id": run_dir.name, "model_name": r.model_name,
                "dataset": r.dataset, "fraction": r.fraction,
                "seed": r.seed, "regime": r.regime}
        summary.append({**meta, "n": len(df), "accuracy": float(correct.mean()),
                        "mean_confidence": float(conf.mean()),
                        "overconfidence_gap": float(conf.mean() - correct.mean()),
                        "ece": ece, "mce": mce, "n_bins": args.n_bins})
        rel.extend({**meta, **r} for r in rows)

    if not summary:
        raise SystemExit("no predictions.parquet found -- run src.eval.evaluate first")
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(summary).sort_values(["regime", "model_name", "fraction", "seed"])
    s.to_csv(tables / "calibration.csv", index=False)
    pd.DataFrame(rel).to_csv(tables / "reliability.csv", index=False)
    print(f"[calibration] {len(s)} runs -> tables/calibration.csv, reliability.csv\n")
    print(s[["run_id", "accuracy", "mean_confidence", "overconfidence_gap", "ece", "mce"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
