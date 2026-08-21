"""Walk results/runs/ -> results/tables/master.csv (+ curves_long.csv).

master.csv is the single source of truth for the report: every table and every
figure is generated from it, so no number in the write-up can exist that is not
traceable to a run folder on disk. Two files, same data, two shapes:

  master.csv       one row per run, wide. Scalars plus every corruption cell
                   (corr_<family>_s<sev>_top1) and every frequency point
                   (freq_lp_r<r>_top1, freq_hp_r<r>_top1, freq_band_<lo>-<hi>_top1).
                   Convenient for tables and for eyeballing.
  curves_long.csv  one row per (run, curve, x). Convenient for plotting and for
                   groupby-over-seeds, which is how every figure gets its band.

Derived summaries computed here rather than in plots.py, so the report and the
figures cannot disagree:
  corr_mean_top1        mean top-1 over all 15 corruption cells
  corr_rel_drop         (clean - corr_mean) / clean, i.e. relative robustness cost
  freq_lp_auc / hp_auc  area under the accuracy-vs-cutoff curve, normalized to
                        [0, 1] by trapezoid over normalized radius. A compact
                        scalar for "how much of this model's accuracy lives at
                        low (resp. high) frequency".
  freq_band_min_top1    worst band accuracy, and freq_band_argmin the band that
                        caused it -- the one-line summary of where a model is
                        most vulnerable in frequency.

Runs missing an eval battery are still emitted, with NaN in those columns; the
matrix is often partially evaluated while it is still running.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.runid import parse_run_id

REPO_ROOT = Path(__file__).resolve().parents[2]
NYQUIST = 112.0


def _flatten_battery(res, row, long_rows, base):
    """Battery JSON -> wide columns on `row` + long rows on `long_rows`."""
    if not res:
        return
    if "clean" in res:
        row["batt_clean_top1"] = res["clean"]["top1"]

    cells = []
    for fam, sevs in res.get("corruptions", {}).items():
        for sev, m in sevs.items():
            row[f"corr_{fam}_s{sev}_top1"] = m["top1"]
            cells.append(m["top1"])
            long_rows.append({**base, "curve": f"corruption:{fam}", "x": int(sev),
                              "x_norm": int(sev), **{k: m[k] for k in ("top1", "top5", "macro_f1")}})
    if cells:
        row["corr_mean_top1"] = float(np.mean(cells))
        clean = row.get("test_top1") or row.get("batt_clean_top1")
        if clean:
            row["corr_rel_drop"] = float((clean - np.mean(cells)) / clean)

    freq = res.get("frequency", {})
    for key, tag in (("lowpass", "lp"), ("highpass", "hp")):
        pts = freq.get(key, {})
        if not pts:
            continue
        radii = sorted(int(r) for r in pts)
        accs = [pts[str(r)]["top1"] for r in radii]
        for r, a in zip(radii, accs):
            row[f"freq_{tag}_r{r}_top1"] = a
            m = pts[str(r)]
            long_rows.append({**base, "curve": f"frequency:{key}", "x": r,
                              "x_norm": r / NYQUIST,
                              **{k: m[k] for k in ("top1", "top5", "macro_f1")}})
        # Normalized AUC over normalized radius: a scalar "how much accuracy
        # lives below/above the cutoff", comparable across models.
        xs = np.array(radii) / NYQUIST
        row[f"freq_{tag}_auc"] = float(np.trapz(accs, xs) / (xs[-1] - xs[0]))

    bands = freq.get("band_noise", {})
    if bands:
        keys = sorted(bands, key=lambda k: int(k.split("-")[0]))
        accs = [bands[k]["top1"] for k in keys]
        for k, a in zip(keys, accs):
            row[f"freq_band_{k}_top1"] = a
            lo, hi = (int(v) for v in k.split("-"))
            m = bands[k]
            long_rows.append({**base, "curve": "frequency:band_noise",
                              "x": (lo + hi) / 2, "x_norm": (lo + hi) / 2 / NYQUIST,
                              **{k2: m[k2] for k2 in ("top1", "top5", "macro_f1")}})
        row["freq_band_min_top1"] = float(min(accs))
        row["freq_band_argmin"] = keys[int(np.argmin(accs))]


def _extra_tables(runs_dir, tables, master):
    """Report-ready tables that master.csv cannot hold in one row per run.

    Everything the write-up needs must exist as a file on disk, so that no
    number in the report or the slides is ever typed by hand from a console.
    """
    import json as _json

    # per-class accuracy, long format -- master.csv can only carry summaries of
    # the 256-element per-class vector, and rare-class behaviour needs the vector.
    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        ce = run_dir / "clean_eval.json"
        if not ce.exists() or "sweep" in run_dir.name:
            continue
        c = _json.loads(ce.read_text())
        r = parse_run_id(run_dir.name)
        for cls, acc in enumerate(c["per_class_acc"]):
            rows.append({"run_id": run_dir.name, "model_name": r.model_name,
                         "dataset": r.dataset, "fraction": r.fraction, "seed": r.seed,
                         "regime": r.regime, "class_id": cls, "accuracy": acc})
    if rows:
        pd.DataFrame(rows).to_csv(tables / "per_class.csv", index=False)

    # declared LR sweeps -- excluded from master.csv (they are not matrix cells)
    # but they are the evidence for the per-model LR choice, so they get a table.
    sweep = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if "sweep" not in run_dir.name or not (run_dir / "metrics.json").exists():
            continue
        m = _json.loads((run_dir / "metrics.json").read_text())
        sweep.append({"model_name": m["model_name"], "dataset": m["dataset"],
                      "fraction": m["fraction"], "seed": m["seed"], "lr": m["lr"],
                      "epochs_ran": m["epochs_ran"],
                      "best_val_top1": m["best_val_top1"],
                      "best_val_top5": m["best_val_top5"],
                      "total_wall_min": m["total_wall_min"], "run_id": m["run_id"]})
    if sweep:
        sw = pd.DataFrame(sweep).sort_values(["model_name", "lr"])
        sw["selected"] = False
        for mdl, grp in sw.groupby("model_name"):
            sw.loc[grp["best_val_top1"].idxmax(), "selected"] = True
        sw.to_csv(tables / "lr_sweep.csv", index=False)

    # deployment cost: static cost (params/FLOPs, from provenance) joined to
    # cost measured on the real workload (peak VRAM, throughput, wall time).
    cost_path = tables / "model_cost.csv"
    if cost_path.exists() and not master.empty:
        meas = (master[master["regime"] == "fullft"]
                .groupby("model_name")
                .agg(peak_vram_mb=("peak_vram_mb", "max"),
                     train_imgs_per_sec=("train_imgs_per_sec", "mean"),
                     mean_epoch_time_s=("mean_epoch_time_s", "mean"),
                     total_gpu_hours=("total_wall_min", lambda s: s.sum() / 60.0),
                     runs=("run_id", "count")).reset_index())
        pd.read_csv(cost_path).merge(meas, on="model_name", how="left") \
          .to_csv(tables / "deployment.csv", index=False)

    # compute ledger: where the GPU-hours actually went, per block.
    if not master.empty and "total_wall_min" in master.columns:
        led = (master.groupby(["dataset", "regime", "model_name"])
               .agg(runs=("run_id", "count"),
                    gpu_hours=("total_wall_min", lambda s: round(s.sum() / 60.0, 2)),
                    mean_run_min=("total_wall_min", "mean"),
                    total_epochs=("epochs_ran", "sum")).reset_index())
        led.to_csv(tables / "compute_ledger.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--include-sweeps", action="store_true",
                    help="also emit LR-sweep runs (excluded by default: they are "
                         "8-epoch probes of the LR grid, not matrix cells)")
    args = ap.parse_args()

    out_root = REPO_ROOT / args.out_root
    runs_dir = out_root / "runs"
    rows, long_rows = [], []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        mpath = run_dir / "metrics.json"
        if not mpath.exists():
            continue
        if "sweep" in run_dir.name and not args.include_sweeps:
            continue
        m = json.loads(mpath.read_text())
        row = {k: m.get(k) for k in (
            "run_id", "model_name", "timm_name", "dataset", "num_classes",
            "fraction", "seed", "regime", "lr", "weight_decay", "optimizer",
            "effective_batch_size", "epochs_cap", "epochs_ran", "early_stopped",
            "best_epoch", "best_val_top1", "best_val_top5", "n_train", "n_val",
            "params_m", "peak_vram_mb", "mean_epoch_time_s", "train_imgs_per_sec",
            "total_wall_min", "torch", "timm")}

        ce = run_dir / "clean_eval.json"
        if ce.exists():
            c = json.loads(ce.read_text())
            row.update(test_top1=c["top1"], test_top5=c["top5"],
                       test_macro_f1=c["macro_f1"], n_test=c["n_test"])
            pc = np.array(c["per_class_acc"], dtype=float)
            # Rare-class behaviour: the mean over the worst decile of classes.
            # Caltech-256 is long-tailed, and a headline top-1 hides whether a
            # model is simply abandoning its rare classes.
            row["worst_decile_class_acc"] = float(np.sort(pc)[: max(1, len(pc) // 10)].mean())
            row["zero_acc_classes"] = int((pc == 0).sum())

        base = {k: row[k] for k in ("run_id", "model_name", "dataset", "fraction",
                                    "seed", "regime")}
        er = run_dir / "eval_results.json"
        _flatten_battery(json.loads(er.read_text()) if er.exists() else None,
                         row, long_rows, base)
        rows.append(row)

    if not rows:
        raise SystemExit("no completed runs found under results/runs/")

    df = pd.DataFrame(rows).sort_values(
        ["dataset", "regime", "model_name", "fraction", "seed"]).reset_index(drop=True)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "master.csv", index=False)
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df.to_csv(tables / "curves_long.csv", index=False)

    _extra_tables(runs_dir, tables, df)

    print(f"[aggregate] {len(df)} runs -> {(tables / 'master.csv').relative_to(REPO_ROOT)}")
    print(f"[aggregate] {len(long_df)} curve points -> curves_long.csv")

    # Coverage summary. Columns appear only once something has produced them, so
    # this must stay valid mid-matrix, when most runs have no battery yet.
    def _done(col):
        return (lambda s: int(s.notna().sum())) if col in df.columns else (lambda s: 0)
    spec = {"runs": ("run_id", "count"),
            "evaluated": ("test_top1" if "test_top1" in df.columns else "run_id",
                          _done("test_top1")),
            "battery": ("corr_mean_top1" if "corr_mean_top1" in df.columns else "run_id",
                        _done("corr_mean_top1"))}
    print("\n" + df.groupby(["dataset", "regime", "model_name"]).agg(**spec).to_string())


if __name__ == "__main__":
    main()
