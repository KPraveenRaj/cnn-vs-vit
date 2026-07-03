"""Enumerate and run the committed experiment matrix — sequential, resumable.

Resumability is inherited, not implemented here: train.py skips any run whose
metrics.json exists, and this driver skips evaluation when clean_eval.json
exists. So the matrix can be interrupted (crash, reboot, Ctrl-C) and simply
re-launched; completed work is never redone and never overwritten.

A run failure does NOT stop the matrix: it is recorded, the driver moves on,
and failures are listed at the end (exit code 1) so a later relaunch retries
only what's missing.

Examples
--------
ResNet-50 block, training + frozen-test eval after each run:
    python -m src.train.run_matrix --models resnet50 --with-eval
Both models (the full Caltech-256 fullft grid):
    python -m src.train.run_matrix --models resnet50,vit_b16 --with-eval
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.utils.config import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_id_of(model_name, dataset, frac, seed, regime):
    return f"{model_name}_{dataset}_f{frac}_s{seed}_{regime}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="resnet50", help="comma list: resnet50,vit_b16")
    ap.add_argument("--data", default="configs/data_caltech256.yaml")
    ap.add_argument("--fractions", default=None, help="comma list; default: data yaml")
    ap.add_argument("--seeds", default=None, help="comma list; default: data yaml")
    ap.add_argument("--regime", default="fullft")
    ap.add_argument("--with-eval", action="store_true", help="run frozen-test eval after each run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_cfg = load_yaml(REPO_ROOT / args.data)
    fractions = [int(f) for f in args.fractions.split(",")] if args.fractions else sorted(data_cfg["fractions"])
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else data_cfg["seeds"]
    models = args.models.split(",")

    jobs = [(m, f, s) for f in fractions for s in seeds for m in models]
    runs_dir = REPO_ROOT / "results" / "runs"
    failures, walls = [], []
    t_start = time.time()

    for k, (m, frac, seed) in enumerate(jobs, 1):
        model_yaml = f"configs/model_{m}.yaml"
        model_name = load_yaml(REPO_ROOT / model_yaml)["model_name"]
        rid = run_id_of(model_name, data_cfg["dataset"], frac, seed, args.regime)
        done_train = (runs_dir / rid / "metrics.json").exists()
        done_eval = (runs_dir / rid / "clean_eval.json").exists()

        eta = ""
        if walls:
            remaining = len(jobs) - k + 1
            eta = f"  (ETA ~{remaining * sum(walls) / len(walls) / 60:.0f} min at current pace)"
        print(f"[matrix {k}/{len(jobs)}] {rid}"
              f"{' [train done]' if done_train else ''}{' [eval done]' if done_eval else ''}{eta}",
              flush=True)
        if args.dry_run:
            continue

        t0 = time.time()
        if not done_train:
            r = subprocess.run([sys.executable, "-m", "src.train.train",
                                "--model", model_yaml, "--data", args.data,
                                "--fraction", str(frac), "--seed", str(seed),
                                "--regime", args.regime], cwd=REPO_ROOT)
            if r.returncode != 0:
                failures.append(f"{rid} (train)")
                continue
            walls.append(time.time() - t0)
        if args.with_eval and not done_eval:
            r = subprocess.run([sys.executable, "-m", "src.eval.evaluate", "--run-id", rid],
                               cwd=REPO_ROOT)
            if r.returncode != 0:
                failures.append(f"{rid} (eval)")

    total_h = (time.time() - t_start) / 3600
    print(f"[matrix] finished in {total_h:.2f} h; failures: {failures or 'none'}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
