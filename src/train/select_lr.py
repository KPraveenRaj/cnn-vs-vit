"""Pick the winning LR from a completed declared sweep and write it into the model YAML.

The protocol says the per-model learning rate is "chosen by a declared 3-point
sweep". That claim is only defensible if the choice is mechanical and the
evidence is on disk, so selection happens here rather than by a human reading a
console and typing a number into a config.

Selection rule: highest best_val_top1 across the sweep runs. Val, never test --
the test split is frozen and must never influence a hyperparameter.

The YAML is rewritten with a provenance comment recording the full grid, the
winner, and the date, matching the format already used for ResNet-50 so both
models document their choice identically.
"""
import argparse
import json
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-yaml", required=True)
    ap.add_argument("--pattern", required=True,
                    help="glob over results/runs, e.g. 'vit_b16_*sweep-lr*'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = sorted((REPO_ROOT / "results" / "runs").glob(args.pattern))
    results = []
    for r in runs:
        m = r / "metrics.json"
        if m.exists():
            d = json.loads(m.read_text())
            results.append((d["lr"], d["best_val_top1"], d["epochs_ran"], r.name))
    if len(results) < 2:
        raise SystemExit(f"need >=2 completed sweep runs matching {args.pattern!r}, "
                         f"found {len(results)}")
    results.sort()
    best_lr, best_v1, _, best_run = max(results, key=lambda t: t[1])

    grid = "   ".join(f"{lr:g} -> {v:.4f}" + (" (best)" if lr == best_lr else "")
                      for lr, v, _, _ in results)
    epochs = results[0][2]
    print(f"[select-lr] grid: {grid}")
    print(f"[select-lr] winner: lr={best_lr:g} (val top-1 {best_v1:.4f}, {best_run})")

    idx = results.index(next(r for r in results if r[0] == best_lr))
    interior = 0 < idx < len(results) - 1
    if not interior:
        print("[select-lr] WARNING: winner is at a grid EDGE — the true optimum may "
              "lie outside the swept range, i.e. this model may be under-tuned.")

    path = REPO_ROOT / args.model_yaml
    text = path.read_text()
    # The comment is regenerated from the runs on disk, so it can never disagree
    # with them — including the point count, which changes if the grid had to be
    # extended, and whether the optimum ended up bracketed.
    n = len(results)
    if interior:
        verdict = ("#   The winner is an INTERIOR grid point, so the optimum is bracketed on\n"
                   "#   both sides rather than sitting at the edge of the search.\n")
    else:
        verdict = ("#   WARNING: the winner is at a grid EDGE, so the optimum is NOT bracketed\n"
                   "#   and this model may be under-tuned. Extend the grid in that direction\n"
                   "#   before trusting any cross-model comparison built on this LR.\n")
    extended = ("#   The grid was EXTENDED beyond the original 3 declared points because the\n"
                "#   first winner landed on an edge — an under-tuned model would otherwise\n"
                "#   manufacture a result out of a tuning artefact.\n") if n > 3 else ""
    comment = (f"lr: {best_lr:g}\n"
               f"# ^ Set by the declared {n}-point LR sweep ({time.strftime('%Y-%m-%d')}; "
               f"f100, seed 0, {epochs} epochs,\n"
               f"#   AdamW, effective batch 64) — selection is by best val top-1, never test:\n"
               f"#     {grid}\n"
               f"{extended}{verdict}"
               f"#   Full records: results/runs/{args.pattern}/ and "
               f"results/tables/lr_sweep.csv\n")
    new = re.sub(r"^lr:.*?(?=^\w|\Z)", comment, text, count=1, flags=re.M | re.S)
    if args.dry_run:
        print("\n--- would write ---\n" + new)
        return
    path.write_text(new)
    print(f"[select-lr] wrote lr={best_lr:g} into {args.model_yaml}")


if __name__ == "__main__":
    main()
