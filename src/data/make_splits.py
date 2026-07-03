"""Build the frozen split manifests for a dataset. Run ONCE per dataset; the
resulting CSVs are committed to git and never regenerated.

Design (fixed by the study protocol):

- 70/10/20 train/val/test, stratified per class, drawn from one master
  permutation per class (MASTER_SEED below). THE TEST SET IS FROZEN FOREVER —
  every checkpoint ever trained is evaluated on exactly these files.

- Data fractions are per-class and NESTED within each seed:
  f10 ⊂ f25 ⊂ f50 ⊂ f100. Each seed draws its own independent per-class
  permutation of the train pool, so 3 seeds give 3 independent nestings.
  Nesting holds because ceil is monotone: for one fixed permutation, the
  first ceil(f·n) items for growing f are supersets of each other.

- f100 is seed-independent by construction (the whole train pool), but is
  written once per seed anyway so loader logic stays uniform.

Outputs under <splits_dir> (all committed):
  classes.csv            label,class_name  (sorted directory order)
  test.csv, val.csv      filepath,label,class_name
  train_f{F}_s{S}.csv    one per (fraction, seed)
  split_report.txt       per-class counts, skipped files, assertion summary

Hard assertions before anything is written:
  * class count matches the data config after exclusions
  * zero filepath overlap between train pool / val / test
  * nesting holds for every (class, seed)
  * every referenced file exists on disk
Non-image files (anything not .jpg/.jpeg/.png) are skipped and logged.
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.utils.seed import seed_everything

MASTER_SEED = 1234  # fixes the train/val/test partition; never change
IMG_EXTS = {".jpg", ".jpeg", ".png"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="data config yaml, e.g. configs/data_caltech256.yaml")
    args = ap.parse_args()

    seed_everything(0)
    cfg = load_yaml(REPO_ROOT / args.data)
    root = REPO_ROOT / cfg["root"]
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    excluded = set(cfg.get("exclude_classes", []))
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in excluded)
    assert len(classes) == cfg["num_classes"], (
        f"expected {cfg['num_classes']} classes after excluding {excluded}, found {len(classes)}"
    )

    ratios = cfg["split_ratios"]
    fractions = cfg["fractions"]
    seeds = cfg["seeds"]

    rows = {"test": [], "val": []}
    train_rows = defaultdict(list)  # (frac, seed) -> rows
    report_lines = []
    skipped: list[str] = []
    per_class_counts = []

    for label, cls in enumerate(classes):
        files = sorted(p.name for p in (root / cls).iterdir() if p.is_file())
        imgs = [f for f in files if Path(f).suffix.lower() in IMG_EXTS]
        skipped += [f"{cls}/{f}" for f in files if Path(f).suffix.lower() not in IMG_EXTS]

        n = len(imgs)
        n_test = round(ratios["test"] * n)
        n_val = round(ratios["val"] * n)
        n_train = n - n_test - n_val
        assert min(n_test, n_val, n_train) >= 1, f"class {cls} too small (n={n})"

        # master permutation fixes the partition, per class -> stratified
        perm = np.random.default_rng([MASTER_SEED, label]).permutation(n)
        rel = [f"{cfg['root']}/{cls}/{imgs[i]}" for i in perm]
        test_f, val_f, pool = rel[:n_test], rel[n_test:n_test + n_val], rel[n_test + n_val:]

        rows["test"] += [(p, label, cls) for p in test_f]
        rows["val"] += [(p, label, cls) for p in val_f]
        per_class_counts.append((cls, n, n_train, n_val, n_test))

        for s in seeds:
            # independent nesting permutation per (seed, class)
            perm_s = np.random.default_rng([MASTER_SEED, s, label]).permutation(n_train)
            pool_s = [pool[i] for i in perm_s]
            prev: set = set()
            for frac in sorted(fractions):
                k = math.ceil(frac / 100 * n_train)
                subset = pool_s[:k]
                assert prev.issubset(subset), f"nesting broken: {cls} s{s} f{frac}"
                prev = set(subset)
                train_rows[(frac, s)] += [(p, label, cls) for p in subset]

    # global disjointness + existence
    test_set = {r[0] for r in rows["test"]}
    val_set = {r[0] for r in rows["val"]}
    pool_set = {r[0] for r in train_rows[(max(fractions), seeds[0])]}
    assert not (test_set & val_set) and not (test_set & pool_set) and not (val_set & pool_set), \
        "filepath overlap across train/val/test"
    for p in list(test_set)[:50] + list(val_set)[:50] + list(pool_set)[:50]:
        assert (REPO_ROOT / p).exists(), f"missing file: {p}"
    missing = [p for p in (test_set | val_set | pool_set) if not (REPO_ROOT / p).exists()]
    assert not missing, f"{len(missing)} referenced files missing, e.g. {missing[:3]}"

    out = REPO_ROOT / cfg["splits_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": range(len(classes)), "class_name": classes}).to_csv(out / "classes.csv", index=False)
    cols = ["filepath", "label", "class_name"]
    pd.DataFrame(rows["test"], columns=cols).to_csv(out / "test.csv", index=False)
    pd.DataFrame(rows["val"], columns=cols).to_csv(out / "val.csv", index=False)
    for (frac, s), rws in train_rows.items():
        pd.DataFrame(rws, columns=cols).to_csv(out / f"train_f{frac}_s{s}.csv", index=False)

    n_img = sum(c[1] for c in per_class_counts)
    report_lines.append(f"dataset={cfg['dataset']}  classes={len(classes)}  images={n_img}  "
                        f"master_seed={MASTER_SEED}  ratios={ratios}")
    report_lines.append(f"totals: test={len(test_set)}  val={len(val_set)}  train_pool={len(pool_set)}")
    for frac in sorted(fractions):
        sizes = {s: len(train_rows[(frac, s)]) for s in seeds}
        report_lines.append(f"train f{frac}: " + "  ".join(f"s{s}={n}" for s, n in sizes.items()))
    report_lines.append(f"skipped non-image files ({len(skipped)}): " + (", ".join(skipped) or "none"))
    report_lines.append("assertions passed: class count, disjointness, nesting, existence")
    report_lines.append("")
    report_lines.append(f"{'class':<28} {'n':>5} {'train':>6} {'val':>4} {'test':>5}")
    for cls, n, tr, va, te in per_class_counts:
        report_lines.append(f"{cls:<28} {n:>5} {tr:>6} {va:>4} {te:>5}")
    (out / "split_report.txt").write_text("\n".join(report_lines) + "\n")

    print("\n".join(report_lines[:8]))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
