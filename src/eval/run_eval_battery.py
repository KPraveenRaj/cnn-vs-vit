"""Full inference-only evaluation battery over trained checkpoints.

One battery, run identically on every checkpoint, producing one
results/runs/<run_id>/eval_results.json:

    clean                      top-1 / top-5 / macro-F1 on the frozen test set
    corruptions                3 families x 5 severities   (ImageNet-C convention)
    frequency.lowpass          ideal low-pass, 10 cutoff radii
    frequency.highpass         ideal high-pass, same radii
    frequency.band_noise       6 fixed-energy annuli, DC -> Nyquist

That is 1 + 15 + 20 + 6 = 42 passes over the test set per checkpoint.

Resumability is per-PASS, not per-run
-------------------------------------
eval_results.json is rewritten after every individual pass, and a pass whose
key is already present is skipped. A checkpoint that dies 30 passes in resumes
at 31 rather than at 1. Given that a ViT-B/16 battery is ~20 minutes and the
full sweep across 24 checkpoints is several hours on one laptop GPU, run-level
resumability would not have been enough.

Deliberately fullft-only by default
-----------------------------------
--regime fullft skips linear-probe checkpoints. Probes exist to answer the
data-efficiency question (how much does a frozen representation give you?), and
the robustness/frequency axes are about the fine-tuned models the report
actually compares. Running the battery on probes too would roughly double
compute for a question no committed figure asks. Pass --regime all to override.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.datasets import CsvImageDataset
from src.data.transforms import eval_geometric_tfms, eval_tfms, to_tensor_norm_tfms
from src.eval.corruptions import CORRUPTIONS, SEVERITIES, CorruptedCsvImageDataset
from src.eval.frequency import (CUTOFFS_BINS, NOISE_BANDS, NOISE_RMS,
                                BandNoiseCsvImageDataset, FilteredCsvImageDataset)
from src.models.factory import build_model
from src.utils.config import load_yaml
from src.utils.seed import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]


def _loader(ds, batch_size, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def score(model, loader, device, labels_all):
    """One pass -> top-1 / top-5 / macro-F1. Order is preserved (shuffle=False)."""
    preds, top5 = [], []
    use_amp = device == "cuda"
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
        out = out.float()
        preds.append(out.argmax(1).cpu().numpy())
        top5.append(out.topk(5, dim=1).indices.cpu().numpy())
    preds = np.concatenate(preds)
    top5 = np.concatenate(top5)
    y = labels_all[: len(preds)]
    return {
        "top1": float((preds == y).mean()),
        "top5": float((top5 == y[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(y, preds, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-ids", default=None, help="comma list; default: all with checkpoints")
    ap.add_argument("--regime", default="fullft", help="'fullft' (default) or 'all'")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test images (smoke test only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    seed_everything(0)
    out_root = REPO_ROOT / args.out_root
    runs_dir, ckpt_dir = out_root / "runs", out_root / "checkpoints"

    if args.run_ids:
        run_ids = args.run_ids.split(",")
    else:
        run_ids = sorted(p.name for p in runs_dir.iterdir()
                         if (ckpt_dir / p.name / "best.pt").exists()
                         and "sweep" not in p.name
                         and (args.regime == "all" or p.name.endswith(f"_{args.regime}")))
    print(f"[battery] {len(run_ids)} checkpoint(s): {', '.join(run_ids)}\n", flush=True)

    for n, rid in enumerate(run_ids, 1):
        run_dir = runs_dir / rid
        cfg = load_yaml(run_dir / "config.yaml")
        # A --limit run is a smoke test, not a result: it must never land in
        # eval_results.json, or the partial numbers would be treated as complete
        # and the real passes silently skipped.
        res_path = run_dir / ("eval_results.SMOKE.json" if args.limit
                              else "eval_results.json")
        res = json.loads(res_path.read_text()) if res_path.exists() else {}
        res.setdefault("run_id", rid)
        res.setdefault("corruptions", {})
        res.setdefault("frequency", {"lowpass": {}, "highpass": {}, "band_noise": {}})

        test_csv = REPO_ROOT / cfg["splits_dir"] / "test.csv"
        labels_all = np.asarray(pd.read_csv(test_csv)["label"])
        if args.limit:
            labels_all = labels_all[: args.limit]

        # Build the work list first so we can skip the model load entirely when
        # a checkpoint's battery is already complete.
        todo = []
        if "clean" not in res:
            todo.append(("clean", None))
        for c in CORRUPTIONS:
            for s in SEVERITIES:
                if str(s) not in res["corruptions"].get(c, {}):
                    todo.append(("corruption", (c, s)))
        for mode in ("lowpass", "highpass"):
            for rc in CUTOFFS_BINS:
                if str(rc) not in res["frequency"][mode]:
                    todo.append((mode, rc))
        for lo, hi in NOISE_BANDS:
            if f"{lo}-{hi}" not in res["frequency"]["band_noise"]:
                todo.append(("band", (lo, hi)))

        if not todo:
            print(f"[battery {n}/{len(run_ids)}] {rid}: complete, skipping", flush=True)
            continue

        print(f"[battery {n}/{len(run_ids)}] {rid}: {len(todo)} pass(es) to run", flush=True)
        ckpt = torch.load(ckpt_dir / rid / "best.pt", map_location=args.device,
                          weights_only=False)
        model = build_model(cfg["timm_name"], cfg["num_classes"]).to(args.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        timm_name, resn = cfg["timm_name"], cfg["resolution"]
        geo, tens = eval_geometric_tfms(resn), to_tensor_norm_tfms(timm_name)

        def run(ds, tag):
            if args.limit:
                ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
            t0 = time.perf_counter()
            m = score(model, _loader(ds, args.batch_size, args.num_workers),
                      args.device, labels_all)
            print(f"    {tag:<34s} top1 {m['top1']:.4f}  top5 {m['top5']:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            return m

        t_ckpt = time.perf_counter()
        for kind, spec in todo:
            if kind == "clean":
                ds = CsvImageDataset(test_csv, eval_tfms(timm_name, resn))
                res["clean"] = run(ds, "clean")
            elif kind == "corruption":
                c, s = spec
                ds = CorruptedCsvImageDataset(test_csv, geo, tens, c, s)
                res["corruptions"].setdefault(c, {})[str(s)] = run(ds, f"{c} s{s}")
            elif kind in ("lowpass", "highpass"):
                mode = "low" if kind == "lowpass" else "high"
                ds = FilteredCsvImageDataset(test_csv, geo, tens, spec, mode, resn)
                res["frequency"][kind][str(spec)] = run(ds, f"{kind} r={spec}")
            else:
                lo, hi = spec
                ds = BandNoiseCsvImageDataset(test_csv, geo, tens, lo, hi)
                res["frequency"]["band_noise"][f"{lo}-{hi}"] = run(ds, f"band [{lo},{hi})")

            res["meta"] = {
                "n_test": int(len(labels_all)),
                "limit": args.limit,
                "noise_rms": NOISE_RMS,
                "cutoffs_bins": list(CUTOFFS_BINS),
                "noise_bands": [list(b) for b in NOISE_BANDS],
                "device": args.device,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            res_path.write_text(json.dumps(res, indent=2))  # after EVERY pass

        print(f"  [done] {rid} in {(time.perf_counter() - t_ckpt) / 60:.1f} min\n", flush=True)

    print("[battery] all checkpoints complete", flush=True)


if __name__ == "__main__":
    main()
