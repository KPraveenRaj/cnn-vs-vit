"""Linear probes on frozen features: how much does the pre-trained representation give?

Two passes, for one reason: features from a FROZEN backbone do not depend on
the data fraction or the seed, so extracting them once per (model, dataset) and
reusing them across all 12 (fraction, seed) cells turns a 24-run block into one
forward pass per model plus 24 trivial logistic regressions.

    pass 1  cache   every unique image across train/val/test -> feature matrix
                    keyed by filepath, saved to results/features/ (gitignored)
    pass 2  fit     one linear head per (fraction, seed), reading rows by
                    filepath from the cache

Because pass 1 keys by FILEPATH rather than by split, it is agnostic to how the
fractions were nested -- no assumption that train_f100_s0 and train_f100_s1
contain the same images in the same order. Any split CSV is a gather.

Declared protocol differences from full fine-tuning (both applied identically
to both models, so the comparison stays controlled)
--------------------------------------------------
1. No train-time augmentation. Features are cached once; augmenting would mean
   re-extracting every epoch, which defeats the entire point of caching. The
   deterministic eval transform is used for all splits.
2. ONE learning rate for both architectures, chosen by the same declared
   3-point sweep discipline used for fine-tuning (--sweep). Fitting a linear
   head on frozen features is a near-convex problem with no architecture-
   specific optimization dynamics, so a shared LR here is more controlled, not
   less -- unlike fine-tuning, where forcing one LR would cripple one family.

Everything else matches train.py: AdamW, cosine schedule with warmup, early
stopping on val top-1, and the same metrics.json / train_log.csv contract, so
probe runs drop straight into aggregate.py alongside the fullft runs.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.datasets import CsvImageDataset
from src.data.transforms import eval_tfms
from src.models.factory import feature_extractor
from src.utils.config import load_yaml, merge, save_yaml
from src.utils.hw_monitor import peak_vram_mb, reset_peak_vram
from src.utils.seed import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_LR_GRID = (1e-4, 1e-3, 1e-2)   # declared 3-point grid, log-spaced x10
PROBE_LR_DEFAULT = 1e-3


# --------------------------------------------------------------------------
# Pass 1: feature cache
# --------------------------------------------------------------------------
def cache_features(cfg, device, batch_size=128, num_workers=8):
    """Extract pooled features for every unique image in the dataset's splits."""
    splits = REPO_ROOT / cfg["splits_dir"]
    cache_dir = REPO_ROOT / "results" / "features" / f"{cfg['model_name']}_{cfg['dataset']}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "features.npz"
    if cache_path.exists():
        print(f"[cache] reusing {cache_path.relative_to(REPO_ROOT)}")
        z = np.load(cache_path, allow_pickle=True)
        return {p: i for i, p in enumerate(z["paths"])}, z["feats"], cache_path

    paths = []
    for csv in sorted(splits.glob("*.csv")):
        if csv.name == "classes.csv":
            continue
        paths.extend(pd.read_csv(csv)["filepath"].tolist())
    paths = sorted(set(paths))
    print(f"[cache] {len(paths)} unique images -> {cfg['model_name']} features", flush=True)

    tmp = cache_dir / "_all.csv"
    pd.DataFrame({"filepath": paths, "label": 0}).to_csv(tmp, index=False)
    ds = CsvImageDataset(tmp, eval_tfms(cfg["timm_name"], cfg["resolution"]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = feature_extractor(cfg["timm_name"]).to(device).eval()
    reset_peak_vram()
    out, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                out.append(model(x).float().cpu())
            if i % 40 == 0:
                print(f"  {i * batch_size}/{len(paths)}", flush=True)
    feats = torch.cat(out).numpy().astype(np.float32)
    tmp.unlink()
    np.savez(cache_path, paths=np.array(paths), feats=feats)
    print(f"[cache] {feats.shape} in {(time.perf_counter() - t0) / 60:.1f} min "
          f"-> {cache_path.relative_to(REPO_ROOT)}", flush=True)
    return {p: i for i, p in enumerate(paths)}, feats, cache_path


def gather(index, feats, csv_path):
    df = pd.read_csv(csv_path)
    rows = np.fromiter((index[p] for p in df["filepath"]), dtype=np.int64, count=len(df))
    return torch.from_numpy(feats[rows]), torch.tensor(df["label"].to_numpy(), dtype=torch.long)


# --------------------------------------------------------------------------
# Pass 2: one linear head
# --------------------------------------------------------------------------
def fit_head(Xtr, ytr, Xva, yva, num_classes, lr, epochs, warmup, patience,
             batch_size, device, seed, log_rows=None):
    seed_everything(seed)
    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    steps = math.ceil(len(Xtr) / batch_size)

    def lr_at(step):
        if step < steps * warmup:
            return (step + 1) / max(1, steps * warmup)
        prog = (step - steps * warmup) / max(1, steps * epochs - steps * warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = LambdaLR(opt, lr_at)
    crit = nn.CrossEntropyLoss()
    Xtr, ytr, Xva, yva = (t.to(device) for t in (Xtr, ytr, Xva, yva))

    best, best_ep, left, best_state = -1.0, -1, patience, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot, corr = 0.0, 0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            out = head(Xtr[idx])
            loss = crit(out, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            corr += (out.argmax(1) == ytr[idx]).sum().item()
        head.eval()
        with torch.no_grad():
            vo = head(Xva)
            v1 = (vo.argmax(1) == yva).float().mean().item()
            v5 = vo.topk(5, 1).indices.eq(yva[:, None]).any(1).float().mean().item()
            vl = crit(vo, yva).item()
        if log_rows is not None:
            log_rows.append({"epoch": ep, "train_loss": round(tot / len(Xtr), 5),
                             "train_top1": round(corr / len(Xtr), 5),
                             "val_loss": round(vl, 5), "val_top1": round(v1, 5),
                             "val_top5": round(v5, 5), "lr": opt.param_groups[0]["lr"],
                             "epoch_time_s": 0.0, "imgs_per_sec": 0.0,
                             "peak_vram_mb": round(peak_vram_mb(), 1)})
        if v1 > best:
            best, best_ep, left = v1, ep, patience
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            best_top5 = v5
        else:
            left -= 1
            if left <= 0:
                break
    head.load_state_dict(best_state)
    return head, best, best_top5, best_ep


@torch.no_grad()
def write_clean_eval(head, index, feats, test_csv, run_dir, run_id, num_classes,
                     device, best_epoch, best_val_top1):
    """Score the frozen test set from cached features; emit evaluate.py's outputs."""
    from sklearn.metrics import f1_score

    df = pd.read_csv(test_csv)
    X, y = gather(index, feats, test_csv)
    head.eval()
    logits = head(X.to(device)).float().cpu()
    prob = logits.softmax(1)
    conf, pred = prob.max(1)
    preds = pred.numpy()
    labels = y.numpy()
    correct = preds == labels
    top5 = logits.topk(5, dim=1).indices.numpy()

    per_class = [float(correct[labels == c].mean()) if (labels == c).any() else 0.0
                 for c in range(num_classes)]
    summary = {
        "run_id": run_id,
        "checkpoint_epoch": int(best_epoch), "checkpoint_val_top1": float(best_val_top1),
        "n_test": int(len(labels)),
        "top1": float(correct.mean()),
        "top5": float((top5 == labels[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_acc": [round(a, 6) for a in per_class],
        "eval_seconds": 0.0,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "clean_eval.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({
        "filepath": df["filepath"],
        "label": labels.astype("int32"),
        "pred": preds.astype("int32"),
        "confidence": conf.numpy().astype("float32"),
        "correct": correct,
    }).to_parquet(run_dir / "predictions.parquet", index=False)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--fractions", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="run the declared 3-point probe LR sweep on f100 s0 and exit")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    cfg = merge(load_yaml(REPO_ROOT / "configs/base.yaml"),
                load_yaml(REPO_ROOT / args.model),
                load_yaml(REPO_ROOT / args.data))
    device = args.device
    splits = REPO_ROOT / cfg["splits_dir"]
    out_root = REPO_ROOT / args.out_root

    index, feats, cache_path = cache_features(cfg, device)
    Xva, yva = gather(index, feats, splits / "val.csv")

    if args.sweep:
        Xtr, ytr = gather(index, feats, splits / "train_f100_s0.csv")
        print(f"\n[probe sweep] {cfg['model_name']}: declared grid {PROBE_LR_GRID}")
        for lr in PROBE_LR_GRID:
            _, v1, _, ep = fit_head(Xtr, ytr, Xva, yva, cfg["num_classes"], lr,
                                    args.epochs, 3, cfg["early_stop_patience"],
                                    args.batch_size, device, 0)
            print(f"  lr={lr:<8g} best val top-1 {v1:.4f} (epoch {ep})", flush=True)
        return

    lr = args.lr if args.lr is not None else PROBE_LR_DEFAULT
    fractions = [int(f) for f in args.fractions.split(",")] if args.fractions else sorted(cfg["fractions"])
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]

    for frac in fractions:
        for seed in seeds:
            rid = f"{cfg['model_name']}_{cfg['dataset']}_f{frac}_s{seed}_linprobe"
            run_dir = out_root / "runs" / rid
            if (run_dir / "metrics.json").exists():
                print(f"[skip] {rid} already complete")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir = out_root / "checkpoints" / rid
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            Xtr, ytr = gather(index, feats, splits / f"train_f{frac}_s{seed}.csv")
            rows, t0 = [], time.perf_counter()
            head, v1, v5, best_ep = fit_head(Xtr, ytr, Xva, yva, cfg["num_classes"], lr,
                                             args.epochs, 3, cfg["early_stop_patience"],
                                             args.batch_size, device, seed, rows)
            wall = (time.perf_counter() - t0) / 60

            # Frozen-test evaluation happens HERE, not via src.eval.evaluate.
            # evaluate.py reconstructs a full backbone and loads the checkpoint into
            # it; a probe checkpoint holds only an nn.Linear head, so that path
            # cannot work. It is also unnecessary — the test features are already
            # in the cache, so scoring is one matmul instead of 5,952 forward
            # passes. Output format is byte-compatible with evaluate.py's, so probe
            # runs are first-class citizens for aggregate / calibration /
            # error_overlap.
            write_clean_eval(head, index, feats, splits / "test.csv", run_dir,
                             rid, cfg["num_classes"], device, best_ep, v1)

            rcfg = dict(cfg)
            rcfg.update({"fraction": frac, "seed": seed, "regime": "linprobe",
                         "lr": lr, "run_id": rid, "feature_dim": int(Xtr.shape[1]),
                         "probe_lr_grid": list(PROBE_LR_GRID),
                         "feature_cache": str(cache_path.relative_to(REPO_ROOT))})
            save_yaml(rcfg, run_dir / "config.yaml")
            pd.DataFrame(rows).to_csv(run_dir / "train_log.csv", index=False)
            torch.save({"state_dict": head.state_dict(), "run_id": rid,
                        "epoch": best_ep, "val_top1": v1, "config": rcfg},
                       ckpt_dir / "best.pt")
            json.dump({
                "run_id": rid, "model_name": cfg["model_name"], "timm_name": cfg["timm_name"],
                "dataset": cfg["dataset"], "num_classes": cfg["num_classes"],
                "fraction": frac, "seed": seed, "regime": "linprobe",
                "lr": lr, "weight_decay": cfg["weight_decay"], "optimizer": "adamw",
                "effective_batch_size": args.batch_size, "micro_batch_size": args.batch_size,
                "grad_accum_steps": 1, "epochs_cap": args.epochs,
                "warmup_epochs_actual": 3, "epochs_ran": len(rows),
                "early_stopped": len(rows) < args.epochs,
                "best_epoch": best_ep, "best_val_top1": v1, "best_val_top5": v5,
                "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
                "feature_dim": int(Xtr.shape[1]),
                "params_m": round(head.weight.numel() / 1e6, 4),
                "peak_vram_mb": round(peak_vram_mb(), 1),
                "mean_epoch_time_s": 0.0, "train_imgs_per_sec": 0.0,
                "total_wall_min": round(wall, 2),
                "torch": torch.__version__, "timm": timm.__version__,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, open(run_dir / "metrics.json", "w"), indent=2)
            ce = json.loads((run_dir / "clean_eval.json").read_text())
            print(f"[done] {rid}: val {v1:.4f} / test {ce['top1']:.4f} "
                  f"(epoch {best_ep}), n_train={len(Xtr)}, {wall:.2f} min", flush=True)


if __name__ == "__main__":
    main()
