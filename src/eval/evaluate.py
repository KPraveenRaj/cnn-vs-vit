"""Frozen-test evaluation: checkpoint -> clean metrics + per-image prediction table.

Reads the run's own merged config (results/runs/<run_id>/config.yaml), so a
checkpoint can never be evaluated under different data settings than it was
trained with.

Outputs into results/runs/<run_id>/:
  clean_eval.json         top-1, top-5, macro-F1, per-class accuracy, n_test
  predictions.parquet     filepath, label, pred, confidence, correct

The prediction table is the raw material for the CPU-only analyses (error
overlap between models, calibration / ECE, rare-class behaviour), so it is
saved for EVERY evaluated checkpoint, always — never just the summary.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.datasets import build_loader
from src.data.transforms import eval_tfms
from src.models.factory import build_model
from src.utils.config import load_yaml
from src.utils.seed import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    seed_everything(0)  # eval path is deterministic; seed anyway for uniformity
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = out_root / "runs" / args.run_id
    cfg = load_yaml(run_dir / "config.yaml")
    ckpt_path = out_root / "checkpoints" / args.run_id / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

    model = build_model(cfg["timm_name"], cfg["num_classes"]).cuda()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_csv = REPO_ROOT / cfg["splits_dir"] / "test.csv"
    df = pd.read_csv(test_csv)  # loader below preserves CSV order (shuffle=False)
    loader = build_loader(test_csv, eval_tfms(cfg["timm_name"], cfg["resolution"]),
                          batch_size=args.batch_size, shuffle=False, seed=0,
                          num_workers=cfg.get("num_workers", 8))

    preds, confs, top5_hits = [], [], []
    labels = np.array(df["label"])
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
            prob = out.float().softmax(1)
            conf, pred = prob.max(1)
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())
            top5_hits.append(out.topk(5, dim=1).indices.cpu().eq(y[:, None]).any(1).numpy())
    preds = np.concatenate(preds)
    confs = np.concatenate(confs)
    top5_hits = np.concatenate(top5_hits)
    eval_s = time.perf_counter() - t0

    from sklearn.metrics import f1_score
    correct = preds == labels
    per_class = [float(correct[labels == c].mean()) for c in range(cfg["num_classes"])]
    summary = {
        "run_id": args.run_id,
        "checkpoint_epoch": ckpt["epoch"], "checkpoint_val_top1": ckpt["val_top1"],
        "n_test": int(len(labels)),
        "top1": float(correct.mean()),
        "top5": float(top5_hits.mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_acc": [round(a, 6) for a in per_class],
        "eval_seconds": round(eval_s, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "clean_eval.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({
        "filepath": df["filepath"],
        "label": labels.astype("int32"),
        "pred": preds.astype("int32"),
        "confidence": confs.astype("float32"),
        "correct": correct,
    }).to_parquet(run_dir / "predictions.parquet", index=False)
    print(f"[eval] {args.run_id}: top1 {summary['top1']:.4f}  top5 {summary['top5']:.4f}  "
          f"macroF1 {summary['macro_f1']:.4f}  ({summary['n_test']} imgs, {eval_s:.0f}s)")


if __name__ == "__main__":
    main()
