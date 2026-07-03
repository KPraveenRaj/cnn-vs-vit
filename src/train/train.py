"""One training run: merged config in -> best.pt + metrics.json + train_log.csv out.

Run identity
------------
    {model}_{dataset}_f{frac}_s{seed}_{regime}[-suffix]
e.g. resnet50_caltech256_f100_s0_fullft, and everything the run produces is
keyed by it:
    results/runs/<run_id>/          config.yaml, train_log.csv, metrics.json
    results/checkpoints/<run_id>/   best.pt  (highest val top-1)
If metrics.json already exists the run is skipped — that makes every driver
script crash-resumable by construction (run_matrix relies on this).
LR-sweep runs use --run-suffix so they never collide with matrix run IDs.

Protocol facts encoded here (do not change casually)
----------------------------------------------------
- AMP autocast + gradient accumulation: effective batch = micro_batch x accum,
  fixed in base.yaml; the micro/accum split is per-model memory bookkeeping.
- Cosine LR schedule with linear warmup, stepped per OPTIMIZER step (not per
  micro-batch). Warmup is auto-shortened to epochs//4 for short sweep runs.
- Early stopping on val top-1 with base.yaml patience.
- Merged config is written to the run folder at start; metrics.json at the
  end is the "run completed" marker.
- train_log.csv per epoch: losses, accuracies, LR, epoch time, imgs/sec,
  peak VRAM — the deployment-cost axis comes from here for free.
"""
import argparse
import json
import math
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import pandas as pd

from src.data.datasets import build_loader
from src.data.transforms import eval_tfms, train_tfms
from src.models.factory import build_model
from src.utils.config import load_yaml, merge, save_yaml
from src.utils.hw_monitor import param_count_m, peak_vram_mb, reset_peak_vram
from src.utils.seed import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    ap = argparse.ArgumentParser(description="single training run")
    ap.add_argument("--model", required=True, help="configs/model_*.yaml")
    ap.add_argument("--data", required=True, help="configs/data_*.yaml")
    ap.add_argument("--fraction", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--regime", default="fullft")
    ap.add_argument("--lr", type=float, default=None, help="override model yaml lr")
    ap.add_argument("--epochs", type=int, default=None, help="override base yaml epochs")
    ap.add_argument("--run-suffix", default="", help="appended to run_id (used by LR sweep)")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--micro-batch", type=int, default=None)
    return ap.parse_args()


def lr_lambda_factory(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        top1 += (out.argmax(1) == y).sum().item()
        top5 += out.topk(5, dim=1).indices.eq(y[:, None]).any(1).sum().item()
        n += x.size(0)
    return loss_sum / n, top1 / n, top5 / n


def main():
    args = parse_args()
    cli = {"fraction": args.fraction, "seed": args.seed, "regime": args.regime}
    if args.lr is not None:
        cli["lr"] = args.lr
    if args.epochs is not None:
        cli["epochs"] = args.epochs
    if args.micro_batch is not None:
        cli["micro_batch_size"] = args.micro_batch
    cfg = merge(load_yaml(REPO_ROOT / "configs/base.yaml"),
                load_yaml(REPO_ROOT / args.model),
                load_yaml(REPO_ROOT / args.data),
                cli)

    if cfg.get("lr") is None:
        raise SystemExit("lr is null: pass --lr or set the swept value in the model yaml")
    eff, micro = cfg["effective_batch_size"], cfg["micro_batch_size"]
    if eff % micro != 0:
        raise SystemExit(f"effective_batch_size {eff} not divisible by micro_batch_size {micro}")
    accum = eff // micro
    cfg["grad_accum_steps"] = accum

    seed_everything(cfg["seed"])
    assert torch.cuda.is_available(), "CUDA required (8 GB laptop GPU is the study platform)"
    device = "cuda"

    run_id = f"{cfg['model_name']}_{cfg['dataset']}_f{cfg['fraction']}_s{cfg['seed']}_{cfg['regime']}"
    if args.run_suffix:
        run_id += f"-{args.run_suffix}"
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = out_root / "runs" / run_id
    ckpt_dir = out_root / "checkpoints" / run_id
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {run_id} already complete")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    splits = REPO_ROOT / cfg["splits_dir"]
    train_loader = build_loader(splits / f"train_f{cfg['fraction']}_s{cfg['seed']}.csv",
                                train_tfms(cfg["timm_name"], cfg["resolution"]),
                                batch_size=micro, shuffle=True, seed=cfg["seed"],
                                num_workers=cfg["num_workers"])
    val_loader = build_loader(splits / "val.csv",
                              eval_tfms(cfg["timm_name"], cfg["resolution"]),
                              batch_size=128, shuffle=False, seed=cfg["seed"],
                              num_workers=cfg["num_workers"])
    n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)

    model = build_model(cfg["timm_name"], cfg["num_classes"]).to(device)
    params_m = param_count_m(model)

    epochs = cfg["epochs"]
    warmup_actual = min(cfg["warmup_epochs"], max(1, epochs // 4))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = LambdaLR(optimizer, lr_lambda_factory(steps_per_epoch * epochs,
                                                      steps_per_epoch * warmup_actual))
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    cfg["warmup_epochs_actual"] = warmup_actual
    cfg["run_id"] = run_id
    save_yaml(cfg, run_dir / "config.yaml")
    print(f"[run] {run_id}: {params_m:.1f}M params, {n_train} train / {n_val} val images, "
          f"lr={cfg['lr']}, epochs<={epochs} (warmup {warmup_actual}), "
          f"micro_batch={micro} x accum={accum}")

    log_rows, best_top1, best_epoch, patience_left = [], -1.0, -1, cfg["early_stop_patience"]
    t_start = time.perf_counter()
    for epoch in range(epochs):
        reset_peak_vram()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, correct, seen = 0.0, 0, 0
        t0 = time.perf_counter()
        n_batches = len(train_loader)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
                # constant 1/accum scaling (last partial group scales the same;
                # negligible and keeps LR semantics simple)
                loss = criterion(out, y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += loss.item() * accum * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
        train_time = time.perf_counter() - t0

        val_loss, val_top1, val_top5 = validate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": round(loss_sum / seen, 5),
            "train_top1": round(correct / seen, 5),
            "val_loss": round(val_loss, 5),
            "val_top1": round(val_top1, 5),
            "val_top5": round(val_top5, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(train_time, 1),
            "imgs_per_sec": round(seen / train_time, 1),
            "peak_vram_mb": round(peak_vram_mb(), 1),
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
        print(f"  epoch {epoch:02d}  train {row['train_top1']:.4f}  val {val_top1:.4f}  "
              f"({row['epoch_time_s']}s, {row['peak_vram_mb']:.0f} MB)")

        if val_top1 > best_top1:
            best_top1, best_epoch = val_top1, epoch
            patience_left = cfg["early_stop_patience"]
            torch.save({"state_dict": model.state_dict(), "run_id": run_id,
                        "epoch": epoch, "val_top1": val_top1, "config": cfg},
                       ckpt_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch} (best {best_top1:.4f} @ {best_epoch})")
                break

    total_min = (time.perf_counter() - t_start) / 60
    best_row = log_rows[best_epoch]
    metrics = {
        "run_id": run_id,
        "model_name": cfg["model_name"], "timm_name": cfg["timm_name"],
        "dataset": cfg["dataset"], "num_classes": cfg["num_classes"],
        "fraction": cfg["fraction"], "seed": cfg["seed"], "regime": cfg["regime"],
        "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
        "optimizer": cfg["optimizer"], "effective_batch_size": eff,
        "micro_batch_size": micro, "grad_accum_steps": accum,
        "epochs_cap": epochs, "warmup_epochs_actual": warmup_actual,
        "epochs_ran": len(log_rows), "early_stopped": len(log_rows) < epochs,
        "best_epoch": best_epoch, "best_val_top1": best_top1,
        "best_val_top5": best_row["val_top5"],
        "n_train": n_train, "n_val": n_val,
        "params_m": round(params_m, 2),
        "peak_vram_mb": max(r["peak_vram_mb"] for r in log_rows),
        "mean_epoch_time_s": round(sum(r["epoch_time_s"] for r in log_rows) / len(log_rows), 1),
        "train_imgs_per_sec": round(sum(r["imgs_per_sec"] for r in log_rows) / len(log_rows), 1),
        "total_wall_min": round(total_min, 1),
        "torch": torch.__version__, "timm": timm.__version__,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] {run_id}: best val top-1 {best_top1:.4f} (epoch {best_epoch}), "
          f"{total_min:.0f} min, peak {metrics['peak_vram_mb']:.0f} MB")


if __name__ == "__main__":
    main()
