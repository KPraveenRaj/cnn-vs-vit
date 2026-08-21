#!/usr/bin/env bash
# Watch the running ViT LR probe; extend the grid upward while the winner keeps
# landing on the top edge, then hand off to the full pipeline.
#
# WHY: a sweep whose winner sits at an edge has not bracketed the optimum, so the
# model may simply be under-tuned. In THIS study that is not a cosmetic issue —
# the entire claim rests on neither family being handicapped by its learning
# rate, so an under-tuned ViT would manufacture a data-efficiency result out of
# a tuning artefact. The grid therefore extends until the winner is interior.
#
# Capped at MAX_POINTS so it cannot walk away unattended; if the cap is reached
# with the winner still on the edge, it stops and says so rather than guessing.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}
MAX_POINTS=6
NEXT_LR=(1e-3 3e-3)

wait_for_train() { while pgrep -f "src\.train\.train" >/dev/null; do sleep 60; done; }

for attempt in 0 1; do
  wait_for_train
  n=$(ls -d results/runs/vit_b16_*sweep-lr*/ 2>/dev/null | wc -l)
  top=$($PY - <<'PY'
import json, glob
pts = []
for d in glob.glob("results/runs/vit_b16_*sweep-lr*/metrics.json"):
    m = json.load(open(d)); pts.append((m["lr"], m["best_val_top1"]))
pts.sort()
best = max(pts, key=lambda t: t[1])
print("EDGE" if best[0] == pts[-1][0] else "INTERIOR")
PY
)
  echo "[resolve-lr] $n points swept; winner is $top"
  if [ "$top" = "INTERIOR" ] || [ "$n" -ge "$MAX_POINTS" ]; then break; fi
  lr=${NEXT_LR[$attempt]}
  echo "[resolve-lr] winner still on the top edge — probing lr=$lr"
  $PY -m src.train.train --model configs/model_vit_b16.yaml \
      --data configs/data_caltech256.yaml --fraction 100 --seed 0 --regime fullft \
      --lr "$lr" --epochs 8 --micro-batch 32 --run-suffix "sweep-lr${lr}" \
      >> logs/vit_sweep_extend.log 2>&1
done

echo "[resolve-lr] final grid:"
$PY - <<'PY'
import json, glob
pts = sorted((json.load(open(d))["lr"], json.load(open(d))["best_val_top1"])
             for d in glob.glob("results/runs/vit_b16_*sweep-lr*/metrics.json"))
best = max(pts, key=lambda t: t[1])
for lr, v in pts:
    print(f"    {lr:<10g} {v:.4f}" + ("   <- winner" if lr == best[0] else ""))
print(f"    winner is {'AT THE EDGE' if best[0]==pts[-1][0] else 'INTERIOR'}")
PY

echo "[resolve-lr] handing off to the pipeline"
exec bash scripts/run_pipeline.sh
