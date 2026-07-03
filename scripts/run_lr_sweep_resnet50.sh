#!/usr/bin/env bash
# Declared 3-point LR sweep for ResNet-50 — runs BEFORE the experiment matrix.
# Protocol: f100, seed 0, short 8-epoch runs, schedule shape preserved
# (cosine + warmup auto-shortened to 2 epochs). Grid declared here once:
# 1e-4 / 3e-4 / 1e-3 (AdamW, effective batch 64).
# The winning LR (by best val top-1) is written into configs/model_resnet50.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}

for LR in 1e-4 3e-4 1e-3; do
  $PY -m src.train.train \
    --model configs/model_resnet50.yaml --data configs/data_caltech256.yaml \
    --fraction 100 --seed 0 --regime fullft \
    --lr "$LR" --epochs 8 --run-suffix "sweep-lr${LR}"
done
echo "SWEEP COMPLETE"
