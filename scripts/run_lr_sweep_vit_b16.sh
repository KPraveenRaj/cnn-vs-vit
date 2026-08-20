#!/usr/bin/env bash
# Declared 3-point LR sweep for ViT-B/16 — runs BEFORE the ViT matrix block.
# Protocol mirrors the ResNet-50 sweep exactly (scripts/run_lr_sweep_resnet50.sh):
# f100, seed 0, short 8-epoch runs, cosine+warmup schedule shape preserved
# (warmup auto-shortened to 2 epochs), AdamW, effective batch 64.
#
# The GRID DIFFERS from ResNet-50's (1e-4/3e-4/1e-3) and that is the point of
# "declared per-model tuning": AdamW fine-tuning of a pre-trained ViT-B/16 lives
# roughly a decade lower than a ResNet's. Reusing the CNN grid would put the
# optimum at or below the bottom edge, i.e. would cripple the ViT at exactly the
# comparison the study exists to make. Same 3 points, same log spacing (x3.16),
# shifted one decade down: 1e-5 / 3e-5 / 1e-4.
#
# micro_batch 32 x accum 2 = 64 = effective_batch_size. The 32/2 split (rather
# than the yaml's 16/4) is memory bookkeeping only — measured peak 3.3 GB of
# 8 GB, ~90 img/s vs ~67 at 16/4. Effective batch, and therefore the optimization
# protocol, is unchanged.
#
# The winning LR (by best val top-1) is written into configs/model_vit_b16.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}

for LR in 1e-5 3e-5 1e-4; do
  $PY -m src.train.train \
    --model configs/model_vit_b16.yaml --data configs/data_caltech256.yaml \
    --fraction 100 --seed 0 --regime fullft \
    --lr "$LR" --epochs 8 --micro-batch 32 --run-suffix "sweep-lr${LR}"
done
echo "VIT SWEEP COMPLETE"
