#!/usr/bin/env bash
# Apply the LR-schedule fix and relaunch the full Caltech-256 matrix.
#
# WHAT WENT WRONG
# base.yaml declared a 30-epoch cosine with early_stop_patience 5. Over 30
# epochs the LR is still ~9e-5 at epoch 8, so validation sits on a noisy plateau
# and patience expires before the annealing phase that actually converges the
# model. Every single ViT run early-stopped; measured on f100 seed 0, same LR and
# same seed:
#     ViT-B/16   8-epoch annealed sweep 0.9037   vs   30-epoch truncated 0.8734
#     ResNet-50  8-epoch annealed sweep 0.8943   vs   30-epoch truncated 0.8920
# The harm is 15x larger for the transformer, which biases the comparison the
# whole project exists to make -- and it inverted the f100 ordering.
#
# THE FIX
# epochs 15 so the cosine completes, patience 8 so early stopping can only fire
# after epoch 9 and only on 8 consecutive non-improving epochs, i.e. genuine
# divergence rather than plateau noise. 15 and not 12 because the completed runs
# show the small fractions need the budget: ViT peaks at epoch 14-17 on f10 and
# ResNet at 18-25, so a shorter schedule would trade one bias for another.
#
# BOTH arms are redone, even though ResNet is barely affected: running the two
# models under different schedules would break the control.
#
# The ep30 runs are ARCHIVED, not deleted. The interaction is a real
# methodological finding and belongs in the report.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}
STAMP=ep30_truncated
ARCHIVE=results/archive/${STAMP}

echo "[fix] archiving the ep30 matrix runs to ${ARCHIVE}"
mkdir -p "${ARCHIVE}/runs" "${ARCHIVE}/checkpoints"
moved=0
for d in results/runs/*_caltech256_f*_s*_fullft; do
  b=$(basename "$d")
  case "$b" in *sweep*|*diag*) continue;; esac
  [ -d "$d" ] || continue
  mv "$d" "${ARCHIVE}/runs/$b"
  [ -d "results/checkpoints/$b" ] && mv "results/checkpoints/$b" "${ARCHIVE}/checkpoints/$b"
  moved=$((moved+1))
done
echo "[fix] archived ${moved} runs"

echo "[fix] relaunching the full matrix under the corrected schedule"
exec $PY -m src.train.run_matrix --models resnet50,vit_b16 --with-eval
