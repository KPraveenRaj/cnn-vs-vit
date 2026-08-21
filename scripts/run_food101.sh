#!/usr/bin/env bash
# Food-101 cross-dataset confirmation block — the LAST thing that runs.
#
# Scope is deliberately minimal and was fixed in the project plan: 2 models x
# {100%, 25%} x 1 seed, full fine-tuning only. Its job is to check whether the
# Caltech-256 ordering survives a second, larger, finer-grained dataset — not to
# reproduce the whole matrix.
#
# The evaluation battery is NOT run on these checkpoints. The battery answers
# robustness and frequency questions, and every committed figure on those axes
# is defined on Caltech-256; running 42 extra passes per Food-101 checkpoint
# would roughly double this block's cost to answer a question no figure asks.
# Clean frozen-test evaluation IS run, since the confirmation claim needs it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}

bash scripts/extract_food101.sh

if [ ! -f data/splits/food101/test.csv ]; then
  echo "[food101] building frozen splits ..."
  $PY -m src.data.make_splits --data configs/data_food101.yaml
fi

$PY -m src.train.run_matrix --models resnet50,vit_b16 \
    --data configs/data_food101.yaml --fractions 100,25 --seeds 0 --with-eval
echo "FOOD101 BLOCK COMPLETE"
