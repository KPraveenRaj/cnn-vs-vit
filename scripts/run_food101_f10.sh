#!/usr/bin/env bash
# Add the f10 cell to Food-101 so the contribution claim can be tested over the
# same wide data range where it was found on Caltech-256.
#
# WHY: the claim is that the CNN's spectral robustness profile moves with the
# data budget and the transformer's does not. On Caltech that effect is 12x over
# f10->f100 but only 2.2x over f25->f100. Food-101 was originally run at
# {25, 100} only, so the wide-range version was never tested there — and
# Food-101's f25 is 17,675 images, over 3x Caltech's f25, so it plausibly sits
# past the data-starved regime the effect lives in.
#
# Regenerating the splits left test.csv and val.csv byte-identical (the master
# permutation is seed-fixed), so every existing Food-101 result remains valid.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ===" | tee -a logs/pipeline.log; }

say "FOOD101 f10  training both models"
$PY -m src.train.run_matrix --models resnet50,vit_b16 \
    --data configs/data_food101.yaml --fractions 10 --seeds 0 --with-eval \
    2>&1 | tee -a logs/food101.log

say "FOOD101 f10  battery"
$PY -m src.eval.run_eval_battery --regime fullft 2>&1 | tee -a logs/battery.log

say "FOOD101 f10  refreshing analysis and documents"
for M in src.analysis.aggregate src.analysis.calibration src.analysis.error_overlap \
         src.analysis.replication src.analysis.plots src.analysis.docs_index; do
  $PY -m "$M" 2>&1 | tee -a logs/analysis.log
done
$PY report/build/build_decks.py            2>&1 | tee -a logs/analysis.log
$PY report/build/build_reports.py          2>&1 | tee -a logs/analysis.log
$PY report/build/build_submission_guide.py 2>&1 | tee -a logs/analysis.log
$PY report/build/build_handbook.py         2>&1 | tee -a logs/analysis.log
say "FOOD101 f10 COMPLETE"
