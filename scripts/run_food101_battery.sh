#!/usr/bin/env bash
# Run the evaluation battery on the Food-101 checkpoints too, then refresh
# everything. Waits for the main chain (which is running the Food-101 training
# block) to finish first.
#
# WHY THIS EXISTS: the battery was originally skipped on Food-101 to save
# compute, on the grounds that every committed robustness/frequency figure is
# defined on Caltech-256. That reasoning was wrong for the purpose Food-101
# actually serves. As a CONFIRMATION dataset its job is to test whether the
# findings replicate — and the headline finding is the frequency one, so an
# accuracy-only confirmation tests everything except the contribution.
#
# Cost: Food-101's test split is 20,200 images against Caltech's 5,952 (3.4x),
# so ~26 min per ResNet checkpoint and ~52 min per ViT, about 2.6 h for four.
#
# run_eval_battery --regime fullft discovers by suffix, so it picks up the new
# Food-101 checkpoints and skips the 24 Caltech ones already complete.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ===" | tee -a logs/pipeline.log; }

while pgrep -f "run_rest_after_matrix\.sh" >/dev/null; do sleep 300; done
say "FOOD101 BATTERY  main chain finished; running battery on Food-101 checkpoints"
$PY -m src.eval.run_eval_battery --regime fullft 2>&1 | tee -a logs/battery.log

say "FOOD101 BATTERY  refreshing all tables, figures and documents"
for M in src.analysis.aggregate src.analysis.calibration src.analysis.error_overlap \
         src.analysis.plots src.analysis.docs_index; do
  $PY -m "$M" 2>&1 | tee -a logs/analysis.log
done
$PY report/build/build_decks.py            2>&1 | tee -a logs/analysis.log
$PY report/build/build_reports.py          2>&1 | tee -a logs/analysis.log
$PY report/build/build_submission_guide.py 2>&1 | tee -a logs/analysis.log
say "FOOD101 BATTERY COMPLETE"
