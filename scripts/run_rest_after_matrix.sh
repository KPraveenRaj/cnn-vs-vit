#!/usr/bin/env bash
# Everything after the corrected matrix: probes -> battery -> analysis ->
# documents -> Food-101 -> refresh. Waits for the matrix rather than being
# chained inside it, because the matrix was relaunched by apply_schedule_fix.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}
mkdir -p logs
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ===" | tee -a logs/pipeline.log; }

# NOTE: do NOT wait on apply_schedule_fix.sh — that script ends with `exec`,
# which REPLACES its own shell process, so the name disappears from the process
# table the moment the matrix starts and any such wait returns instantly. Wait
# on the actual worker instead.
while pgrep -f "src\.train\.run_matrix" >/dev/null; do sleep 300; done
say "STAGE 3  linear probes (both models, 4 fractions x 3 seeds)"
for M in resnet50 vit_b16; do
  $PY -m src.train.linear_probe --model configs/model_${M}.yaml \
      --data configs/data_caltech256.yaml 2>&1 | tee -a logs/linprobe.log
done

say "STAGE 4  eval battery (corruptions + frequency) over fullft checkpoints"
$PY -m src.eval.run_eval_battery --regime fullft 2>&1 | tee -a logs/battery.log

say "STAGE 5  aggregate, analyse, plot"
for M in provenance:src.utils.provenance aggregate:src.analysis.aggregate \
         calibration:src.analysis.calibration overlap:src.analysis.error_overlap \
         assets:src.analysis.visual_assets plots:src.analysis.plots \
         docs:src.analysis.docs_index; do
  $PY -m "${M#*:}" 2>&1 | tee -a logs/analysis.log
done

say "STAGE 6  build decks and reports"
$PY report/build/build_decks.py   2>&1 | tee -a logs/analysis.log
$PY report/build/build_reports.py 2>&1 | tee -a logs/analysis.log

say "STAGE 7  Food-101 confirmation block"
bash scripts/run_food101.sh 2>&1 | tee -a logs/food101.log

say "STAGE 8  final refresh with Food-101 included"
for M in src.analysis.aggregate src.analysis.calibration src.analysis.error_overlap \
         src.analysis.plots src.analysis.docs_index; do
  $PY -m "$M" 2>&1 | tee -a logs/analysis.log
done
$PY report/build/build_decks.py   2>&1 | tee -a logs/analysis.log
$PY report/build/build_reports.py 2>&1 | tee -a logs/analysis.log
say "ALL COMPLETE"
