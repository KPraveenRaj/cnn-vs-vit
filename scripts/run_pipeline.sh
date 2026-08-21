#!/usr/bin/env bash
# Full Phase-1 pipeline, sequential and resumable, from a completed ViT LR sweep
# through to every table and figure the report needs.
#
# Every stage is skip-if-done (train.py skips runs with metrics.json, the battery
# skips completed passes, aggregate/plots are pure functions of what exists), so
# this can be re-launched after any interruption and picks up where it stopped.
#
#   bash scripts/run_pipeline.sh            # core pipeline
#   FOOD101=1 bash scripts/run_pipeline.sh  # also run the Food-101 block last
#
# All stage output is tee'd into logs/ so the console history is a durable
# artifact rather than scrollback that disappears with the terminal.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}
mkdir -p logs
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(stamp)] === $* ===" | tee -a logs/pipeline.log; }

# 0. wait out any sweep still running from a previous launch -----------------
while pgrep -f "run_lr_sweep_vit_b16.sh" >/dev/null; do
  say "waiting for ViT LR sweep to finish"; sleep 120
done

# 1. select the ViT LR from the sweep ---------------------------------------
say "STAGE 1  select ViT-B/16 LR from declared sweep"
$PY -m src.train.select_lr --model-yaml configs/model_vit_b16.yaml \
    --pattern 'vit_b16_*sweep-lr*' 2>&1 | tee -a logs/pipeline.log

# 2. ViT-B/16 matrix: 4 fractions x 3 seeds, eval after each -----------------
say "STAGE 2  ViT-B/16 Caltech-256 matrix (12 runs)"
$PY -m src.train.run_matrix --models vit_b16 --with-eval 2>&1 | tee -a logs/vit_matrix.log

# 3. linear probes for both models ------------------------------------------
say "STAGE 3  linear probes (both models, 4 fractions x 3 seeds)"
for M in resnet50 vit_b16; do
  $PY -m src.train.linear_probe --model configs/model_${M}.yaml \
      --data configs/data_caltech256.yaml 2>&1 | tee -a logs/linprobe.log
done

# 4. evaluate probe checkpoints on the frozen test set ----------------------
say "STAGE 4  frozen-test eval for linear probes"
for RID in $(ls results/runs | grep linprobe); do
  [ -f "results/runs/$RID/clean_eval.json" ] && continue
  $PY -m src.eval.evaluate --run-id "$RID" 2>&1 | tee -a logs/linprobe.log
done

# 5. the full battery over every fullft checkpoint --------------------------
say "STAGE 5  eval battery (corruptions + frequency) over fullft checkpoints"
$PY -m src.eval.run_eval_battery --regime fullft 2>&1 | tee -a logs/battery.log

# 6. aggregate everything and regenerate all figures ------------------------
say "STAGE 6  aggregate, analyse, plot"
$PY -m src.utils.provenance          2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.aggregate        2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.calibration      2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.error_overlap    2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.visual_assets    2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.plots            2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.docs_index       2>&1 | tee -a logs/analysis.log

# 7. Food-101 confirmation block, explicitly last and opt-in ---------------
if [ "${FOOD101:-0}" = "1" ]; then
  say "STAGE 7  Food-101 confirmation block"
  bash scripts/run_food101.sh 2>&1 | tee -a logs/food101.log
  $PY -m src.analysis.aggregate 2>&1 | tee -a logs/analysis.log
  $PY -m src.analysis.plots     2>&1 | tee -a logs/analysis.log
fi

say "STAGE 8  rebuild decks and reports from the final tables"
$PY report/build/build_decks.py   2>&1 | tee -a logs/analysis.log
$PY report/build/build_reports.py 2>&1 | tee -a logs/analysis.log

say "PIPELINE COMPLETE"
