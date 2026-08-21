#!/usr/bin/env bash
# Run the Food-101 confirmation block once the main pipeline has finished, then
# refresh every table, figure and document so the deliverables include it.
#
# Separate from run_pipeline.sh on purpose: the pipeline was already running when
# Food-101 became viable, and editing a live bash script is unsafe — bash reads
# scripts by byte offset, so shifting content under a running interpreter makes
# it resume mid-token. This waits for that process to exit instead.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}
mkdir -p logs
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ===" | tee -a logs/pipeline.log; }

while pgrep -f "run_pipeline\.sh" >/dev/null; do sleep 300; done
say "FOLLOW-ON  main pipeline finished; starting Food-101 block"

bash scripts/run_food101.sh 2>&1 | tee -a logs/food101.log

say "FOLLOW-ON  refreshing tables, figures and documents with Food-101 included"
$PY -m src.analysis.aggregate     2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.calibration   2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.error_overlap 2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.plots         2>&1 | tee -a logs/analysis.log
$PY -m src.analysis.docs_index    2>&1 | tee -a logs/analysis.log
$PY report/build/build_decks.py   2>&1 | tee -a logs/analysis.log
$PY report/build/build_reports.py 2>&1 | tee -a logs/analysis.log
say "FOLLOW-ON COMPLETE"
