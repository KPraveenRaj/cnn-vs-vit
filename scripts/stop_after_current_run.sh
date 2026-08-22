#!/usr/bin/env bash
# Stop all background work at the next safe boundary: when the in-flight
# training run writes its metrics.json.
#
# train.py only writes metrics.json at the END of a run, so killing mid-run
# discards everything that run has done. Waiting for the marker costs a few
# minutes and loses nothing. The battery is safe to interrupt at any point (it
# persists after every pass), so only the training boundary matters.
set -uo pipefail
cd "$(dirname "$0")/.."
TARGET=${1:?usage: stop_after_current_run.sh <run_id>}
MARK="results/runs/${TARGET}/metrics.json"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ===" | tee -a logs/pipeline.log; }

say "STOP-WATCH  waiting for ${TARGET} to finish before stopping"
while [ ! -f "$MARK" ]; do sleep 60; done
say "STOP-WATCH  ${TARGET} complete; stopping all background work"

for pat in "run_food101_battery\.sh" "run_rest_after_matrix\.sh" \
           "src\.train\.run_matrix" "src\.eval\.run_eval_battery" "src\.train\.train"; do
  for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  sleep 2
done
sleep 5
for pat in "src\.train\.train" "src\.eval\.run_eval_battery"; do
  for p in $(pgrep -f "$pat"); do kill -9 "$p" 2>/dev/null; done
done
sleep 3
say "STOP-WATCH  all stopped; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use"
