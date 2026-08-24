#!/usr/bin/env bash
# Rebuild every document from results/tables in one command.
#
# EXISTS BECAUSE: three documents went stale at different times, each because it
# was written once by hand or by a one-off script rather than by a generator that
# gets re-run. Every document is now generated, and this is the single entry point
# that regenerates all of them, so "did I rebuild everything?" has one answer.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/torch_env/bin/python}

$PY -m src.analysis.aggregate
$PY -m src.analysis.calibration
$PY -m src.analysis.error_overlap
$PY -m src.analysis.replication
$PY -m src.analysis.plots
$PY -m src.analysis.visual_assets
$PY -m src.analysis.docs_index

$PY report/build/build_decks.py
$PY report/build/build_reports.py
$PY report/build/build_submission_guide.py
$PY report/build/build_handbook.py
$PY report/build/build_status_docs.py
$PY report/build/build_weekly_note.py
echo "ALL DOCUMENTS REBUILT"
