# What is left

*Generated 2026-08-24 19:00 by `report/build/build_status_docs.py`.*

## Experiments: all complete

Nothing outstanding. Caltech-256 is 24/24 runs,
24/24 probes and 24/24 batteries; Food-101 is
6 runs and 6 batteries.

## Housekeeping

- **Back up to the external HDD.** CLAUDE.md asks weekly; it has never been done because the drive was not mounted in any session. `results/archive/ep30_truncated/checkpoints/` (4.3 GB) exists ONLY on this machine — everything else is on GitHub.
- **Back up tables and report to Google Drive**, also per CLAUDE.md.
- **Send a progress deck to Dr. Bini.** `progress_01_resumption.pptx` is ready, or `report/weekly_notes/week_07.md` if prose suits better.
- [x] ~~Freeze `requirements.txt`~~ — done, pinned to the versions that produced
  the results.
- [x] ~~Qualitative saliency~~ — attempted, measured degenerate for ViT-B/16, and
  documented as a negative result in `src/analysis/attention_maps.py`. Not reported.

## Open questions for Phase II

- **Does the invariance claim hold on a third dataset?** The narrow contribution
  held on Caltech-256 and not on Food-101; the rescoped version (ViT's spectral
  robustness invariant, ResNet's contingent) holds across every condition measured
  here but was formed after seeing both datasets.
- **Controlling the pre-training recipe.** Both checkpoints are ImageNet-1k but
  come from different training recipes, which the audit identifies as a confound
  for the robustness result. Settling it needs both architectures pre-trained from
  scratch under one recipe.
