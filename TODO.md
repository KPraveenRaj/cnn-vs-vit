# What's left

*Written 2026-08-22 09:25. The Caltech-256 study — the whole thesis — is
complete. Everything below is either the Food-101 confirmation block or
housekeeping.*

## Queued and unattended (resume with one command)

```bash
bash scripts/run_rest_after_matrix.sh    # finishes Food-101 training
bash scripts/run_food101_battery.sh      # then the battery + full refresh
```

Both skip completed work, so re-running costs only what genuinely remains.

| # | task | time | notes |
|---|---|---|---|
| 1 | Food-101 `resnet50 f25` | ~25 min | 17,675 train images |
| 2 | Food-101 `vit_b16 f25` | ~60 min | |
| 3 | Battery on 4 Food-101 checkpoints | ~155 min | test split is 20,200 images, 3.4x Caltech's |
| 4 | `src.analysis.replication` + regenerate all documents | ~10 min | the pipeline does this automatically |

**Total: ~4 h of unattended GPU time.**

`resnet50 f100` and `vit_b16 f100` are already done.

## Then: read the replication verdict

`python -m src.analysis.replication` prints one row per Caltech finding and
whether Food-101 points the same way. Also written to
`results/tables/replication.csv` and into the end-semester report (8.1) and
`SUBMISSION_GUIDE.md` (3b).

Remember the two cautions that travel with it: Food-101 is **one seed** by
design, so direction only, never significance; and disagreement would mean the
finding is *dataset-dependent*, not that the Caltech measurement was wrong.

## Housekeeping

- [ ] **Back up to the external HDD.** CLAUDE.md asks for this weekly and it has
      never been done — the drive was not mounted at any point in these sessions.
      Plug it in and copy `results/` (including the 4.3 GB of archived ep30
      checkpoints, which are local-only) and `report/`.
- [ ] **Back up tables + report to Google Drive**, also per CLAUDE.md.
- [ ] **Send `progress_01_resumption.pptx` to Dr. Bini.** It is ready and honest
      about the pause. `report/weekly_notes/week_07.md` is a drafted written note
      covering the same ground if you would rather send prose.
- [x] ~~Freeze `requirements.txt`~~ — done 2026-08-22, pinned to the exact
      versions that produced the results.

## Optional, explicitly "only if time permits" in the plan

- [ ] **Attention maps / Grad-CAM.** Qualitative only. No committed figure needs
      them, and the mechanism story is already carried quantitatively by the
      frequency probes. Lowest value of anything remaining.
- [ ] **A third seed for Food-101**, if you ever want significance there rather
      than direction. Roughly 5 GPU-h per extra seed.

## Not for this phase

- Phase II (EC790): generative / restoration-oriented vision transformers. The
  frequency machinery built here transfers directly — restoration is explicitly
  a frequency problem and the tooling is already validated.
