# RESUME — where this project stands

*Generated 2026-08-21 22:03. Regenerate with the analysis pass.*

Work was paused cleanly. **Nothing is mid-write and nothing is lost.** Everything
below is on disk and committed; the only incomplete item is the evaluation
battery, which is resumable per-pass.

## One command to resume everything

```bash
bash scripts/run_rest_after_matrix.sh          # battery -> analysis -> documents -> Food-101
```

It skips all completed work automatically: `train.py` skips runs with a
`metrics.json`, and the battery skips any pass already present in a run's
`eval_results.json`. Re-running costs nothing but the work that genuinely remains.

## Done

| Block | State |
|---|---|
| Frozen splits, Caltech-256 + Food-101 | complete, committed |
| Declared LR sweeps (fine-tune + probe, per model) | complete |
| Caltech-256 full fine-tune, 2 models x 4 fractions x 3 seeds | **24/24** |
| Linear probes, same grid on cached features | **24/24** |
| Evaluation battery (42 passes/checkpoint) | **16/24** |
| Figures / tables / 6 decks / 2 reports | generated from current data |

### Headline result — frozen-test top-1, mean ± SD over 3 seeds

| fraction | ResNet-50 | ViT-B/16 | gap (pp) |
|---|---|---|---|
| f10 | 76.18 ± 0.27 | 78.13 ± 0.45 | **+1.95** |
| f25 | 83.19 ± 0.42 | 85.19 ± 0.14 | **+1.99** |
| f50 | 86.65 ± 0.12 | 87.52 ± 0.18 | **+0.87** |
| f100 | 88.87 ± 0.04 | 89.62 ± 0.23 | **+0.76** |

ViT-B/16 leads at every fraction and every gap exceeds the pooled seed spread.

## Remaining

1. **Battery: 9 checkpoints left** (~15 min each on the RTX 4060):
   - resnet50_food101_f100_s0_fullft
   - vit_b16_caltech256_f10_s1_fullft
   - vit_b16_caltech256_f10_s2_fullft
   - vit_b16_caltech256_f25_s0_fullft
   - vit_b16_caltech256_f25_s1_fullft
   - vit_b16_caltech256_f25_s2_fullft
   - vit_b16_caltech256_f50_s0_fullft
   - vit_b16_caltech256_f50_s1_fullft
   - vit_b16_caltech256_f50_s2_fullft
2. **Food-101 confirmation block** — data extracted and splits committed, runs not
   started. 2 models x {100, 25}% x 1 seed, ~5 GPU-h. Lowest priority by design.
3. **Regenerate documents** once the above land (the resume command does this).

## Decisions made along the way that you should know about

- **The LR schedule was corrected mid-project.** A 30-epoch cosine with patience-5
  early stopping terminated runs before the LR annealed, costing ViT 2.7-3.0 pp
  against ResNet's 0.2 — and inverting the headline. `base.yaml` now uses 15
  epochs / patience 8. Superseded runs are archived under
  `results/archive/ep30_truncated/` and the finding is written up in both reports
  (section 3.6) and both review decks.
- **Probe LRs are swept per model** (ResNet 1e-1, ViT 1e-3). A shared LR cost the
  ResNet probe 21 pp at f10.
- **Food-101 comes from the official ETH tarball**, not the local
  `archive_food.zip`, which is corrupt (zip64 offsets, plus 101,000 AppleDouble
  `._*.jpg` files that would have doubled the dataset with 4 KB junk).
- **`epochs: 15` is a compromise.** ViT's f100 optimum sits at a shorter budget
  (8 epochs -> 0.9037 vs 15 -> 0.8926) while f10 needs the longer one. One shared
  budget is the declared protocol; the sensitivity is documented as a limitation.

## Where things live

- `results/DOCUMENTATION_INDEX.md` — every artifact, what made it, what it is for
- `results/tables/master.csv` — the single source of truth for every number
- `report/phase1/decks/` — 4 staged progress decks + mid-sem + end-sem
- `report/phase1/*.docx` — mid-sem and end-sem reports
- Decks and reports are **generated**, never hand-edited: rebuild with
  `python report/build/build_decks.py` and `build_reports.py`.
