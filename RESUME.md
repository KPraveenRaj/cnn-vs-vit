# Project state

*Generated 2026-08-24 19:05 by `report/build/build_status_docs.py`. Do not edit by hand —
this file went stale twice when it was hand-written.*

## Status: COMPLETE

| block | state |
|---|---|
| Caltech-256 full fine-tune | **24/24** runs |
| Caltech-256 linear probes | **24/24** runs |
| Caltech-256 evaluation battery | **24/24** checkpoints |
| Food-101 full fine-tune | **6** runs (f10, f25, f100 x 2 models) |
| Food-101 evaluation battery | **6** checkpoints |
| Training compute | 15.2 GPU-hours (evaluation battery is extra) |

## Headline result

Frozen-test top-1, mean ± SD over 3 seeds, Caltech-256:

| training data | ResNet-50 | ViT-B/16 | gap (pp) |
|---|---|---|---|
| 10% | 76.18 ± 0.27 | 78.13 ± 0.45 | **+1.95** |
| 25% | 83.19 ± 0.42 | 85.19 ± 0.14 | **+1.99** |
| 50% | 86.65 ± 0.12 | 87.52 ± 0.18 | **+0.87** |
| 100% | 88.87 ± 0.04 | 89.62 ± 0.23 | **+0.76** |

## Deliverables

| file | what |
|---|---|
| `SUBMISSION_GUIDE.md` | **read first** — what to submit, when, and what is safe to claim |
| `HANDBOOK.md` | self-contained theory, method, mathematics, reproduction recipe |
| `AUDIT.md` | independent audit: 463 consistency checks + literature cross-check |
| `report/phase1/decks/` | 4 progress decks + mid-sem + end-sem, all with speaker notes |
| `report/phase1/*.docx` | mid-semester and end-semester reports |
| `results/DOCUMENTATION_INDEX.md` | every artifact, what produced it, what it is for |

All documents are **generated**. Rebuild with:

```bash
python report/build/build_decks.py
python report/build/build_reports.py
python report/build/build_submission_guide.py
python report/build/build_handbook.py
python report/build/build_status_docs.py
```

## To re-run any experiment

```bash
bash scripts/run_pipeline.sh          # skips everything already done
```
