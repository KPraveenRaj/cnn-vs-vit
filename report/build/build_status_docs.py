"""Regenerate TODO.md and RESUME.md from what is actually on disk.

Both were hand-written and both went stale within two days — TODO.md still listed
Food-101 as pending after it had finished, and RESUME.md still reported a
16-of-24 battery. Anything that describes project state has to be generated from
the state, for the same reason no number in a report is typed by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import Facts

REPO_ROOT = Path(__file__).resolve().parents[2]


def counts():
    runs = REPO_ROOT / "results" / "runs"
    out = {}
    for ds, total in (("caltech256", 24), ("food101", 6)):
        ft = [p for p in runs.iterdir()
              if f"_{ds}_" in p.name and p.name.endswith("_fullft")
              and "sweep" not in p.name and "diag" not in p.name
              and (p / "metrics.json").exists()]
        batt = [p for p in ft if (p / "eval_results.json").exists()]
        out[ds] = (len(ft), len(batt), total)
    probes = [p for p in runs.iterdir() if p.name.endswith("_linprobe")
              and (p / "metrics.json").exists()]
    out["probes"] = len(probes)
    return out


def main():
    f = Facts()
    c = counts()
    av = f.available()
    eff = "\n".join(
        f"| {fr}% | {f.top1('resnet50', fr)[0]*100:.2f} ± {f.top1('resnet50', fr)[1]*100:.2f} "
        f"| {f.top1('vit_b16', fr)[0]*100:.2f} ± {f.top1('vit_b16', fr)[1]*100:.2f} "
        f"| **{f.gap_at(fr):+.2f}** |" for fr in (10, 25, 50, 100))
    stamp = __import__("time").strftime("%Y-%m-%d %H:%M")

    done = all([c["caltech256"][0] == 24, c["caltech256"][1] == 24, c["probes"] == 24,
                c["food101"][0] >= 4, c["food101"][1] >= 4])

    resume = f"""# Project state

*Generated {stamp} by `report/build/build_status_docs.py`. Do not edit by hand —
this file went stale twice when it was hand-written.*

## Status: {"COMPLETE" if done else "IN PROGRESS"}

| block | state |
|---|---|
| Caltech-256 full fine-tune | **{c['caltech256'][0]}/24** runs |
| Caltech-256 linear probes | **{c['probes']}/24** runs |
| Caltech-256 evaluation battery | **{c['caltech256'][1]}/24** checkpoints |
| Food-101 full fine-tune | **{c['food101'][0]}** runs (f10, f25, f100 x 2 models) |
| Food-101 evaluation battery | **{c['food101'][1]}** checkpoints |
| Total compute | {f.gpu_hours()} GPU-hours |

## Headline result

Frozen-test top-1, mean ± SD over 3 seeds, Caltech-256:

| training data | ResNet-50 | ViT-B/16 | gap (pp) |
|---|---|---|---|
{eff}

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
"""
    (REPO_ROOT / "RESUME.md").write_text(resume)

    remaining = []
    if not av["food101"]:
        remaining.append("- Food-101 confirmation block")
    remaining += [
        "- **Back up to the external HDD.** CLAUDE.md asks weekly; it has never been "
        "done because the drive was not mounted in any session. "
        "`results/archive/ep30_truncated/checkpoints/` (4.3 GB) exists ONLY on this "
        "machine — everything else is on GitHub.",
        "- **Back up tables and report to Google Drive**, also per CLAUDE.md.",
        "- **Send a progress deck to Dr. Bini.** `progress_01_resumption.pptx` is ready, "
        "or `report/weekly_notes/week_07.md` if prose suits better.",
    ]
    todo = f"""# What is left

*Generated {stamp} by `report/build/build_status_docs.py`.*

## Experiments: {"all complete" if done else "in progress"}

{"Nothing outstanding. " if done else ""}Caltech-256 is {c['caltech256'][0]}/24 runs,
{c['probes']}/24 probes and {c['caltech256'][1]}/24 batteries; Food-101 is
{c['food101'][0]} runs and {c['food101'][1]} batteries.

## Housekeeping

{chr(10).join(remaining)}
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
"""
    (REPO_ROOT / "TODO.md").write_text(todo)
    print(f"[status] RESUME.md + TODO.md regenerated ({'complete' if done else 'in progress'})")


if __name__ == "__main__":
    main()
