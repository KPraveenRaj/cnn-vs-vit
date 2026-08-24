# What to submit, and what it says

*Generated 2026-08-24 19:36 from `results/tables/`. Re-run `python report/build/build_submission_guide.py` after new results land.*

Read this first. It maps every artifact to when you would use it, states the results in language you can defend, and marks the few things you should NOT claim.

## 1. The result, in one table

Caltech-256, 256 classes, 5,952 frozen test images. Frozen-test top-1, mean ± SD over 3 seeds. Full fine-tuning.

| training data | ResNet-50 | ViT-B/16 | gap (pp) |
|---|---|---|---|
| **10%** | 76.18 ± 0.27 | 78.13 ± 0.45 | **+1.95** |
| **25%** | 83.19 ± 0.42 | 85.19 ± 0.14 | **+1.99** |
| **50%** | 86.65 ± 0.12 | 87.52 ± 0.18 | **+0.87** |
| **100%** | 88.87 ± 0.04 | 89.62 ± 0.23 | **+0.76** |

**What you can say:** under a single controlled transfer-learning protocol from ImageNet-1k, ViT-B/16 outperforms ResNet-50 at every training-set size, and its advantage is largest when data is scarce. Every gap is larger than the seed-to-seed spread, so none of them is noise.

**Why that is interesting:** the folk claim is that ViTs are data-hungry and lose to CNNs in low-data regimes. That claim is about training from scratch. Under *transfer learning*, where both models arrive pre-trained, the ordering reverses — the transformer's representation transfers better precisely when there is least data to adapt it with.

**What you must NOT say:** that this generalises beyond these two architectures, this dataset, and this protocol. You measured two archetypes on one ImageNet-adjacent dataset.

### Linear probes (frozen features, no fine-tuning)

| training data | ResNet-50 | ViT-B/16 |
|---|---|---|
| **10%** | 74.34 | 76.59 |
| **25%** | 82.20 | 81.37 |
| **50%** | 85.75 | 83.80 |
| **100%** | 87.84 | 85.84 |

Note the **inversion**: ViT's frozen features win at 10% data, but ResNet's win at 25% and above — the opposite ordering to full fine-tuning. Worth a sentence in the discussion: the transformer's advantage under fine-tuning is not simply that its frozen features are better; it is that they *adapt* better.

## 2. The mechanism — why ViT is more robust

The evaluation battery has run on **24 of 24** Caltech-256 checkpoints and **6 of 6** Food-101 checkpoints, 42 inference passes each.

Under equal-energy noise confined to one frequency band (f100):

| noise band (DFT bins) | 0–8 | 8–16 | 16–32 | 32–56 | 56–88 | 88–159 |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.609 | 0.384 | 0.416 | 0.627 | 0.775 | 0.747 |
| ViT-B/16 | 0.739 | 0.735 | 0.699 | 0.789 | 0.858 | 0.886 |

**The story:** ResNet-50 has a sharp low-frequency vulnerability — it collapses in the 8–16 bin band. ViT-B/16's profile is far flatter with no comparable weak band. That difference *predicts* the corruption results: ViT degrades far more gracefully under Gaussian noise, blur and JPEG, and the margin widens with severity.

This is what turns the project from a benchmark into a characterisation. You are not just reporting that one model is more robust; you are showing which part of the input spectrum each family depends on, and using it to explain the robustness ordering.

## 2b. YOUR CONTRIBUTION — the one result that is genuinely new

Everything above (ViT more accurate, ViT more robust) is a good controlled replication. **This is the part that is yours.**

Change in relative retention per frequency band between 10% and 100% training data, averaged over 3 seeds. Positive = the model got more robust in that band as data grew:

| model | 0-8 | 8-16 | 16-32 | 32-56 | 56-88 | 88-159 |
|---|---|---|---|---|---|---|
| **ResNet-50** | +0.019 | +0.015 | +0.021 | +0.060 | +0.124 | +0.263 |
| **ViT-B/16** | -0.008 | +0.017 | +0.008 | +0.022 | +0.016 | +0.010 |

**ResNet-50's spectral robustness profile moves with the data budget — +0.263 in the 88-159 bin band, and the movement is concentrated at high frequency. ViT-B/16's is essentially invariant (+0.022). A 12x difference.**

**⚠ Read the warning below before using this.** The numbers above are Caltech-256 only, and this specific claim did **not** reproduce on Food-101. The defensible version is in the box further down.

**Why this would matter — on Caltech-256:** it is a *mechanism* for the data-efficiency result, not a restatement of it. ViT's advantage is largest exactly where the CNN has least data from which to learn what the ViT already has. Three axes — accuracy, robustness, frequency — become one story. **On Food-101 that chain does not hold**, so present the mechanism as a Caltech-256 observation and the invariance contrast (below) as the cross-dataset claim.

**How it differs from Park & Kim (2022),** which your guide will probably raise: they show the two families differ in frequency response at full scale and largely from scratch. The extension here is that the difference is *itself contingent* — on the data budget (Caltech-256) and on the dataset (it vanishes on Food-101) — which is only visible under a protocol that varies the data budget while holding everything else fixed. Note also that they characterise what the *operations* do to feature maps, whereas this measures *input-frequency robustness*: related, not the same measurement.

Figure: `results/figures/fig_frequency_shift.pdf`. Deck slide: "Frequency reliance × data fraction" in both review decks. Report: §6.4.

> **⚠ IMPORTANT — this one did not reproduce on Food-101.**
>
> | test | Caltech-256 | Food-101 |
> |---|---|---|
> | ResNet's profile shifts MORE than ViT's (max over bands) | 0.263 vs 0.022 (f10->f100) | 0.061 vs 0.110 (f10->f100) |
> | ResNet's HIGH-frequency robustness improves more with data than ViT's | +0.263 vs +0.010 | -0.011 vs +0.019 |
> | ViT's high-frequency robustness is INVARIANT across dataset and data budget; ResNet's is contingent | ResNet 0.078-0.878 (11.2x, both datasets) | ViT 0.961-0.987 (1.03x, both datasets) |
>
> **Do not present this as a general result.** The defensible phrasing is: *on Caltech-256, the CNN's spectral robustness is data-dependent and the transformer's is not; this did not reproduce on Food-101 over the range tested.*
>
> Three points to have ready, because they are real and an examiner may not think of them:
> 1. Food-101 runs **one seed**, so a single noisy band can dominate the statistic and nothing can be said about significance.
> 2. Caltech's shift is **structured** — it grows monotonically with frequency. Food-101's bounces in sign, which is what one seed of noise looks like.
> 3. In the high band specifically, both Food-101 values are ~0. The honest description is *neither model moved*, not *the effect reversed*.
>
> Also: the fractions match in proportion but not absolute size. Food-101's 10% is 7,070 images against Caltech's 2,196 — it may simply sit past the data-starved regime the effect lives in.
>
> **Volunteering this is much stronger than being caught by it**, and it sets up Phase II with a concrete open question.

### What DOES survive both datasets — use this framing instead

Accuracy retained under high-frequency noise, every condition tested:

| model | range across all conditions | spread |
|---|---|---|
| **ResNet-50** | 0.078 – 0.878 | **11.2×** |
| **ViT-B/16** | 0.961 – 0.987 | **1.0×** |

**ViT-B/16's spectral robustness is invariant (1.03× spread). ResNet-50's is contingent on both task and data budget (11.2× spread).** That holds across every dataset and data size measured.

On Food-101 the CNN retains under a tenth of its accuracy against high-frequency noise at *every* data size and never improves. On Caltech it improves a lot. So the *direction* is dataset-specific; the *stability contrast* is not.

**Say it like this:** *"The transformer's robustness to fine detail is the same whatever we train it on and however much data we give it. The CNN's depends on both — and on one dataset it never gets there at all."*

**And be upfront that this framing came after seeing both datasets**, not before. It is a well-supported hypothesis, not a pre-registered result. Saying so costs nothing and is exactly the judgement an examiner is looking for.

## 3. The methodology finding (your best viva material)

Mid-project, the declared schedule turned out to be self-defeating: a 30-epoch cosine with patience-5 early stopping terminated runs while the learning rate was still high, before the annealing phase that converges the model.

| model | 30-ep truncated | 8-ep annealed | 15-ep annealed | runs early-stopped |
|---|---|---|---|---|
| ResNet-50 | 89.20 | 89.43 | 89.73 | 8 of 12 |
| ViT-B/16 | 87.34 | 90.37 | 89.26 | 10 of 10 |

Measured against the protocol actually adopted, it cost the transformer +1.92 pp and the CNN +0.54 pp — a 3.6x asymmetry. (Against the ViT's best annealed run it looks like 12.9x, but that compares to a schedule we did not adopt — do not quote it.) Under the broken protocol the measured gap decayed and reversed (+1.38 → +0.91 → −0.45 pp); under the corrected one it never crosses. **Same data, same seeds — the only difference was whether the cosine was allowed to finish.**

Present this deliberately. It demonstrates that you understand the difference between an architectural result and a training artefact, and it is the honest answer to *"why does your protocol say 15 epochs?"*. The superseded runs are kept in `results/archive/ep30_truncated/` as evidence.

## 3b. Does Food-101 back it up?

| finding | Caltech-256 | Food-101 | verdict |
|---|---|---|---|
| ViT-B/16 beats ResNet-50 at 25% data | +1.99 pp | +4.87 pp | **replicates** |
| ViT-B/16 beats ResNet-50 at 100% data | +0.76 pp | +2.10 pp | **replicates** |
| Gap narrows from 25% to 100% data | -1.24 pp change | -2.76 pp change | **replicates** |
| ViT-B/16 degrades less under corruption | 27.3% vs 16.7% drop | 56.5% vs 35.9% drop | **replicates** |
| ViT-B/16 retains more accuracy under low-pass filtering | 0.817 vs 0.789 | 0.765 vs 0.686 | **replicates** |
| ResNet's profile shifts MORE than ViT's (max over bands) | 0.263 vs 0.022 (f10->f100) | 0.061 vs 0.110 (f10->f100) | **differs** |
| ResNet's HIGH-frequency robustness improves more with data than ViT's | +0.263 vs +0.010 | -0.011 vs +0.019 | **differs** |
| ViT's high-frequency robustness is INVARIANT across dataset and data budget; ResNet's is contingent | ResNet 0.078-0.878 (11.2x, both datasets) | ViT 0.961-0.987 (1.03x, both datasets) | **replicates** |

**How to talk about this.** Food-101 runs one seed by design, so it supports claims about *direction*, never significance. And if something differs, that is not evidence the Caltech work is wrong — the Caltech result stands on its own internal validity (frozen split, three seeds, one protocol). Disagreement would mean the finding is dataset-dependent, which is a legitimate and arguably more interesting result: it would say the CNN-versus-ViT ordering depends on how close the target task sits to the pre-training distribution.

The thing that would genuinely invalidate the work is a protocol bug — which is why the schedule interaction in section 3 was worth chasing.

## 4. Which artifact to use when

### Progress updates for Dr. Bini — send one at a time

These are deliberately **undated**: each describes a milestone, so send it when that milestone is what you want to report.

| file | slides | what it says | send when |
|---|---|---|---|
| `progress_01_resumption.pptx` | 8 | Work paused and has resumed; framework complete; CNN arm fully run with the declared LR sweep | **now** — it is the honest stall-and-resumed update |
| `progress_02_vit_arm.pptx` | 6 | ViT sweep, both arms complete, the head-to-head data-efficiency table, deployment cost | after she has seen #1 |
| `progress_03_robustness.pptx` | 7 | What the battery does, corruption results, how the FFT probes are built, frequency curves | once the battery is complete |
| `progress_04_mechanism.pptx` | 5 | The interaction figure, error overlap, calibration, status against plan | just before mid-sem |

### Formal submissions

| file | what it is |
|---|---|
| `decks/midsem_review.pptx` (13 slides) | Mid-semester presentation. Carries the whole story: motivation, protocol, the schedule finding, data efficiency, corruption, frequency, the interaction figure, overlap, calibration, cost, and what remains. |
| `midsem_report.docx` | Mid-semester report, same arc in prose with an explicit "work remaining" section. |
| `decks/endsem_review.pptx` (15 slides) | End-semester presentation. Adds Food-101, a limitations slide, and the Phase-II direction. |
| `endsem_report.docx` | End-semester report. Adds the Food-101 section, limitations, and the EC790 plan. |

All eight are **generated**. Never hand-edit them — edit `report/build/build_decks.py` or `build_reports.py` and rebuild, or your changes vanish on the next run.

## 5. Questions you will be asked

**How are the data fractions built?**

Per-class and nested: f10 ⊂ f25 ⊂ f50 ⊂ f100, with three independent seed nestings. So varying the fraction varies quantity alone, never composition.

**Did both models get identical hyperparameters?**

Controlled data and protocol, with declared per-model tuning. Everything is identical except the learning rate, which is chosen for each model by a documented sweep — forcing one LR on both would cripple one family and invalidate the comparison. The ViT grid sits a decade below the CNN's, and that gap is the evidence the tuning was necessary.

**Park & Kim already showed CNNs and ViTs differ in frequency response.**

At full scale and largely from scratch. This measures how that dependence shifts with the transfer-learning data fraction, and whether it predicts low-data robustness. The interaction is the contribution.

**Isn't this just benchmarking?**

No — the frequency probes give a mechanism. The band-noise profile explains the corruption ordering rather than merely accompanying it.

**Why 15 epochs?**

Because at 30 the cosine never annealed before early stopping fired, which cost the ViT 3 pp and the CNN 0.2 and inverted the headline. Measured, archived, and reported in section 3.6.

**How do you know the FFT filtering is correct?**

Three assertions in the module's self-test: the low-pass mask at the corner radius is the identity (4e-07), low-pass plus high-pass reconstructs the original, and every noise band carries identical spatial RMS with no spectral leak. The identity anchor also holds on real checkpoints — low-pass at r=159 reproduces clean accuracy exactly.

**Why is the test set trustworthy?**

Generated once, committed to git, and never used for any decision — every learning rate was selected on validation only.

## 6. What is not finished

**Nothing.** Every committed experiment is complete: Caltech-256 (24 fine-tuning runs, 24 linear probes, 24 batteries) and Food-101 (6 runs, 6 batteries).

Two things are deliberately NOT done. Both are documented rather than missing, and saying so is stronger than leaving a gap:

- **Qualitative saliency (attention maps / Grad-CAM).** Attempted and measured degenerate for ViT-B/16: attention rollout gives a uniform map (border-to-centre ratio 1.03) and Grad-CAM gives map standard deviation 0.0000, because Grad-CAM assumes non-negative activations that LayerNorm'd transformer tokens do not provide. Kept as a documented negative result in `src/analysis/attention_maps.py`, reproducible with `--diagnose`. No conclusion in the project depends on a saliency picture.
- **Controlling the pre-training recipe.** Both checkpoints are ImageNet-1k, but they come from different training recipes, which the audit flags as a confound for the robustness result. Settling it would require pre-training both architectures from scratch under one recipe — far beyond a laptop GPU, and a natural Phase-II question.

Everything is generated from one results table and rebuilds with one command per document.
