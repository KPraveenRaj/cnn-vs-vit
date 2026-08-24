# Independent audit of this project's findings

*2026-08-24. A deliberate attempt to find errors: internal consistency checks over
every result file, then a cross-check of each claim against the published
literature. Written before any conclusion was adjusted.*

**Summary: no computational errors found. One genuine confound found that
narrows a headline claim, plus three interpretation caveats that belong in the
report. Details below.**

---

## A. Internal consistency — 463 automated checks

Every result file was checked for properties that must hold if the code is
correct.

| check | result |
|---|---|
| top-5 ≥ top-1, all 54 evaluated runs | pass |
| all metrics within [0,1] | pass |
| corruption accuracy falls from severity 1 → 5 | pass |
| corruption severity 1 ≤ clean | pass |
| accuracy increases with data fraction, every model/dataset/regime | pass |
| seeds produce genuinely different results (seeding not broken) | pass |
| ECE ≤ MCE (MCE is the maximum gap, so this is definitional) | pass |
| κ within [-1,1]; outcome shares sum to 1 | pass |
| oracle top-1 ≥ best single model | pass |
| **low-pass at r=159 reproduces clean accuracy** (the identity anchor) | pass, ≤ 4 images |

### The three apparent violations, and why none is a defect

**1. Identity anchor off by 0–4 images.** The filtered path performs an extra FFT
round-trip in float32 (~1e-7 residue) plus a clamp to [0,1], which the clean path
does not. An image sitting exactly on a decision boundary can flip. Maximum
observed deviation 4 images of 5,952. Expected numerical behaviour.

**2. High-pass "non-monotonicity", up to 53 images.** Every instance occurs at
accuracy ≈ 0.010 on Food-101, where chance is 1/101 = 0.0099. These are
fluctuations around chance level, not violations of a trend.

**3. Low-pass "non-monotonicity" at r=112 → r=159, up to 17 images.** Accuracy
falls 0.29 pp when the final corner frequencies are restored. Since r=159 is the
identity, this says mild low-pass filtering slightly *improves* accuracy — a real
and well-known denoising effect, not an error.

---

## B. Cross-check against the published literature

### B1. Park & Kim (ICLR 2022): "MSAs are low-pass filters, Convs are high-pass"

**Our results are consistent with this.** Four independent lines agree:

| our measurement | value | implication | consistent? |
|---|---|---|---|
| ViT retention under high-frequency noise | 0.961–0.987 across all conditions | attenuates high frequency, so barely perturbed by it | yes |
| ViT low-pass AUC vs ResNet | 0.817 vs 0.789 | works better when only low frequencies survive | yes |
| ResNet retention under high-frequency noise, Food-101 | 0.078 | depends heavily on high-frequency content | yes |
| ResNet low-pass accuracy at r=16 | 0.433 vs ViT's 0.549 | suffers more when high frequency is removed | yes |

**A distinction worth preserving:** Park & Kim characterise what the *operations*
do to feature maps. This project measures *input-frequency robustness*. These are
related but not identical, and the report should not claim to have replicated
their measurement — only that the two are consistent.

### B2. ViT robustness literature — one apparent contradiction, examined

Naseer et al. (NeurIPS 2021) and Paul & Chen (AAAI 2022) report ViTs as more
robust to common corruptions than CNNs. **Consistent with ours.**

However, **Bhojanapalli et al. (2021) report that with ImageNet-1k pre-training
specifically, ViTs were *worse* than CNNs on corruption robustness**, converging
only with ImageNet-21k or JFT-300M pre-training. We used ImageNet-1k and found
the ViT substantially *more* robust (16.7% vs 27.3% mean relative drop).

Two differences plausibly account for this, and both belong in the report:

1. **Different evaluation setting.** They evaluate ImageNet-trained models on
   ImageNet-C. We fine-tune on a *different* target dataset and corrupt that.
   Robustness after transfer is not the same quantity as robustness in-domain.
2. **Pre-training recipe — see the confound below.**

### B3. THE CONFOUND: pre-training data was controlled; pre-training RECIPE was not

This is the most substantive finding of the audit.

The protocol fixes the pre-training *source* (ImageNet-1k for both), which is the
control most published comparisons get wrong, and getting it right matters. But
the two checkpoints come from **different training recipes**:

| | ResNet-50 `a1_in1k` | ViT-B/16 `augreg_in1k` |
|---|---|---|
| origin | Wightman, *ResNet strikes back* (2021) | Steiner et al., *How to train your ViT?* (2021) |
| approach | LAMB optimiser, BCE loss, mixup/CutMix, RandAugment, ~600 epochs | "AugReg" — strong augmentation plus heavy regularization |
| purpose of recipe | modernise ResNet training | compensate for ViT's weak inductive bias |

Both use heavy augmentation, so this is not a naive mismatch. But **augmentation
strength and regularization during pre-training are known to affect corruption
robustness substantially**, and the recipes differ in optimiser, loss function,
augmentation composition, epoch count and regularization strength.

**Consequence.** The robustness gap reported here may partly reflect the
pre-training recipe rather than the architecture. The data-efficiency result is
much less exposed — it concerns how quickly each model adapts, not what its
pre-training instilled — but the corruption and frequency results are.

**This does not invalidate anything measured.** It bounds the claim. The
defensible statement is *"these two publicly released ImageNet-1k checkpoints,
fine-tuned identically, differ in this way"* rather than *"CNNs and ViTs
intrinsically differ in this way"*.

**Fixing it properly** would require pre-training both architectures from scratch
under one recipe — far beyond a laptop GPU and this project's scope. It is
recorded as a limitation and is a natural Phase-II question.

### B4. Band-noise interpretation and the 1/f spectrum

The band-noise probe holds noise RMS constant across bands. Natural images have
approximately 1/f² power spectra, so image energy is heavily concentrated at low
frequency. Fixed noise RMS therefore does **not** mean fixed signal-to-noise
ratio per band — the same noise is a much larger *relative* perturbation at high
frequency, where there is little signal.

Both models face identical noise, so **model-versus-model comparison remains
valid**. But statements of the form "model X relies most on band B" are
complicated by this, and the report should say so rather than reading the curve
minimum as a pure reliance measure.

### B5. Absolute accuracy plausibility

| result | ours | comment |
|---|---|---|
| ResNet-50, Caltech-256, full fine-tune | 88.87% | in the expected range for ImageNet-pretrained ResNet-50 at 224 |
| ViT-B/16, Caltech-256 | 89.62% | plausible |
| ResNet-50, Food-101 | 82.75% | below the ~85–88% commonly reported, consistent with our deliberately short 15-epoch budget and modest augmentation |
| ViT-B/16, Food-101 | 84.85% | same |
| ResNet-50 FLOPs | 4.09 GMACs | matches the published ~4.1 |
| ViT-B/16 FLOPs | 16.85 GMACs | matches the published ~17.6 GFLOPs |

Nothing anomalous. The Food-101 figures being a few points below published
fine-tuning results is expected and explained by the training budget.

---

## C. Qualitative saliency: attempted, measured, not reported

Both standard methods are degenerate for ViT-B/16 on these checkpoints:

- **Attention rollout** — class-token map has border-to-centre ratio **1.03**,
  i.e. uniform. Twelve rounds of residual mixing destroy spatial selectivity.
- **Grad-CAM** — map standard deviation **0.0000** on every image tested. Grad-CAM
  assumes non-negative activations (true after a CNN's ReLU); transformer tokens
  are LayerNorm'd and roughly zero-mean, so the channel-weighted sum is mostly
  negative and the ReLU zeroes it.

Raw last-layer attention *is* structured (ratio 3.08) but concentrates on image
borders — the documented attention-sink / register-token behaviour (Darcet et
al., 2023). A real property of the model, and exactly why raw attention is a poor
saliency method here.

Grad-CAM on ResNet-50 works correctly (std 0.22–0.29) and localises objects
sensibly, but a one-sided saliency figure in a two-model comparison would be half
an argument. **Nothing is reported.** No conclusion in this project depends on a
saliency picture.

---

## D. What changed as a result of this audit

1. The saliency figure was **deleted** rather than shipped, and the negative
   result documented in `src/analysis/attention_maps.py`.
2. The pre-training-recipe confound (B3) is added to the limitations of both
   reports and the handbook.
3. The band-noise 1/f caveat (B4) is added alongside the frequency method.
4. The distinction between Park & Kim's measurement and ours (B1) is stated
   rather than glossed.

## E. What the audit did NOT find

No computational errors. No metric implemented incorrectly. No data leakage — the
test split is frozen, committed, and every hyperparameter was chosen on
validation. No seed reuse. No stale numbers in the generated documents. The
headline data-efficiency result survives scrutiny unchanged, and is corroborated
on a second dataset.
