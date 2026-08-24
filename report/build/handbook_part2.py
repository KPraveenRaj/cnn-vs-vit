"""Part 2 of HANDBOOK.md: protocol, evaluation mathematics, and the figures."""

def sections(f, d):
    rlr, vlr = d["rlr"], d["vlr"]
    classes, total = d["classes"], d["total"]
    train, val, test = d["train"], d["val"], d["test"]
    epochs, patience = d["epochs"], d["patience"]
    return f"""
---

## 4. The controlled protocol

### 4.1 What is held fixed, and what is not

**Identical for both models:** pre-training source (ImageNet-1k), data splits,
augmentation pipeline, input resolution (224×224), schedule *shape*, optimiser
family, random seeds, and the entire evaluation path.

**One declared exception:** each model gets its own learning rate, chosen by a
documented search. This is not a loophole — it is required for fairness. The two
architectures' optima sit a decade apart:

| model | fine-tuning LR | probe LR |
|---|---|---|
| ResNet-50 | {rlr} | 1e-1 |
| ViT-B/16 | {vlr} | 1e-3 |

Forcing one value on both would cripple whichever model it suited less, and the
result would measure that arbitrary choice rather than the architecture. The
phrase used throughout is **"controlled data and protocol, with declared
per-model tuning."**

Per-model normalisation constants (each backbone's own pre-training mean/std) are
a second declared and symmetric difference.

### 4.2 Data and splits

**Caltech-256** (primary): {classes} categories after excluding the `clutter`
class, {total:,} images.

| split | images | purpose |
|---|---|---|
| train | {train:,} | learning |
| validation | {val:,} | choosing hyperparameters, early stopping |
| **test** | **{test:,}** | **frozen — never used for any decision** |

Split 70/10/20, stratified per class, from one master permutation with a fixed
seed. Generated once by `src/data/make_splits.py`, committed to version control,
and never regenerated.

**Nested fractions.** To measure data efficiency, training subsets of 10%, 25%,
50% and 100% are drawn **per class** and **nested**:

    f10 ⊂ f25 ⊂ f50 ⊂ f100

Nesting matters. If each fraction were an independent random draw, a difference
between fractions could come from the *amount* of data or from having drawn easier
images. Nesting removes the ambiguity: only quantity varies. Three independent
seed nestings give the error bars.

Nesting holds because `ceil` is monotone: for one fixed per-class permutation, the
first `ceil(f·n)` items for growing f are supersets of each other.

**Food-101** (confirmation): 101 categories × 1000 images = 101,000, split the
same way. Note that Food-101 ships its own official split; this project applies
its own 70/10/20 instead, because the study compares two models *to each other*
under one protocol rather than to published leaderboard numbers.

![Nested fractions](results/figures/assets/fraction_nesting.png)

![Class distribution](results/figures/assets/class_distribution.png)

Caltech-256 is long-tailed, which is why macro-F1 and worst-decile class accuracy
are reported alongside top-1.

### 4.3 Augmentation

One modest pipeline, identical for both models:

- `RandomResizedCrop(224, scale=0.5–1.0)`
- horizontal flip (p = 0.5)
- light colour jitter (brightness/contrast/saturation 0.2, hue 0)

Heavy augmentation (mixup, CutMix, RandAugment) is **deliberately excluded**. It
interacts strongly with architecture — transformers typically benefit more — so
including it would reintroduce exactly the confound this study removes.

![Augmentation](results/figures/assets/augmentation_samples.png)

Evaluation uses a deterministic path with no randomness: `Resize(256)` then
`CenterCrop(224)`.

### 4.4 Optimisation

**AdamW.** Adam with *decoupled* weight decay. Adam maintains running estimates of
the gradient's mean and variance:

    m_t = β₁·m_{{t−1}} + (1−β₁)·g_t
    v_t = β₂·v_{{t−1}} + (1−β₂)·g_t²
    θ_t = θ_{{t−1}} − η·( m̂_t / (√v̂_t + ε)  +  λ·θ_{{t−1}} )

where m̂ and v̂ are m and v corrected for their bias toward zero in the first few
steps, and λ is the weight decay. Each parameter effectively gets its own learning
rate, scaled down where gradients have been large. The λ·θ term is what makes this
AdamW rather than Adam: the shrinkage is applied *directly to the weights*
(decoupled) instead of being folded into the gradient, where Adam's per-parameter
scaling would distort it. Weight decay λ = 0.05.

**Cosine schedule with linear warmup.** The learning rate starts near zero, rises
linearly over 3 epochs, then follows

    η_t = η_max · ½ · (1 + cos(π · t / T))

decaying smoothly to zero at the end of training. Here t counts optimiser steps
*after* warmup and T is the total number of post-warmup steps, which is how
`src/train/train.py` implements it — the cosine begins where the warmup ends
rather than at step zero. Warmup avoids destabilising
pre-trained weights with large early steps; the cosine decay lets the model
explore early and consolidate late.

**This schedule caused the project's most serious bug — see §10.1.** The final
settings are {epochs} epochs with early-stopping patience {patience}, chosen so
the cosine actually completes.

**Effective batch size 64** for both models, reached by gradient accumulation:
gradients from several micro-batches are summed before an optimiser step.

    effective_batch = micro_batch × accumulation_steps
    ResNet-50: 64 × 1     ViT-B/16: 32 × 2

The split differs because the ViT needs more memory per image; the *effective*
batch, and therefore the optimisation, is identical.

**Mixed precision (AMP).** Forward and backward passes run in 16-bit floating
point where safe, with a 32-bit master copy of the weights. This roughly halves
*activation* memory — weights, gradients and optimiser state remain 32-bit, so
total footprint falls by less than half — and speeds up matrix multiplication on
tensor cores. A gradient scaler multiplies the loss before
backpropagation to stop small gradients underflowing to zero in fp16.

**Determinism.** `src/utils/seed.py` seeds Python, NumPy, PyTorch and CUDA, sets
cuDNN deterministic mode, disables benchmark autotuning, and seeds each dataloader
worker. A rerun reproduces the same numbers.

---

## 5. Evaluation

### 5.1 Clean metrics

- **Top-1 accuracy** — fraction of images whose highest-scoring class is correct.
- **Top-5 accuracy** — fraction where the correct class is among the five highest.
- **Macro-F1** — F1 computed per class and then averaged *unweighted*, so a rare
  class counts as much as a common one. With per-class precision P_c and recall
  R_c:

      F1_c = 2·P_c·R_c / (P_c + R_c),    macro-F1 = (1/C)·Σ_c F1_c

  On a long-tailed dataset, top-1 can look healthy while rare classes are being
  abandoned; macro-F1 catches that.
- **Worst-decile class accuracy** — mean accuracy over the worst 10% of classes.

Every evaluation also writes a **per-image prediction table** (filepath, true
label, prediction, confidence), which is the raw material for the CPU-only
analyses in §5.5–5.6.

### 5.2 Corruption robustness

Three families at five severities each (ImageNet-C convention, Hendrycks &
Dietterich, 2019):

| corruption | severity 1 → 5 |
|---|---|
| Gaussian noise | σ = 0.08, 0.12, 0.18, 0.26, 0.38 |
| Gaussian blur | σ = 1, 2, 3, 4, 6 pixels |
| JPEG | quality = 25, 18, 15, 10, 7 |

![Corruption ladder](results/figures/assets/corruption_ladder.png)

**Where the corruption is applied matters.** It goes on the 224×224 crop, in
[0,1], *before* normalisation. Two reasons:

1. Corrupting before the resize would let the resize filter some of the damage
   away, so effective severity would depend on the source image's original size.
2. Normalisation constants differ per model. Perturbing after normalisation would
   make the *physical* severity model-dependent — the two architectures would no
   longer be seeing the same degraded image.

**Determinism.** The random draw for a given (corruption, severity, image) is
derived from those three things alone via CRC32 — not Python's `hash()`, which is
salted per process. So both models are scored on byte-identical corrupted pixels,
and a rerun reproduces them exactly. Corrupted images are never written to disk.

### 5.3 Frequency probes — the mathematics

This is the mechanism half of the study, and the part most worth understanding.

**The 2-D DFT.** Any image can be decomposed into spatial-frequency components —
the visual analogue of splitting audio into bass and treble. Low frequencies carry
broad shapes and layout; high frequencies carry edges, texture and fine detail.

For one image channel x of size N×N, the discrete Fourier transform is

    X[u, v] = Σ_i Σ_j  x[i, j] · exp(−2πi·(ui + vj)/N)

`torch.fft.fft2` returns X with DC (the zero-frequency, average-brightness term)
at index [0,0], which makes "distance from DC" awkward. `fftshift` moves DC to the
centre, after which the frequency index along each axis is

    u_i = i − N/2,    u ∈ [−N/2, N/2 − 1]      (N = 224 → −112 … 111)

and the **radial frequency** of bin (i, j), in DFT bins, is

    r_ij = √(u_i² + u_j²)

Radius is reported both in bins and normalised by the axis Nyquist N/2 = 112, so
r_norm = 1.0 is Nyquist along an axis. Because the grid is square but r is radial,
the corners reach r_norm = √2 ≈ 1.414.

**Ideal filters.** A binary mask on that radius:

    low-pass(r_c):   M = 1 if r ≤ r_c, else 0
    high-pass(r_c):  M = 1 if r > r_c, else 0

Filtering is multiplication in the frequency domain:

    y = Re{{ F⁻¹ {{ ifftshift( fftshift(F{{x}}) · M ) }} }}

The real part is taken because x is real, so its spectrum is Hermitian-symmetric,
and a radially symmetric mask preserves that symmetry — the inverse transform is
real to within ~1e-7 numerical residue.

**Three properties give free correctness checks**, all asserted in
`src/eval/frequency.py`'s self-test:

1. `M_lp(r) + M_hp(r) = 1` everywhere, and filtering is linear, so
   `lowpass(x, r) + highpass(x, r) = x`. Verified to < 1e-4.
2. The outermost populated bin sits at the corner, r = √(112² + 112²) = 158.39.
   So a low-pass at **159 bins** covers every bin and is the **identity**. Verified
   to 4e-07 — and confirmed on real trained models, where low-pass at r=159
   reproduces clean accuracy exactly.
3. Every band-limited noise sample carries identical spatial RMS with no spectral
   leakage outside its annulus.

Sweep cutoffs (bins): 4, 8, 12, 16, 24, 32, 48, 72, 112, 159.

![How the probes are built](results/figures/assets/fft_construction.png)

![Low-pass sweep](results/figures/assets/frequency_lowpass_ladder.png)

**Honest caveat:** ideal (brick-wall) filters *ring*. A sharp cutoff in frequency
is a sinc in space, so low-pass images show Gibbs halos. This is deliberate and
standard for this analysis (Park & Kim, 2022 use the same construction); a
Butterworth or Gaussian rolloff would trade ringing for an ambiguous cutoff, and
the cutoff is the independent variable of the whole experiment. Clamping back to
[0,1] afterwards is a mild nonlinearity, but feeding out-of-range pixels would be
a different and less physical distortion.

**Band-limited fixed-energy noise.** The second probe holds perturbation *energy*
constant and slides it across the spectrum, asking where the model is most
vulnerable. White Gaussian noise is generated in image space, transformed, masked
to an annulus r_lo ≤ r < r_hi, transformed back, and then **rescaled so its
spatial RMS equals a fixed target** (0.10 in [0,1] units).

The rescaling is the crucial step. An annulus near DC contains far fewer bins than
one near Nyquist, so without renormalisation the high-frequency bands would simply
carry more energy, and the curve would measure bin count rather than model
sensitivity.

Bands (bins): (0,8), (8,16), (16,32), (32,56), (56,88), (88,159).

![Band noise](results/figures/assets/band_noise_ladder.png)

### 5.4 Why relative retention, not raw accuracy

When comparing a 10%-data model with a 100%-data model, raw accuracy under noise
conflates two things: how good the model is, and how much noise hurts it. The
contribution figure therefore divides each run's band accuracy by **its own clean
accuracy**:

    retention(band) = accuracy(band) / accuracy(clean)

This asks "what *fraction* of this model's ability survives noise in this band?",
which is the only normalisation under which a weak model and a strong model are
comparable on one axis.

### 5.5 Calibration

A model outputs a confidence with each prediction. Calibration asks whether that
number can be believed. **Expected Calibration Error** over B equal-width
confidence bins:

    ECE = Σ_b (n_b / N) · | acc(b) − conf(b) |

— the average gap between stated confidence and observed accuracy, weighted by
how many predictions land in each bin. B = 15 here (Guo et al., 2017).

Equal-*width* bins are the standard choice and make ECE comparable to published
numbers; the cost is that accurate models pile mass into the top bin, so bin
counts are reported alongside.

Sign convention: `overconfidence_gap = mean(confidence) − accuracy`. Positive
means the model overstates itself.

### 5.6 Error overlap

Do the two families fail on the same images? For matched (fraction, seed) pairs:

- the 2×2 outcome table: both correct / only A / only B / both wrong
- **Cohen's κ** on the correctness variable:

      κ = (p_o − p_e) / (1 − p_e)

  where p_o is observed agreement and p_e is the agreement expected by chance
  given each model's accuracy. This correction matters: two models that are each
  ~90% accurate agree ~82% of the time by luck alone, so raw agreement would look
  impressive and mean nothing. κ ≈ 0 means effectively independent; κ ≈ 1 means
  interchangeable.
- **same-wrong-label rate** — among images both get wrong, how often they emit the
  *same* wrong class. Shared wrong answers indicate shared inductive bias or
  genuinely ambiguous ground truth; independent wrong answers indicate different
  failure modes.
- **oracle top-1** — accuracy of a hypothetical picker that is right whenever
  *either* model is. The honest upper bound on how complementary they are.
"""
