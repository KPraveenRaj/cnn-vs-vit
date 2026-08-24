"""Part 1 of HANDBOOK.md: purpose, background theory, and the architectures.

Split across files only because it is long. build_handbook.py concatenates the
parts and writes the single output file.
"""

def sections(f, d):
    rp, rg, rf = d["rp"], d["rg"], d["rf"]
    vp, vg, vf = d["vp"], d["vg"], d["vf"]
    return f"""# Project Handbook — theory, method, and how to reproduce everything

*CNNs vs Vision Transformers: A Controlled Comparison under Transfer Learning*
EC789 Major Project · Praveen Raj Konatham (252SP014) · M.Tech SPML, ECE, NITK Surathkal
Guide: Dr. Bini A A

---

## 0. How to use this document

This is written to be self-contained. Someone who has never seen the project
should be able to read this file top to bottom and then reproduce every number in
it, without needing another source.

- **Sections 1–3** are background. Read them if terms like *convolution*,
  *attention* or *transfer learning* are not already familiar.
- **Sections 4–6** are the method: exactly what was run, and why each choice was
  made rather than some other choice.
- **Section 7** is the code, file by file.
- **Section 8** is the step-by-step reproduction recipe.
- **Section 9** is the results.
- **Section 10** lists the mistakes made along the way. It is probably the most
  useful section for anyone doing similar work.

Mathematical notation is kept light. Where a formula appears it is followed by a
plain-language reading of what it does.

---

## 1. The problem

Suppose you have a model that has already been trained on a very large collection
of general photographs, and you want it to solve *your* task, for which you have
far fewer labelled images. This is the ordinary situation in applied computer
vision — almost nobody has a million labelled images of their own problem.

Two broad families of architecture are available:

- **Convolutional neural networks (CNNs)**, the mature choice, represented here by
  **ResNet-50**.
- **Vision Transformers (ViTs)**, adapted from language models in 2020,
  represented here by **ViT-B/16**.

The literature does not answer cleanly which to pick, because published
comparisons usually differ in several ways at once: different pre-training
corpora, different augmentation recipes, different training lengths, different
compute budgets. A conclusion drawn from such a comparison is not isolated from
those confounds.

**This project fixes everything except the architecture, and then asks not only
which model scores higher, but what each one depends on.**

The problem statement, as submitted:

> Vision Transformers now lead image-classification benchmarks, but those results
> rely on large-scale pre-training and compute, and most published CNN-versus-ViT
> comparisons vary many factors at once. This project fine-tunes a representative
> CNN and a representative ViT under a single controlled transfer-learning
> protocol and compares how the two families behave — beginning with their
> efficiency in low-data regimes and their robustness to input degradation. The
> aim is to characterise not just which model scores higher, but how and why each
> one behaves as it does.

---

## 2. Background: the image-classification setup

A classifier is a function that maps an image to a category.

An RGB image at this project's working resolution is a tensor

    x ∈ R^(3 × 224 × 224)

— three colour channels, 224 pixels high, 224 wide. The model produces a vector of
**logits**, one per category:

    z = f(x; θ) ∈ R^C,   C = 256 for Caltech-256

where θ are the model's learned parameters. Logits are unbounded scores. They are
converted to probabilities by the **softmax**:

    p_i = exp(z_i) / Σ_j exp(z_j)

which forces the values to be positive and sum to 1. The predicted class is
`argmax_i p_i`, and `max_i p_i` is the model's **confidence** — a number this
project uses later for calibration analysis.

Training minimises **cross-entropy loss** between the predicted distribution and
the true label y:

    L = − log p_y

Reading it plainly: the loss is small when the model assigned high probability to
the correct answer, and grows without bound as that probability approaches zero.
Minimising it over many examples is what "training" means.

---

## 3. The two architectures

### 3.1 Convolutional networks, and ResNet-50

The **convolution** is the core operation. A small filter (kernel) of learned
weights slides across the image, and at each position computes a weighted sum of
the pixels beneath it:

    y[i, j] = Σ_m Σ_n  w[m, n] · x[i + m, j + n]  +  b

Three properties follow, and they are the reason CNNs work well on images:

1. **Locality** — each output depends only on a small neighbourhood of input.
2. **Weight sharing** — the same filter is applied everywhere, so a pattern
   learned in one part of the image is recognised anywhere in it.
3. **Translation equivariance** — shift the input, and the output shifts with it.

These are *inductive biases*: assumptions built into the architecture rather than
learned from data. They are why CNNs are efficient learners on images — they do
not have to discover from scratch that images are spatially structured.

Stacking convolutions makes deep networks, but naively deep networks train badly.
The specific failure He et al. (2016) identify is *degradation*: adding layers to
a plain network makes even its TRAINING error worse, which cannot be overfitting.
They are explicit that this is not simply vanishing gradients — batch
normalisation already largely addresses those — but an optimisation difficulty in
approximating identity mappings through many nonlinear layers. **ResNet** solves
it with a *residual connection*:

    y = F(x) + x

The block learns a *correction* F(x) to its input rather than a whole new
representation. If the best thing to do is nothing, the block can drive F(x)
toward zero and pass x through — so in principle adding depth need not hurt. This
is what made 50-layer and deeper networks trainable.

**ResNet-50** used here:

| property | value |
|---|---|
| parameters | {rp:.1f} M |
| compute at 224×224 | {rg:.2f} GMACs |
| feature dimension | {rf} |
| pre-training | ImageNet-1k (timm tag `resnet50.a1_in1k`) |

It has 50 weighted layers arranged in four stages. Spatial resolution halves at
each stage while channel count doubles — the standard pyramid. After the last
stage, global average pooling reduces each channel to a single number, giving a
{rf}-dimensional feature vector, which a final linear layer maps to class scores.

### 3.2 Transformers, attention, and ViT-B/16

Transformers came from language modelling (Vaswani et al., 2017). Their core
operation is **self-attention**, which lets every element of a sequence consult
every other element directly.

Given a sequence of N vectors packed into a matrix X ∈ R^(N × d), three linear
projections produce queries, keys and values:

    Q = X·W_Q,   K = X·W_K,   V = X·W_V

and attention is

    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

Reading it step by step:

- `Q·Kᵀ` computes a similarity score between every pair of elements. Element i
  asks, via its query, how relevant every other element's key is.
- Dividing by `√d_k` keeps those scores from growing with dimension, which would
  otherwise push the softmax into a region where gradients vanish.
- `softmax` turns each row of scores into weights summing to 1.
- Multiplying by V produces, for each element, a weighted average of all elements'
  values — weighted by relevance.

**Multi-head attention** runs h of these in parallel with separate projections and
concatenates the results, so different heads can attend to different kinds of
relationship simultaneously.

The crucial difference from convolution: attention is **global from layer one**.
A CNN needs many layers before two distant pixels can influence each other; a
transformer relates any two positions in a single operation.

**Vision Transformer** (Dosovitskiy et al., 2021) applies this to images by
turning a picture into a sequence:

1. **Patch embedding.** Cut the 224×224 image into non-overlapping 16×16 patches.
   That gives (224/16)² = 14 × 14 = **196 patches**. Flatten each patch to a
   vector of 16·16·3 = 768 values and project it linearly to the model dimension
   (also 768 here). Implemented as a strided convolution.
2. **Class token.** Prepend one learned vector whose final state is used as the
   image representation. Sequence length becomes 197.
3. **Positional embeddings.** Attention is permutation-invariant — it has no
   inherent notion of "next to". A learned position vector is added to every token
   (all 197, class token included) so the model knows where each patch came from.
4. **Transformer blocks.** 12 blocks, each = multi-head self-attention +
   feed-forward network, with residual connections and layer normalisation.
5. **Head.** A linear layer on the class token's final state produces class scores.

**ViT-B/16** used here:

| property | value |
|---|---|
| parameters | {vp:.1f} M |
| compute at 224×224 | {vg:.2f} GMACs |
| feature dimension | {vf} |
| patches | 196 (+1 class token) |
| blocks / heads | 12 / 12 |
| pre-training | ImageNet-1k only (timm tag `vit_base_patch16_224.augreg_in1k`) |

**A control that is easy to get wrong.** The most commonly downloaded ViT-B/16
weights are `augreg_in21k_ft_in1k` — pre-trained on ImageNet-**21k**, roughly 14
million images, before fine-tuning on ImageNet-1k. Against an ImageNet-1k ResNet
(~1.3 million images) that is roughly an eleven-fold advantage in pre-training
data, and it would silently invalidate the entire comparison. This project
pins the ImageNet-1k-only tag. It is the single most important control in the
study.

### 3.3 The inductive-bias trade-off

| | ResNet-50 | ViT-B/16 |
|---|---|---|
| built-in assumptions | strong (locality, weight sharing, translation equivariance) | weak (only patch grid + learned positions) |
| receptive field | grows with depth | global from layer 1 |
| data appetite | lower | higher when trained from scratch |
| parameters | {rp:.1f} M | {vp:.1f} M |
| compute | {rg:.2f} GMACs | {vg:.2f} GMACs |

The standard expectation is that ViTs need more data because they must learn what
CNNs get for free. **That expectation is about training from scratch.** This
project tests the transfer-learning regime, where both models arrive pre-trained —
and finds the opposite ordering, which is one of its main results.

### 3.4 Transfer learning

Rather than training from random initialisation, both models start from weights
learned on ImageNet-1k (~1.3 M images, 1000 classes). Two regimes are used here:

**Full fine-tuning.** Replace the final classification layer with a fresh one
sized for the new task, then continue training *all* parameters on the new data at
a small learning rate. The whole model adapts.

**Linear probing.** Freeze every pre-trained parameter. Push images through the
frozen model, take the feature vector it produces, and train only a single linear
layer on top. Nothing inside the model changes.

The contrast is informative. Fine-tuning measures *knowledge plus adaptability*;
probing measures *knowledge alone*. In this project the orderings differ: under
fine-tuning the transformer wins at every fraction, whereas with features frozen
it wins only at 10% and the CNN wins at 25% and above. Note the ordering is NOT
reversed everywhere — at 10% the transformer leads in both regimes. The evidence
therefore points to its advantage lying substantially in adaptability rather than
in raw pre-trained feature quality, without establishing that on its own.
"""
