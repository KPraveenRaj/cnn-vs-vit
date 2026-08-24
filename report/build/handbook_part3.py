"""Part 3 of HANDBOOK.md: codebase, reproduction, results, pitfalls, glossary."""

def sections(f, d, res_tables, pitfalls, compute_table, train_h):
    return f"""
---

## 6. Design decisions, and why they went that way

| decision | chosen | why not the alternative |
|---|---|---|
| ViT weights | `augreg_in1k` | the default `in21k_ft_in1k` is pre-trained on ~11× more data (14M vs 1.3M images); using it would have handed the ViT the comparison |
| Per-model LR | yes, declared | one shared LR would cripple whichever family it suited less — their optima are a decade apart |
| Heavy augmentation | excluded | mixup/CutMix/RandAugment interact with architecture; including them reintroduces the confound |
| Fractions | nested, per class | independent draws would confound *quantity* with *which images* |
| Test split | frozen at the start | any use in tuning turns the reported number into a training metric |
| Filters | ideal (brick-wall) | a soft rolloff makes the cutoff ambiguous, and cutoff is the independent variable |
| Corruption timing | after crop, before normalisation | see §5.2 — otherwise severity becomes model- or size-dependent |
| Band noise | fixed RMS after masking | otherwise the curve measures annulus bin count, not the model |
| Epoch budget | 15, shared | see §10.1; a shared budget is the declared protocol, per-model budgets would be a larger deviation |

---

## 7. The code, file by file

```
configs/          base.yaml (shared protocol) + per-model + per-dataset YAMLs
data/splits/      committed CSV manifests — the frozen splits
src/data/         make_splits · datasets · transforms
src/models/       factory
src/train/        train · linear_probe · run_matrix · select_lr
src/eval/         evaluate · corruptions · frequency · run_eval_battery
src/analysis/     aggregate · plots · calibration · error_overlap ·
                  replication · visual_assets · docs_index
src/utils/        seed · config · hw_monitor · provenance · runid
report/build/     facts · deckkit · docxkit · build_decks · build_reports ·
                  build_submission_guide · build_handbook
scripts/          sweeps, matrix drivers, pipeline chains
```

| file | what it does |
|---|---|
| `src/data/make_splits.py` | **The correctness anchor.** Builds the frozen splits, asserts class counts, zero overlap between train/val/test, nesting, and that every referenced file exists. Run once per dataset. |
| `src/data/datasets.py` | PyTorch Dataset reading a split CSV. Loading plus transform, nothing else. |
| `src/data/transforms.py` | `train_tfms` / `eval_tfms`, plus the split halves (`eval_geometric_tfms`, `to_tensor_norm_tfms`) the battery splices its operators between. |
| `src/models/factory.py` | `build_model` via timm with a fresh head; `feature_extractor` for probes. |
| `src/train/train.py` | One run: merged config in → `best.pt` + `metrics.json` + `train_log.csv` out. AMP, gradient accumulation, cosine+warmup, early stopping. Skips if `metrics.json` exists — this is what makes every driver crash-resumable. |
| `src/train/linear_probe.py` | Two passes: cache features once per model×dataset keyed by filepath, then fit one linear head per (fraction, seed). Evaluates itself from the cached test features. |
| `src/train/select_lr.py` | Mechanical LR selection from a sweep, by validation top-1 only, rewriting the model YAML with full provenance. |
| `src/train/run_matrix.py` | Enumerates the committed grid, sequential, resumable, prints ETA. |
| `src/eval/evaluate.py` | Checkpoint → frozen test set → metrics + per-image prediction table. |
| `src/eval/corruptions.py` | Deterministic on-the-fly corruptions, seeded per (corruption, severity, image). |
| `src/eval/frequency.py` | FFT probes. **Run `python -m src.eval.frequency` to execute its self-test.** |
| `src/eval/run_eval_battery.py` | 42 passes per checkpoint, resumable **per pass**. |
| `src/analysis/aggregate.py` | Walks `results/runs/` → `master.csv` + `curves_long.csv` + per-class / LR-sweep / deployment / compute-ledger tables. |
| `src/analysis/plots.py` | The nine report figures, one house style, colour = model identity only. |
| `src/analysis/calibration.py` · `error_overlap.py` | CPU-only, from the prediction tables. |
| `src/analysis/replication.py` | Re-tests each Caltech finding on Food-101. |
| `src/analysis/visual_assets.py` | The nine qualitative figures, with logged captions. |
| `src/utils/runid.py` | Parses run IDs **from the right** — model names contain underscores, so `split("_")` silently misaligns fields for `vit_b16`. |
| `src/utils/provenance.py` | Environment snapshot + FLOPs via torch's own counter. |
| `report/build/facts.py` | **The only source of numbers for any document.** Reads `results/tables/` and returns `(pending)` for anything absent. |

**Run-ID convention.** Every artifact is keyed by

    {{model}}_{{dataset}}_f{{fraction}}_s{{seed}}_{{regime}}[-suffix]

e.g. `vit_b16_caltech256_f25_s1_fullft`. The optional suffix marks protocol probes
(LR sweeps, diagnostics) that must never be counted as matrix cells.

---

## 8. Reproducing everything

### 8.1 Environment

```bash
conda create -n torch_env python=3.10 && conda activate torch_env
# torch MUST come from the PyTorch index or you get a CPU-only wheel:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify the GPU is actually visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

*If this prints False after a system update,* the usual cause is an NVIDIA
kernel-module / userspace version skew — see §10.4.

### 8.2 Data

- **Caltech-256** → `data/raw/caltech256/256_ObjectCategories/`
- **Food-101** → `bash scripts/extract_food101.sh` (downloads the official ETH
  tarball; do **not** use a Kaggle re-upload — see §10.3)

Splits are already committed. Regenerating them from scratch (rarely needed):

```bash
python -m src.data.make_splits --data configs/data_caltech256.yaml
```

### 8.3 Sanity checks before spending GPU time

```bash
python -m src.eval.frequency      # FFT self-test: identity, complementarity, band energy
python -m src.utils.runid         # run-ID parser self-test
python -m src.utils.provenance    # environment snapshot + FLOPs
```

### 8.4 The full pipeline

```bash
bash scripts/run_lr_sweep_resnet50.sh    # declared LR sweep, CNN
bash scripts/run_lr_sweep_vit_b16.sh     # declared LR sweep, ViT
bash scripts/run_pipeline.sh             # matrix → probes → battery → analysis → documents
FOOD101=1 bash scripts/run_pipeline.sh   # ... and the Food-101 block
```

Every stage is skip-if-done, so an interrupted run is simply relaunched.

### 8.5 Documents

```bash
python report/build/build_decks.py             # six decks, with speaker notes
python report/build/build_reports.py           # mid-sem and end-sem reports
python report/build/build_submission_guide.py  # what to submit and when
python report/build/build_handbook.py          # this file
```

### 8.6 Measured compute

Actual times on an RTX 4060 Laptop (8 GB), from `results/tables/compute_ledger.csv`
— these are what the runs took, not estimates:

{compute_table}

Training totals {train_h} GPU-hours. The evaluation battery adds roughly 4.5 h for
the 24 Caltech-256 checkpoints (~7.7 min each) and about 2.6 h for the 6 Food-101
checkpoints, whose test split is 3.4x larger. Linear probes are effectively free
once features are cached: about 1.3 seconds per run.

---

## 9. Results

{res_tables}

---

## 10. Mistakes made, and what they cost

This section exists because it is the most transferable part of the project.
Every item below was caught during the work, not anticipated.

{pitfalls}

---

## 11. Glossary

| term | meaning |
|---|---|
| **AMP** | Automatic Mixed Precision — running parts of training in 16-bit floats to save memory and time |
| **Attention** | Operation letting every element of a sequence consult every other directly |
| **Calibration** | Whether a model's stated confidence matches its actual accuracy |
| **Cohen's κ** | Agreement between two raters, corrected for agreement expected by chance |
| **Convolution** | Sliding a small learned filter across an image |
| **DFT / FFT** | Discrete Fourier Transform — decomposes a signal into frequency components; FFT is the fast algorithm |
| **ECE** | Expected Calibration Error |
| **Epoch** | One complete pass over the training set |
| **Fine-tuning** | Continuing training a pre-trained model on a new task |
| **Gradient accumulation** | Summing gradients over several small batches before an update, to simulate a larger batch |
| **Inductive bias** | Assumption built into an architecture rather than learned from data |
| **Linear probe** | Training only a final linear layer on a frozen model's features |
| **Logits** | Raw, unbounded class scores before softmax |
| **Learning rate** | How large a correction the model makes per update |
| **Macro-F1** | F1 averaged unweighted over classes, so rare classes count fully |
| **Nyquist frequency** | Highest frequency representable on a given sampling grid |
| **Patch embedding** | Turning image patches into vectors for a transformer |
| **Softmax** | Converts logits into probabilities summing to 1 |
| **Spatial frequency** | Rate of intensity change across an image: low = broad shapes, high = fine detail |
| **timm** | PyTorch Image Models — the library supplying both pre-trained backbones |
| **Top-1 / Top-5** | Whether the correct class is the highest-scoring / among the five highest |
| **Transfer learning** | Reusing knowledge from one task on another |
| **Weight decay** | Penalty that shrinks weights toward zero, discouraging over-complex solutions |

---

## 12. References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Dosovitskiy, A. et al. (2021). *An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR.
3. Touvron, H. et al. (2021). *Training Data-Efficient Image Transformers & Distillation through Attention.* ICML.
4. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
5. Park, N. & Kim, S. (2022). *How Do Vision Transformers Work?* ICLR.
6. Yin, D. et al. (2019). *A Fourier Perspective on Model Robustness in Computer Vision.* NeurIPS.
7. Hendrycks, D. & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR.
8. Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
9. Goldblum, M. et al. (2023). *Battle of the Backbones.* NeurIPS.
10. Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
11. Steiner, A. et al. (2021). *How to Train Your ViT?* TMLR.

---

*Generated by `report/build/build_handbook.py` from `results/tables/`. Every number
above traces to a run on disk; see `results/DOCUMENTATION_INDEX.md`.*
"""
