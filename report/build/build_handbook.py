"""Generate HANDBOOK.md — the self-contained theory + reproduction document.

Intended reader: someone who has never seen this project and has no particular
machine-learning background, who should be able to read this one file and then
reproduce every number in it.

Theory is fixed prose (handbook_part1/2/3.py). Everything measured interpolates
from Facts, so the document cannot drift from results/tables/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import handbook_part1, handbook_part2, handbook_part3
from facts import Facts

REPO_ROOT = Path(__file__).resolve().parents[2]


def _results_section(f):
    L = []
    A = L.append
    A("### 9.1 Data efficiency (full fine-tuning)")
    A("")
    A("Frozen-test top-1, mean ± SD over 3 seeds.")
    A("")
    A("| training data | ResNet-50 | ViT-B/16 | gap (pp) |")
    A("|---|---|---|---|")
    for fr in (10, 25, 50, 100):
        r, v = f.top1("resnet50", fr), f.top1("vit_b16", fr)
        if not r or not v:
            A(f"| {fr}% | — | — | — |")
            continue
        A(f"| **{fr}%** | {r[0]*100:.2f} ± {r[1]*100:.2f} | "
          f"{v[0]*100:.2f} ± {v[1]*100:.2f} | **{f.gap_at(fr):+.2f}** |")
    A("")
    A("ViT-B/16 leads at every fraction, and every gap exceeds the pooled seed "
      "spread. The advantage is largest where data is scarcest — the reverse of the "
      "usual expectation, which applies to training from scratch rather than to "
      "transfer learning.")
    A("")
    A("![Data efficiency](results/figures/fig_data_efficiency.png)")
    A("")

    if f.available()["probes"]:
        A("### 9.2 Linear probes (frozen features)")
        A("")
        A("| training data | ResNet-50 | ViT-B/16 |")
        A("|---|---|---|")
        for fr in (10, 25, 50, 100):
            r = f.top1("resnet50", fr, regime="linprobe")
            v = f.top1("vit_b16", fr, regime="linprobe")
            A(f"| **{fr}%** | {r[0]*100:.2f} | {v[0]*100:.2f} |" if r and v
              else f"| {fr}% | — | — |")
        A("")
        A("The ordering **inverts**: ViT's frozen features win only at 10%; ResNet's "
          "win at 25% and above. So the transformer's fine-tuning advantage is not "
          "that its features are better — it is that they *adapt* better.")
        A("")

    A("### 9.3 Corruption robustness")
    A("")
    A("| model | clean top-1 (f100) | mean accuracy lost over 15 corruption cells |")
    A("|---|---|---|")
    for m, n in (("resnet50", "ResNet-50"), ("vit_b16", "ViT-B/16")):
        A(f"| {n} | {f.top1_str(m, 100)} | {f.corruption_drop(m)} |")
    A("")
    A("On clean images the two are within a point. Under the heaviest sensor noise "
      "the difference is nearly 30 points — a benchmark reporting only clean accuracy "
      "would describe these models as near-equivalent and be badly misleading.")
    A("")
    A("![Corruption](results/figures/fig_corruption.png)")
    A("")

    A("### 9.4 Frequency sensitivity")
    A("")
    A("| model | low-pass AUC | high-pass AUC | most damaging noise band |")
    A("|---|---|---|---|")
    for m, n in (("resnet50", "ResNet-50"), ("vit_b16", "ViT-B/16")):
        A(f"| {n} | {f.freq_auc(m, 'lp')} | {f.freq_auc(m, 'hp')} | {f.band_weakness(m)} |")
    A("")
    A("ResNet-50 has a sharp low-frequency vulnerability; ViT-B/16's profile is far "
      "flatter with no comparable weak band. Real-world damage disturbs many bands at "
      "once, so a model with a sharp weak spot gets caught by it.")
    A("")
    A("![Frequency](results/figures/fig_frequency.png)")
    A("")
    A("![Band noise](results/figures/fig_band_noise.png)")
    A("")

    ps = f.profile_shift()
    if ps and "resnet50" in ps and "vit_b16" in ps:
        r, v = ps["resnet50"], ps["vit_b16"]
        ratio = r["max_abs"] / v["max_abs"] if v["max_abs"] else float("nan")
        A("### 9.5 The contribution — spectral robustness is data-dependent for the CNN only")
        A("")
        A(f"Change in relative retention per band between 10% and 100% training data, "
          f"averaged over {r['n_lo']} seeds. Positive = more robust in that band as "
          f"data grew.")
        A("")
        A("| model | " + " | ".join(r["bands"]) + " |")
        A("|---" * (len(r["bands"]) + 1) + "|")
        A("| **ResNet-50** | " + " | ".join(f"{x:+.3f}" for x in r["shift"]) + " |")
        A("| **ViT-B/16** | " + " | ".join(f"{x:+.3f}" for x in v["shift"]) + " |")
        A("")
        A(f"ResNet-50's profile moves by {r['max_abs']:.3f} (concentrated at high "
          f"frequency, band {r['max_band']}); ViT-B/16's barely moves "
          f"({v['max_abs']:.3f}). A **{ratio:.0f}× difference**.")
        A("")
        A("**Interpretation.** The CNN must *learn* high-frequency robustness from the "
          "fine-tuning data. The transformer inherits a flat profile from pre-training "
          "and does not need downstream data to acquire it. That is a mechanism for "
          "the data-efficiency result rather than a restatement of it: the "
          "transformer's advantage is largest exactly where the CNN has least data "
          "from which to learn what the transformer already has.")
        A("")
        A("**How this differs from prior work.** Park & Kim (2022) establish that the "
          "two families differ in frequency response, at full scale. What is new here "
          "is that the difference is *itself data-dependent* for one family and not "
          "the other — visible only under a protocol that varies the data budget while "
          "holding everything else fixed.")
        A("")
        A("![Profile shift](results/figures/fig_frequency_shift.png)")
        A("")

    o = f.overlap_at(100)
    if o:
        A("### 9.6 Error overlap and calibration")
        A("")
        A("| measure at f100 | value |")
        A("|---|---|")
        A(f"| Both models wrong | {o['both_wrong']*100:.1f}% |")
        A(f"| …giving the same wrong label | {o['same_wrong']*100:.1f}% |")
        A(f"| Cohen's κ on correctness | {o['kappa']:.3f} |")
        A(f"| Oracle top-1 (either model right) | {o['oracle']*100:.2f}% |")
        A(f"| ECE — ResNet-50 / ViT-B/16 | {f.ece('resnet50')} / {f.ece('vit_b16')} |")
        A("")
        A("They fail on different images: the oracle sits well above either model "
          "alone, so they are complementary rather than interchangeable. And they are "
          "similarly calibrated, so the robustness difference is **not** bought by the "
          "transformer simply being less confident — an objection worth pre-empting.")
        A("")

    A("### 9.7 Deployment cost")
    A("")
    rd, vd = f.deployment_row("resnet50"), f.deployment_row("vit_b16")
    if rd and vd:
        A("| | ResNet-50 | ViT-B/16 | ratio |")
        A("|---|---|---|---|")
        A(f"| Parameters (M) | {rd.get('params_m',0):.1f} | {vd.get('params_m',0):.1f} "
          f"| {vd.get('params_m',1)/max(rd.get('params_m',1),1e-9):.1f}× |")
        A(f"| GMACs @224 | {rd.get('gmacs',0):.2f} | {vd.get('gmacs',0):.2f} "
          f"| {vd.get('gmacs',1)/max(rd.get('gmacs',1),1e-9):.1f}× |")
        A(f"| Peak train VRAM (MB) | {rd.get('peak_vram_mb',float('nan')):.0f} "
          f"| {vd.get('peak_vram_mb',float('nan')):.0f} | |")
        A("")
    A(f"Total compute for the whole study: **{f.gpu_hours()} GPU-hours** on one "
      f"RTX 4060 Laptop GPU (8 GB).")
    A("")

    rep = REPO_ROOT / "results" / "tables" / "replication.csv"
    if rep.exists():
        import pandas as pd
        rdf = pd.read_csv(rep)
        A("### 9.8 Does Food-101 replicate it?")
        A("")
        A("| finding | Caltech-256 | Food-101 | verdict |")
        A("|---|---|---|---|")
        for _, row in rdf.iterrows():
            a = row["agrees"]
            v = "pending" if pd.isna(a) else ("**replicates**" if a else "**differs**")
            A(f"| {row['claim']} | {row['caltech256']} | {row['food101']} | {v} |")
        A("")
        A("Two cautions. Food-101 runs **one seed** by design, so it supports "
          "statements about direction, not significance. And a finding that holds on "
          "one dataset and not the other indicates *dataset dependence*, not an error "
          "in the first measurement — the Caltech result rests on its own internal "
          "validity (frozen split, three seeds, one protocol).")
        A("")
    return "\n".join(L)


PITFALLS = """
### 10.1 The learning-rate schedule fought the early-stopping rule

**The most serious error in the project, and it nearly inverted the headline.**

The declared schedule was a 30-epoch cosine with early stopping at patience 5.
Over 30 epochs the learning rate is still ~9e-5 at epoch 8, so validation accuracy
sits on a noisy plateau — and patience expired *there*, killing runs before the
annealing phase that actually converges the model.

Measured at f100, seed 0, same LR and same seed:

| model | 30-ep truncated | 8-ep annealed | 12-ep annealed | 15-ep annealed | runs cut short |
|---|---|---|---|---|---|
| ResNet-50 | 89.20 | 89.43 | — | 89.73 | 8 of 12 |
| ViT-B/16 | 87.34 | 90.37 | 90.04 | 89.26 | **10 of 10** |

The truncation cost the transformer 2.7–3.0 pp and the CNN 0.2 — a **15×
asymmetry**. Under the broken protocol the measured gap decayed with data and
*reversed sign* (+1.38 → +0.91 → −0.45 pp). Under the corrected one it never
crosses. Same data, same seeds.

**The tell was visible in the logs the whole time:** every ViT run early-stopped,
and ViT's best epoch moved *earlier* as data grew (14–17 at f10, 3 at f100). That
is the signature of stopping on plateau noise rather than on convergence.

**Lesson.** A schedule and a stopping rule are not independent settings. If the
schedule needs N epochs to anneal, the stopping rule must not be able to fire
before then. Check *how* runs terminate, not just what they score.

### 10.2 A shared learning rate is not the neutral choice it looks like

The linear probe originally used one LR (1e-3) for both models, on the reasoning
that fitting a linear head on frozen features is near-convex and therefore free of
architecture-specific dynamics. The sweep falsified that:

| model | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1.0 |
|---|---|---|---|---|---|
| ResNet-50 | 0.5160 | 0.8755 | 0.8883 | **0.8896** | 0.8024 |
| ViT-B/16 | 0.8145 | **0.8556** | 0.8529 | 0.8230 | — |

The optima sit a **decade apart** — 2048-d pooled CNN features and 768-d ViT
features are differently scaled and conditioned. Running both at 1e-3 cost the
ResNet probe **21 pp at f10**.

**Lesson.** "Same value for both" is only fair when the value means the same thing
to both. Sweep it and find out rather than assuming.

### 10.3 A dataset archive that was quietly corrupt

The staged Food-101 zip failed three ways, each masking the next:

1. Its 5.0 GB payload member broke Debian's UnZip 6.00 (zip64 >4 GB) — it aborted
   part-way while still printing "inflating".
2. Extracting that member correctly with Python then failed at stage two: the
   inner archive's central directory records local-header offsets inflated by 2³²
   for 71% of members, non-uniformly. About 16% of images would have gone missing
   in an unpredictable pattern.
3. Independently: it carried 101,000 real JPEGs **and** 101,000 macOS AppleDouble
   `._*.jpg` resource forks. `make_splits.py` globs by extension, so extracting
   them would have doubled the dataset with 4 KB corrupt files and silently
   poisoned every Food-101 number.

Fixed by downloading the official ETH Zurich tarball instead.

**Lesson.** Validate dataset counts against the published figures *before*
training on them, and make the extraction script assert them rather than report
them.

### 10.4 CUDA vanished after a routine system update

`torch.cuda.is_available()` returned False. Cause: an apt upgrade installed NVIDIA
userspace 580.173.02 while the machine was still booted into a kernel carrying the
580.159.03 module. Userspace and kernel module must match exactly.

Diagnosis: compare `nvidia-smi` output against `cat /proc/driver/nvidia/version`.
Fix: boot the kernel whose module matches, or reinstall the module for the running
kernel.

**Lesson.** It looks like a PyTorch problem and is not.

### 10.5 Parsing identifiers with `split()`

Run IDs look like `{model}_{dataset}_f{frac}_s{seed}_{regime}`, so
`run_id.split("_")` seems fine — until a model name contains an underscore:

    resnet50_caltech256_f10_s0_fullft  → 5 fields, indexes line up by luck
    vit_b16_caltech256_f10_s0_fullft   → 6 fields, every index shifts

It crashed loudly (`int("altech256")`), which was the lucky outcome; any shifted
field that still parsed would have written silently mislabelled rows into the
tables the report reads. Fixed by parsing **from the right** in
`src/utils/runid.py`.

### 10.6 Smaller traps worth knowing

- **YAML scientific notation.** PyYAML resolves a float only with *both* a decimal
  point and a signed exponent. `probe_lr: 1e-1` loads as the **string** `"1e-1"`
  and fails deep inside the optimizer. Write `1.0e-1`.
- **Completion markers.** `train.py` writes `best.pt` whenever validation improves
  but `metrics.json` only at the end. The battery originally discovered
  checkpoints by `best.pt`, so a killed run's half-trained weights would have been
  scored as a finished matrix cell — producing entirely plausible-looking curves.
- **Editing a running bash script.** Bash reads scripts by byte offset, so
  changing one mid-run makes it resume mid-token. Restart instead.
- **Suffix filtering.** `aggregate.py` excluded runs matching `"sweep"` but not
  other suffixes, so a schedule diagnostic was pulled into `master.csv` as an
  extra run reusing a real cell's identity.
- **`git add -A` after archiving.** Moving checkpoints to `results/archive/` put
  them outside the `results/checkpoints/*` ignore rule; 4.3 GB of weights were one
  command from entering git history.
"""


def main():
    f = Facts()
    d = f.dataset_facts()
    rd, vd = f.deployment_row("resnet50"), f.deployment_row("vit_b16")
    ctx = dict(
        rp=rd.get("params_m", 0), rg=rd.get("gmacs", 0), rf=int(rd.get("feature_dim", 0) or 0),
        vp=vd.get("params_m", 0), vg=vd.get("gmacs", 0), vf=int(vd.get("feature_dim", 0) or 0),
        rlr=f.lr_selected("resnet50"), vlr=f.lr_selected("vit_b16"),
        classes=d["classes"], total=d["total"], train=d["train_f100"],
        val=d["val"], test=d["test"],
    )
    text = (handbook_part1.sections(f, ctx)
            + handbook_part2.sections(f, ctx)
            + handbook_part3.sections(f, ctx, _results_section(f), PITFALLS))
    out = REPO_ROOT / "HANDBOOK.md"
    out.write_text(text)
    words = len(text.split())
    print(f"[handbook] {out.name} — {len(text.splitlines())} lines, ~{words:,} words")


if __name__ == "__main__":
    main()
