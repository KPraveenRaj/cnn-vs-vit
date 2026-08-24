"""Qualitative saliency: ATTEMPTED, MEASURED, AND NOT REPORTED.

This module is kept as a record of a negative result rather than deleted, because
"we tried the obvious thing and here is why it does not work" is more useful than
silence, and because the temptation to ship a pretty but meaningless figure is
exactly what it documents resisting.

The project plan lists attention maps / Grad-CAM as an optional qualitative
extra. Two standard methods were implemented and measured on the f100 seed-0
checkpoints. Both are degenerate for ViT-B/16:

  1. ATTENTION ROLLOUT (Abnar & Zuidema, 2020) multiplies 0.5*A + 0.5*I across
     all twelve blocks. Measured border-to-centre ratio of the resulting
     class-token map: 1.03. That is a uniform field. Twelve rounds of residual
     mixing destroy the spatial selectivity, and the min-max normalisation that
     follows then amplifies numerical noise into apparent structure.

     Raw LAST-LAYER attention is not uniform (border-to-centre 3.08), but what it
     concentrates on is image borders, not objects — the documented attention-sink
     / register-token behaviour of ViTs (Darcet et al., 2023, "Vision Transformers
     Need Registers"), where a few low-information background patches acquire
     large activations and act as scratch space. Real property of the model, and
     precisely why raw attention is a poor saliency method here.

  2. GRAD-CAM (Selvaraju et al., 2017) on the last block's token outputs,
     reshaped to the 14x14 patch grid. Measured standard deviation of the
     resulting map, after min-max normalisation, across four test images:
     0.0000 in every case — a perfectly uniform map, i.e. nothing at all.

     The cause is structural, not a coding error. Grad-CAM assumes non-negative
     activations, which holds after a ReLU in a CNN. Transformer token activations
     are LayerNorm'd and roughly zero-mean, so the channel-weighted sum
     sum_k alpha_k * A^k is mostly negative and the ReLU zeroes it.

Grad-CAM on ResNet-50 works correctly (map std ~0.22-0.29, with 8-26% of pixels
above half activation and a small high-activation core) and localises objects
sensibly. But a saliency figure showing only one of the two models compared in
this study would be half an argument, so nothing is reported.

CONCLUSION FOR THE REPORT: qualitative saliency for ViT-B/16 is a genuinely open
problem and is not solved by the two standard methods. The mechanism claims in
this project rest entirely on the frequency and corruption batteries, which are
quantitative, seeded, and reproducible. No conclusion depends on a picture.

Run `python -m src.analysis.attention_maps --diagnose` to reproduce the numbers
above.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import eval_geometric_tfms, to_tensor_norm_tfms
from src.models.factory import build_model
from src.utils.config import load_yaml
from src.utils.seed import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCENT = "#1F4E79"


def gradcam_resnet(model, x, class_idx=None):
    """Grad-CAM on the last convolutional stage. x: (1,3,H,W) normalized."""
    acts, grads = {}, {}
    target = model.layer4[-1]

    def fwd_hook(_, __, out):
        acts["v"] = out.detach()

    def bwd_hook(_, __, gout):
        grads["v"] = gout[0].detach()

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, idx].backward()
        a, g = acts["v"], grads["v"]                      # (1,C,h,w)
        alpha = g.mean(dim=(2, 3), keepdim=True)          # channel importance
        cam = F.relu((alpha * a).sum(1, keepdim=True))    # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), idx
    finally:
        h1.remove(); h2.remove()


def gradcam_vit(model, x, class_idx=None):
    """Grad-CAM on the last transformer block's token outputs.

    The block emits (1, 197, 768): one class token plus 196 patch tokens. Dropping
    the class token and reshaping the rest to 14x14 recovers the spatial grid, so
    the same channel-importance weighting used for the CNN applies unchanged.
    """
    acts, grads = {}, {}
    target = model.blocks[-1]

    def fwd_hook(_, __, out):
        acts["v"] = out

    h1 = target.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        a = acts["v"]                                    # (1, N, C), graph attached
        g = torch.autograd.grad(logits[0, idx], a, retain_graph=False)[0]

        a = a[:, 1:, :]                                  # drop class token
        g = g[:, 1:, :]
        n = a.shape[1]
        side = int(n ** 0.5)
        a = a.transpose(1, 2).reshape(1, -1, side, side)  # (1,C,14,14)
        g = g.transpose(1, 2).reshape(1, -1, side, side)

        alpha = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), idx
    finally:
        h1.remove()


def main():
    """Reproduce the measurements that led to not reporting these maps."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnose", action="store_true",
                    help="measure both methods and print why they were rejected")
    ap.add_argument("--data", default="configs/data_caltech256.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.diagnose:
        print(__doc__)
        print("Nothing is generated. Pass --diagnose to reproduce the measurements.")
        return

    seed_everything(0)
    dcfg = load_yaml(REPO_ROOT / args.data)
    test = pd.read_csv(REPO_ROOT / dcfg["splits_dir"] / "test.csv")
    geo = eval_geometric_tfms(224)

    nets = {}
    for m in ("resnet50", "vit_b16"):
        rid = f"{m}_{dcfg['dataset']}_f100_s0_fullft"
        cfg = load_yaml(REPO_ROOT / "results" / "runs" / rid / "config.yaml")
        net = build_model(cfg["timm_name"], cfg["num_classes"]).to(args.device)
        net.load_state_dict(torch.load(
            REPO_ROOT / "results" / "checkpoints" / rid / "best.pt",
            map_location=args.device, weights_only=False)["state_dict"])
        net.eval()
        nets[m] = (net, to_tensor_norm_tfms(cfg["timm_name"]))

    print("  Saliency-map quality. A useful map has high std and a small "
          "high-activation area;\n  a uniform map has std ~ 0.\n")
    print(f"  {'image':>6}  {'model':10} {'std':>8} {'frac>0.5':>9} {'frac>0.8':>9}")
    for i in (1500, 2200, 3100, 4000):
        pil = geo(Image.open(REPO_ROOT / test.iloc[i]["filepath"]).convert("RGB"))
        for m, fn in (("resnet50", gradcam_resnet), ("vit_b16", gradcam_vit)):
            net, tf = nets[m]
            cam, _ = fn(net, tf(pil).unsqueeze(0).to(args.device))
            print(f"  {i:>6}  {m:10} {cam.std():8.4f} "
                  f"{float((cam > 0.5).mean()):9.3f} {float((cam > 0.8).mean()):9.3f}")
    print("\n  ViT-B/16 Grad-CAM is uniform (std 0.0000) -> not reported.")
    print("  See the module docstring for the attention-rollout measurement too.")


if __name__ == "__main__":
    main()
