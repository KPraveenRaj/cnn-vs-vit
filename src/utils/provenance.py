"""Capture everything about HOW a result was produced, not just the result.

metrics.json records what a run scored. This records the machine, the library
stack, the driver, and the git commit that produced it -- the things an examiner
or a stranger reproducing the work needs, and exactly the things that are
invisible in a results table and impossible to reconstruct months later.

This project has already been bitten once by the environment moving underneath
it (an NVIDIA userspace/kernel-module version skew silently disabled CUDA), so
driver and kernel versions are captured explicitly rather than assumed constant.

Also computes the deployment-cost axis that no other module produces:
per-model parameter count and FLOPs. FLOPs come from torch's built-in
FlopCounterMode -- no extra dependency, and it counts the real traced graph
rather than a hand-maintained table of layer formulas.

Outputs
  results/tables/environment.json   one snapshot, overwritten each call
  results/tables/model_cost.csv     params / FLOPs / feature dim per model
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sh(cmd, default="unavailable"):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=15).stdout.strip() or default
    except Exception:
        return default


def environment_snapshot() -> dict:
    import numpy, pandas, sklearn, timm, torchvision
    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "total_vram_gb": round(p.total_memory / 1024**3, 2),
               "capability": f"{p.major}.{p.minor}",
               "multi_processor_count": p.multi_processor_count}
    return {
        "captured": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _sh("git rev-parse HEAD"),
        "git_branch": _sh("git rev-parse --abbrev-ref HEAD"),
        "git_dirty": bool(_sh("git status --porcelain", "")),
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "processor": _sh("lscpu | grep 'Model name' | cut -d: -f2 | xargs"),
                 "cpu_count": _sh("nproc"),
                 "ram_gb": _sh("free -g | awk '/^Mem:/{print $2}'"),
                 "kernel": platform.release()},
        "gpu": gpu,
        "nvidia_driver": _sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "nvrm_kernel_module": _sh("cat /proc/driver/nvidia/version | head -1"),
        "libs": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                 "cudnn": str(torch.backends.cudnn.version()),
                 "torchvision": torchvision.__version__, "timm": timm.__version__,
                 "numpy": numpy.__version__, "pandas": pandas.__version__,
                 "scikit_learn": sklearn.__version__},
        "determinism": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark},
    }


def model_cost(timm_name: str, num_classes: int, resolution: int = 224) -> dict:
    """params / FLOPs / feature dim for one architecture, measured on CPU."""
    import timm
    from torch.utils.flop_counter import FlopCounterMode
    m = timm.create_model(timm_name, pretrained=False, num_classes=num_classes).eval()
    x = torch.randn(1, 3, resolution, resolution)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            m(x)
    total_flops = fc.get_total_flops()
    feat = timm.create_model(timm_name, pretrained=False, num_classes=0).eval()
    with torch.no_grad():
        fdim = int(feat(x).shape[1])
    return {
        "timm_name": timm_name,
        "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
        "trainable_params_m": round(
            sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 3),
        # FlopCounterMode counts multiply-accumulates as 2 flops; GMACs = GFLOPs/2
        # is the convention most vision papers quote, so report both.
        "gflops": round(total_flops / 1e9, 3),
        "gmacs": round(total_flops / 2e9, 3),
        "feature_dim": fdim,
        "resolution": resolution,
    }


def main():
    import pandas as pd
    from src.utils.config import load_yaml
    tables = REPO_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    env = environment_snapshot()
    (tables / "environment.json").write_text(json.dumps(env, indent=2))
    print(f"[provenance] environment.json  (git {env['git_commit'][:8]}, "
          f"torch {env['libs']['torch']}, driver {env['nvidia_driver']})")

    rows = []
    for yml in sorted((REPO_ROOT / "configs").glob("model_*.yaml")):
        cfg = load_yaml(yml)
        data = load_yaml(REPO_ROOT / "configs/data_caltech256.yaml")
        print(f"[provenance] measuring {cfg['model_name']} ...", flush=True)
        rows.append({"model_name": cfg["model_name"],
                     **model_cost(cfg["timm_name"], data["num_classes"])})
    df = pd.DataFrame(rows)
    df.to_csv(tables / "model_cost.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
