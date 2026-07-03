"""Peak-VRAM / throughput / parameter-count helpers used by train and eval.

Deployment cost (params, peak VRAM, imgs/sec) is one of the committed
evaluation axes, so it is measured inside the training loop itself — per
epoch, on the real workload — rather than reconstructed later.
"""
import time

import torch


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def param_count_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


class Stopwatch:
    """with Stopwatch() as t: ...; t.seconds"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        return False
