"""YAML config handling with the fixed merge order: base <- model <- data <- CLI.

Later sources override earlier ones key-by-key (nested dicts merge
recursively). The exact merged dict is saved into the run folder as
config.yaml, so every run is self-describing — nothing about a run should
ever need to be reconstructed from memory or shell history.
"""
from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def merge(*dicts) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
    return out


def save_yaml(d: dict, path) -> None:
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
