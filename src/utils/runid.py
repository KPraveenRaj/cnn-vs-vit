"""Parse a run ID back into its fields.

Run IDs are built as

    {model}_{dataset}_f{fraction}_s{seed}_{regime}[-suffix]

and the obvious `run_id.split("_")` parse is WRONG, because model names may
themselves contain underscores:

    resnet50_caltech256_f10_s0_fullft  -> 5 fields, indexes happen to line up
    vit_b16_caltech256_f10_s0_fullft   -> 6 fields, everything shifts by one

The second form silently made `dataset` land where `fraction` was expected. So
parse from the RIGHT: regime, seed, fraction and dataset are always the last
four fields, and the model name is whatever precedes them, however many
underscores it contains.

The optional "-suffix" (used by LR-sweep runs, e.g. `...-sweep-lr1e-4`) is split
off the regime and returned separately rather than being silently glued to it.
"""
from typing import NamedTuple


class RunID(NamedTuple):
    model_name: str
    dataset: str
    fraction: int
    seed: int
    regime: str
    suffix: str = ""

    @property
    def is_sweep(self) -> bool:
        return bool(self.suffix)


def parse_run_id(run_id: str) -> RunID:
    parts = run_id.split("_")
    if len(parts) < 5:
        raise ValueError(f"not a run ID: {run_id!r}")
    regime_field = parts[-1]
    regime, _, suffix = regime_field.partition("-")
    frac_field, seed_field = parts[-3], parts[-2]
    if not (frac_field.startswith("f") and seed_field.startswith("s")):
        raise ValueError(f"not a run ID: {run_id!r}")
    return RunID(model_name="_".join(parts[:-4]),
                 dataset=parts[-4],
                 fraction=int(frac_field[1:]),
                 seed=int(seed_field[1:]),
                 regime=regime,
                 suffix=suffix)


if __name__ == "__main__":
    cases = [
        ("resnet50_caltech256_f10_s0_fullft", "resnet50", "caltech256", 10, 0, "fullft", ""),
        ("vit_b16_caltech256_f100_s2_fullft", "vit_b16", "caltech256", 100, 2, "fullft", ""),
        ("vit_b16_caltech256_f25_s1_linprobe", "vit_b16", "caltech256", 25, 1, "linprobe", ""),
        ("resnet50_food101_f25_s0_fullft", "resnet50", "food101", 25, 0, "fullft", ""),
        ("vit_b16_caltech256_f100_s0_fullft-sweep-lr1e-4", "vit_b16", "caltech256",
         100, 0, "fullft", "sweep-lr1e-4"),
    ]
    for rid, *want in cases:
        got = parse_run_id(rid)
        assert tuple(got) == tuple(want), f"{rid}\n  got  {tuple(got)}\n  want {tuple(want)}"
        print(f"  OK  {rid:<50s} -> {got.model_name}/{got.dataset}/f{got.fraction}/"
              f"s{got.seed}/{got.regime}{'/' + got.suffix if got.suffix else ''}")
    print("\nrunid self-test passed.")
