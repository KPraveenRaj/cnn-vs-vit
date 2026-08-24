"""The ONLY source of numbers for every deck and report. Nothing is typed by hand.

CLAUDE.md forbids fabricating results or filling sections with unverified
claims. That is easy to violate by accident when writing slides at 2 a.m., so
this module makes it structurally hard: every generated document imports facts
from here, and here reads exclusively from results/tables/*.csv.

Two consequences worth stating:
  - A number that does not exist on disk cannot appear in a document. Missing
    data surfaces as PENDING, which renders visibly as "(pending)" in the
    output, rather than as a plausible-looking invention.
  - Regenerating documents after more runs land updates every figure, table and
    inline number at once, so a deck can never quietly disagree with master.csv.

`available()` lets a document ask what stage the project is actually at, so the
progress decks can describe exactly the work that is genuinely finished.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"
ASSETS = FIGURES / "assets"

PENDING = "(pending)"
MODEL_DISPLAY = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16"}


def _read(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


class Facts:
    def __init__(self):
        self.master = _read("master.csv")
        self.curves = _read("curves_long.csv")
        self.calib = _read("calibration.csv")
        self.overlap = _read("error_overlap.csv")
        self.deploy = _read("deployment.csv")
        self.sweep = _read("lr_sweep.csv")
        self.ledger = _read("compute_ledger.csv")
        self.cost = _read("model_cost.csv")

    # ---- availability -----------------------------------------------------
    def models_done(self, regime="fullft", dataset="caltech256"):
        if self.master.empty:
            return []
        d = self.master[(self.master["regime"] == regime)
                        & (self.master["dataset"] == dataset)
                        & self.master["test_top1"].notna()]
        return sorted(d["model_name"].unique())

    def n_runs(self, regime=None, dataset=None):
        if self.master.empty:
            return 0
        d = self.master
        if regime is not None:
            d = d[d["regime"] == regime]
        if dataset is not None:
            d = d[d["dataset"] == dataset]
        return len(d)

    def battery_done(self, dataset=None):
        """Checkpoints with a completed battery. Scope to one dataset by default-free
        argument — the totals differ per dataset, and mixing them produced a
        '30 of 24' in an earlier draft of the submission guide."""
        if self.master.empty or "corr_mean_top1" not in self.master.columns:
            return 0
        d = self.master
        if dataset is not None:
            d = d[d["dataset"] == dataset]
        return int(d["corr_mean_top1"].notna().sum())

    def battery_total(self, dataset="caltech256"):
        """How many fullft checkpoints exist for that dataset."""
        if self.master.empty:
            return 0
        d = self.master[(self.master["dataset"] == dataset)
                        & (self.master["regime"] == "fullft")]
        return int(len(d))

    def available(self):
        """What genuinely exists right now -- drives which claims a deck may make."""
        return {
            "resnet_fullft": "resnet50" in self.models_done("fullft"),
            "vit_fullft": "vit_b16" in self.models_done("fullft"),
            "probes": self.n_runs("linprobe") > 0,
            "battery": self.battery_done() > 0,
            "overlap": not self.overlap.empty,
            "food101": (not self.master.empty
                        and (self.master["dataset"] == "food101").any()),
        }

    # ---- headline numbers -------------------------------------------------
    def top1(self, model, frac, regime="fullft", dataset="caltech256"):
        """mean +/- SD frozen-test top-1 over seeds, as (mean, sd, n)."""
        if self.master.empty:
            return None
        d = self.master[(self.master["model_name"] == model)
                        & (self.master["fraction"] == frac)
                        & (self.master["regime"] == regime)
                        & (self.master["dataset"] == dataset)]
        d = d[d["test_top1"].notna()]
        if d.empty:
            return None
        return float(d["test_top1"].mean()), float(d["test_top1"].std(ddof=0)), len(d)

    def top1_str(self, model, frac, **kw):
        r = self.top1(model, frac, **kw)
        if r is None:
            return PENDING
        m, s, n = r
        return f"{m*100:.2f}%" + (f" ± {s*100:.2f}" if n > 1 else "")

    def efficiency_row(self, model, regime="fullft"):
        return {f: self.top1_str(model, f, regime=regime) for f in (10, 25, 50, 100)}

    def gap_at(self, frac, regime="fullft"):
        """ViT minus ResNet top-1 in percentage points at one fraction."""
        a, b = self.top1("vit_b16", frac, regime=regime), self.top1("resnet50", frac, regime=regime)
        if a is None or b is None:
            return None
        return (a[0] - b[0]) * 100

    def gap_str(self, frac, **kw):
        g = self.gap_at(frac, **kw)
        return PENDING if g is None else f"{g:+.2f} pp"

    def lr_selected(self, model):
        if self.sweep.empty:
            return PENDING
        d = self.sweep[(self.sweep["model_name"] == model) & self.sweep["selected"]]
        return PENDING if d.empty else f"{d.iloc[0]['lr']:g}"

    def lr_grid(self, model):
        if self.sweep.empty:
            return []
        d = self.sweep[self.sweep["model_name"] == model].sort_values("lr")
        return [(f"{r['lr']:g}", f"{r['best_val_top1']*100:.2f}%", bool(r["selected"]))
                for _, r in d.iterrows()]

    def corruption_drop(self, model, frac=100, dataset="caltech256"):
        """Relative top-1 lost, averaged over all 15 corruption cells.

        MUST filter by dataset. Without it this silently averaged Caltech-256 and
        Food-101 together the moment Food-101 landed, turning 27.3% into 34.6%.
        """
        if self.master.empty or "corr_rel_drop" not in self.master.columns:
            return PENDING
        d = self.master[(self.master["model_name"] == model)
                        & (self.master["dataset"] == dataset)
                        & (self.master["fraction"] == frac)
                        & (self.master["regime"] == "fullft")]
        d = d[d["corr_rel_drop"].notna()]
        return PENDING if d.empty else f"{d['corr_rel_drop'].mean()*100:.1f}%"

    def freq_auc(self, model, tag="lp", frac=100, dataset="caltech256"):
        col = f"freq_{tag}_auc"
        if self.master.empty or col not in self.master.columns:
            return PENDING
        d = self.master[(self.master["model_name"] == model)
                        & (self.master["dataset"] == dataset)
                        & (self.master["fraction"] == frac)
                        & (self.master["regime"] == "fullft")]
        d = d[d[col].notna()]
        return PENDING if d.empty else f"{d[col].mean():.3f}"

    def band_weakness(self, model, frac=100, dataset="caltech256"):
        """Which frequency band hurts this model most."""
        if self.master.empty or "freq_band_argmin" not in self.master.columns:
            return PENDING
        d = self.master[(self.master["model_name"] == model)
                        & (self.master["dataset"] == dataset)
                        & (self.master["fraction"] == frac)
                        & (self.master["regime"] == "fullft")]
        d = d[d["freq_band_argmin"].notna()]
        return PENDING if d.empty else str(d.iloc[0]["freq_band_argmin"]) + " bins"

    def ece(self, model, frac=100, dataset="caltech256"):
        if self.calib.empty:
            return PENDING
        d = self.calib[(self.calib["model_name"] == model)
                       & (self.calib["dataset"] == dataset)
                       & (self.calib["fraction"] == frac)
                       & (self.calib["regime"] == "fullft")]
        return PENDING if d.empty else f"{d['ece'].mean():.4f}"

    def overlap_at(self, frac=100, dataset="caltech256"):
        if self.overlap.empty:
            return None
        d = self.overlap[(self.overlap["dataset"] == dataset)
                         & (self.overlap["fraction"] == frac)
                         & (self.overlap["regime"] == "fullft")]
        if d.empty:
            return None
        return {"kappa": d["kappa"].mean(), "both_wrong": d["both_wrong"].mean(),
                "same_wrong": d["same_wrong_pred"].mean(),
                "oracle": d["oracle_top1"].mean()}

    def deployment_row(self, model):
        if self.deploy.empty:
            return {}
        d = self.deploy[self.deploy["model_name"] == model]
        return {} if d.empty else d.iloc[0].to_dict()

    def gpu_hours(self):
        """TRAINING compute only. The ledger is built from metrics.json wall times,
        which cover training; the evaluation battery is not included. Use
        gpu_hours_total() when the label says 'total'."""
        if self.ledger.empty:
            return PENDING
        return f"{self.ledger['gpu_hours'].sum():.1f}"

    def gpu_hours_battery(self):
        """Battery compute, from the per-pass timings recorded in eval_results.json."""
        import json
        total = 0.0
        runs = REPO_ROOT / "results" / "runs"
        if not runs.is_dir():
            return None
        for d in runs.iterdir():
            p = d / "eval_results.json"
            if not p.exists():
                continue
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            secs = r.get("meta", {}).get("elapsed_seconds")
            if secs:
                total += secs / 3600.0
        return total or None

    def gpu_hours_total(self):
        """Training plus battery, when the battery duration is recoverable."""
        if self.ledger.empty:
            return PENDING
        train = self.ledger["gpu_hours"].sum()
        batt = self.gpu_hours_battery()
        return f"{train + batt:.1f}" if batt else f"{train:.1f}+"

    def dataset_facts(self):
        splits = REPO_ROOT / "data" / "splits" / "caltech256"
        out = {"classes": 256, "excluded": "257.clutter"}
        for name, f in (("test", "test.csv"), ("val", "val.csv"),
                        ("train_f100", "train_f100_s0.csv")):
            p = splits / f
            out[name] = sum(1 for _ in p.open()) - 1 if p.exists() else 0
        out["total"] = out["test"] + out["val"] + out["train_f100"]
        return out

    def profile_shift(self, lo=10, hi=100):
        """How much each model's spectral robustness profile MOVES with data budget.

        Returns {model: {"bands": [...], "lo": [...], "hi": [...], "shift": [...],
                         "max_abs": float, "max_band": str, "n_lo": int, "n_hi": int}}
        using relative retention (band accuracy / that run's own clean accuracy),
        which is the only normalisation under which a 10%-data and a 100%-data run
        are comparable on one axis. None if the battery has not covered both ends.
        """
        import json
        import numpy as np
        out = {}
        for model in ("resnet50", "vit_b16"):
            got = {}
            for frac in (lo, hi):
                rows = []
                for seed in (0, 1, 2):
                    p = (REPO_ROOT / "results" / "runs" /
                         f"{model}_caltech256_f{frac}_s{seed}_fullft" / "eval_results.json")
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    b = r.get("frequency", {}).get("band_noise", {})
                    if not b or "clean" not in r:
                        continue
                    ks = sorted(b, key=lambda x: int(x.split("-")[0]))
                    rows.append([b[k]["top1"] / r["clean"]["top1"] for k in ks])
                if rows:
                    got[frac] = (np.array(rows), ks)
            if lo in got and hi in got:
                a_lo, ks = got[lo]
                a_hi, _ = got[hi]
                shift = a_hi.mean(0) - a_lo.mean(0)
                i = int(np.abs(shift).argmax())
                out[model] = {"bands": ks, "lo": a_lo.mean(0).tolist(),
                              "hi": a_hi.mean(0).tolist(), "shift": shift.tolist(),
                              "max_abs": float(abs(shift[i])), "max_band": ks[i],
                              "n_lo": len(a_lo), "n_hi": len(a_hi)}
        return out or None

    def contribution_verdict(self):
        """Did the contribution replicate on Food-101? Returns dict or None.

        The documents must state this honestly whichever way it goes, so the
        verdict is read from results/tables/replication.csv rather than being
        asserted in prose that could go stale after a rerun.
        """
        p = TABLES / "replication.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        rows = df[df["claim"].str.contains("profile shifts MORE|HIGH-frequency",
                                           case=False, na=False)]
        if rows.empty:
            return None
        out = []
        for _, r in rows.iterrows():
            a = r["agrees"]
            out.append({"claim": str(r["claim"]),
                        "caltech": str(r["caltech256"]),
                        "food101": str(r["food101"]),
                        "agrees": None if pd.isna(a) else bool(a)})
        tested = [o for o in out if o["agrees"] is not None]
        return {"rows": out,
                "all_replicate": bool(tested) and all(o["agrees"] for o in tested),
                "any_tested": bool(tested)}

    def invariance(self):
        """High-frequency retention across every (dataset, fraction) condition.

        The broader claim that survives both datasets: the transformer's spectral
        robustness is invariant, the CNN's is contingent on task and data budget.
        """
        import json
        import numpy as np
        out = {}
        for model in ("resnet50", "vit_b16"):
            vals = {}
            for ds in ("caltech256", "food101"):
                for fr in (10, 25, 50, 100):
                    rows = []
                    for seed in (0, 1, 2):
                        p = (REPO_ROOT / "results" / "runs" /
                             f"{model}_{ds}_f{fr}_s{seed}_fullft" / "eval_results.json")
                        if not p.exists():
                            continue
                        r = json.loads(p.read_text())
                        b = r.get("frequency", {}).get("band_noise", {})
                        if not b or "clean" not in r:
                            continue
                        ks = sorted(b, key=lambda x: int(x.split("-")[0]))
                        rows.append(b[ks[-1]]["top1"] / r["clean"]["top1"])
                    if rows:
                        vals[(ds, fr)] = float(np.mean(rows))
            if vals:
                lo, hi = min(vals.values()), max(vals.values())
                out[model] = {"values": vals, "min": lo, "max": hi,
                              "range": hi / max(lo, 1e-9)}
        if len(out) < 2:
            return None
        out["ratio"] = out["resnet50"]["range"] / max(out["vit_b16"]["range"], 1e-9)
        return out

    def schedule_finding(self):
        """The schedule/early-stopping result, read from the runs that produced it.

        Superseded runs live under results/archive/ep30_truncated/ and the
        12-epoch probe under its own suffixed run, so this table is generated
        from disk like everything else rather than transcribed from a console.
        Returns None if the archive is absent (e.g. a fresh clone).
        """
        import json
        arch = REPO_ROOT / "results" / "archive" / "ep30_truncated" / "runs"
        runs = REPO_ROOT / "results" / "runs"
        if not arch.is_dir():
            return None

        def best(p):
            f = p / "metrics.json"
            if not f.exists():
                return None
            m = json.loads(f.read_text())
            return m["best_val_top1"], m["epochs_cap"], m["epochs_ran"], m["early_stopped"]

        out = {}
        for model in ("resnet50", "vit_b16"):
            cell = f"{model}_caltech256_f100_s0_fullft"
            sweep_lr = {"resnet50": "3e-4", "vit_b16": "1e-4"}[model]
            row = {
                "truncated": best(arch / cell),
                "annealed_8": best(runs / f"{cell}-sweep-lr{sweep_lr}"),
                "annealed_15": best(runs / cell),
            }
            row["annealed_12"] = best(runs / f"{cell}-diag-ep12")
            out[model] = row

        # how often the truncated protocol early-stopped, per model
        stops = {}
        for model in ("resnet50", "vit_b16"):
            tot = hit = 0
            for d in arch.glob(f"{model}_caltech256_f*_s*_fullft"):
                b = best(d)
                if b:
                    tot += 1
                    hit += bool(b[3])
            stops[model] = (hit, tot)
        out["early_stop_counts"] = stops
        return out

    def figure(self, name):
        """Absolute path to a figure, or None if it does not exist yet."""
        for base in (FIGURES, ASSETS):
            p = base / name
            if p.exists():
                return str(p)
        return None


if __name__ == "__main__":
    f = Facts()
    print("availability:", f.available())
    print("dataset:", f.dataset_facts())
    print("runs:", f.n_runs(), "| battery:", f.battery_done(), "| GPU-h:", f.gpu_hours())
    for m in ("resnet50", "vit_b16"):
        print(f"{m:10s} lr={f.lr_selected(m):8s} eff={f.efficiency_row(m)}")
    print("gap@10:", f.gap_str(10), " gap@100:", f.gap_str(100))
