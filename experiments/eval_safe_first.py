"""Evaluate one NeuroSafe config + frozen baselines on full grid."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.run_fair_tune import run_agent_episodes, summarize
from neurosafe_cbo.config import CFG

OUT = os.path.join(ROOT, "results", "day7_tuned")


def eval_config(name_suffix, overrides):
    base = dict(CFG)
    cfg_ns = dict(base)
    cfg_ns.update(overrides)
    frames = []
    for name in ["Chance", "NeuroSafeCBO"]:
        cfg = cfg_ns if name == "NeuroSafeCBO" else base
        df = run_agent_episodes(
            name,
            cfg,
            base["SHIFT_MAGS"],
            base["OBS_NOISES"],
            8,
            12,
            f"{name_suffix}-{name}",
        )
        frames.append(df)
    final = pd.concat(frames, ignore_index=True)
    # also keep full baselines from previous final if present; here only Chance vs NS
    g = (
        final.groupby("agent", as_index=False)
        .agg(
            mean_return=("return", "mean"),
            std_return=("return", "std"),
            mean_viol=("viol", "mean"),
            std_viol=("viol", "std"),
            mean_cvar=("cvar", "mean"),
        )
        .sort_values("mean_viol")
    )
    path = os.path.join(OUT, f"summary_{name_suffix}.csv")
    g.to_csv(path, index=False)
    print(g.to_string(index=False))
    return g


if __name__ == "__main__":
    # safe-first pick from tuning table
    safe_first = {"NEURO_GAIN": 0.0, "LAMBDA_CVAR": 1.6, "ALPHA": 0.05}
    print("=== Safe-first config on full grid ===")
    eval_config("safe_first", safe_first)
    with open(os.path.join(OUT, "safe_first_config.json"), "w", encoding="utf-8") as f:
        json.dump(safe_first, f, indent=2)
