"""Evaluate Chance-anchored NeuroSafe for dual dominance."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.run_fair_tune import run_agent_episodes
from neurosafe_cbo.config import CFG

OUT = os.path.join(ROOT, "results", "day7_dual_anchor")


def main():
    os.makedirs(OUT, exist_ok=True)
    base = dict(CFG)
    base.update(
        {
            "ALPHA": 0.15,
            "NS_ALPHA": 0.15,
            "SAFE_Z_MARGIN": 0.0,
            "LAMBDA_CVAR": 0.0,
            "SCORE_MODE": "mean",
            "DECOUPLE_HARM": True,
            "NEURO_GAIN": 0.0,
            "SAFE_FLOOR_BONUS": 0.0,
            "DOMINANCE_UTIL_EPS": 0.05,
            "PC_UTIL_MIX": 0.35,
        }
    )

    frames = []
    for name in ["Chance", "NeuroSafeCBO", "RiskNeutral", "CVaR", "Greedy", "Random"]:
        frames.append(
            run_agent_episodes(
                name,
                base,
                base["SHIFT_MAGS"],
                base["OBS_NOISES"],
                10,
                20,
                f"anchor-{name}",
            )
        )
    final = pd.concat(frames, ignore_index=True)
    final.to_csv(os.path.join(OUT, "results_final.csv"), index=False)
    summary = (
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
    summary.to_csv(os.path.join(OUT, "summary_final.csv"), index=False)
    print(summary.to_string(index=False))

    ch = summary[summary.agent == "Chance"].iloc[0]
    ns = summary[summary.agent == "NeuroSafeCBO"].iloc[0]
    dual = (ns.mean_return >= ch.mean_return - 1e-9) and (ns.mean_viol <= ch.mean_viol + 1e-9)
    # strict dual
    dual_strict = (ns.mean_return > ch.mean_return) and (ns.mean_viol < ch.mean_viol)
    verdict = {
        "dual_weak": bool(dual),
        "dual_strict": bool(dual_strict),
        "chance": {"return": float(ch.mean_return), "viol": float(ch.mean_viol)},
        "neurosafe": {"return": float(ns.mean_return), "viol": float(ns.mean_viol)},
    }
    with open(os.path.join(OUT, "dual_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print("\nWeak dual (>=, <=):", dual)
    print("Strict dual (>, <):", dual_strict)


if __name__ == "__main__":
    main()
