"""Seek dual dominance: return >= Chance and viol <= Chance.

Algorithm changes (NeuroSafe only):
  - DECOUPLE_HARM: harm filter uses z=0 (Chance-calibrated)
  - SCORE_MODE=mean: exploit mu_U inside feasible set
  - NS_ALPHA / SAFE_Z_MARGIN: conservative safety
  - NS_BASE_BETA unused when SCORE_MODE=mean

Chance remains frozen at default CFG.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_fair_tune import run_agent_episodes, summarize
from neurosafe_cbo.config import CFG

OUT = os.path.join(ROOT, "results", "day7_dual")

TUNE_SM, TUNE_ON = 0.15, 0.15
TUNE_SEEDS, TUNE_EPS = 5, 10
FINAL_SEEDS, FINAL_EPS = 8, 15

GRID = {
    "NS_ALPHA": [0.04, 0.06, 0.08],
    "SAFE_Z_MARGIN": [0.0, 0.25, 0.5],
    "LAMBDA_CVAR": [0.4, 0.8, 1.2],
    "SCORE_MODE": ["mean"],
    "DECOUPLE_HARM": [True],
    "NEURO_GAIN": [0.0],
}


def select_dual(rows, chance_r, chance_v):
    """Strict dual dominance on tuning slice; else closest safe improve."""
    dual = [
        r
        for r in rows
        if (r["mean_return"] >= chance_r - 1e-9) and (r["mean_viol"] <= chance_v + 1e-9)
    ]
    if dual:
        best = max(dual, key=lambda r: (r["mean_return"] - chance_r) - 5.0 * (r["mean_viol"] - chance_v))
        # prefer max return among dual, with viol tie-break
        best = max(dual, key=lambda r: (r["mean_return"], -r["mean_viol"]))
        rule = "dual_dominance_on_tune"
    else:
        # minimize viol excess, then maximize return surplus
        def key(r):
            return (
                -(max(0.0, r["mean_viol"] - chance_v)),
                r["mean_return"] - chance_r,
            )

        best = max(rows, key=key)
        rule = "no_dual_on_tune_closest"
    best = dict(best)
    best["selection_rule"] = rule
    best["chance_return_tune"] = chance_r
    best["chance_viol_tune"] = chance_v
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    base = dict(CFG)
    # freeze Chance-facing defaults
    base["ALPHA"] = 0.10
    base["BASE_BETA"] = 0.8
    t0 = time.time()

    print("=== Chance on tuning slice (frozen) ===")
    df_c = run_agent_episodes(
        "Chance", base, [TUNE_SM], [TUNE_ON], TUNE_SEEDS, TUNE_EPS, "tune-Chance"
    )
    csum = summarize(df_c)
    print(csum)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"=== NeuroSafe grid ({len(combos)} configs) ===")
    rows = []
    for vals in combos:
        ov = dict(zip(keys, vals))
        cfg = dict(base)
        cfg.update(ov)
        tag = ",".join(f"{k}={ov[k]}" for k in ["NS_ALPHA", "SAFE_Z_MARGIN", "LAMBDA_CVAR"])
        df = run_agent_episodes(
            "NeuroSafeCBO", cfg, [TUNE_SM], [TUNE_ON], TUNE_SEEDS, TUNE_EPS, f"tune-{tag}"
        )
        s = summarize(df)
        s.update(ov)
        s["tag"] = tag
        s["dual"] = (s["mean_return"] >= csum["mean_return"]) and (
            s["mean_viol"] <= csum["mean_viol"]
        )
        rows.append(s)
        mark = "DUAL" if s["dual"] else "    "
        print(f"{mark} {tag} ret={s['mean_return']:.2f} viol={s['mean_viol']:.4f}")

    tune_df = pd.DataFrame(rows)
    tune_df.to_csv(os.path.join(OUT, "sweep_tuning.csv"), index=False)
    best = select_dual(rows, csum["mean_return"], csum["mean_viol"])
    with open(os.path.join(OUT, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "chance_tune": csum, "n_dual": int(tune_df["dual"].sum())}, f, indent=2)
    print("\nSelected:", best)
    print("Dual configs on tune:", int(tune_df["dual"].sum()), "/", len(tune_df))

    # Final full-grid eval
    cfg_best = dict(base)
    for k in keys:
        cfg_best[k] = best[k]

    frames = []
    for name in ["Random", "Greedy", "RiskNeutral", "Chance", "CVaR"]:
        frames.append(
            run_agent_episodes(
                name, base, base["SHIFT_MAGS"], base["OBS_NOISES"], FINAL_SEEDS, FINAL_EPS, f"final-{name}"
            )
        )
    frames.append(
        run_agent_episodes(
            "NeuroSafeCBO",
            cfg_best,
            base["SHIFT_MAGS"],
            base["OBS_NOISES"],
            FINAL_SEEDS,
            FINAL_EPS,
            "final-NeuroSafeCBO",
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
    print("\n=== FINAL SUMMARY ===")
    print(summary.to_string(index=False))

    ch = summary[summary.agent == "Chance"].iloc[0]
    ns = summary[summary.agent == "NeuroSafeCBO"].iloc[0]
    dual_final = (ns.mean_return >= ch.mean_return) and (ns.mean_viol <= ch.mean_viol)
    print("\nDUAL DOMINANCE ON FINAL GRID:", dual_final)
    print(f"  Chance  ret={ch.mean_return:.3f} viol={ch.mean_viol:.4f}")
    print(f"  NeuroSafe ret={ns.mean_return:.3f} viol={ns.mean_viol:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(summary["mean_viol"], summary["mean_return"], s=70)
    for _, r in summary.iterrows():
        plt.text(r["mean_viol"] + 0.002, r["mean_return"], r["agent"], fontsize=8)
    plt.xlabel("Mean violation rate")
    plt.ylabel("Mean return")
    plt.title("Dual-dominance search: tuned NeuroSafe vs frozen baselines")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pareto_final.png"), dpi=160)
    plt.close()

    with open(os.path.join(OUT, "dual_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dual_final": bool(dual_final),
                "chance": {"return": float(ch.mean_return), "viol": float(ch.mean_viol)},
                "neurosafe": {"return": float(ns.mean_return), "viol": float(ns.mean_viol)},
                "best_config": best,
            },
            f,
            indent=2,
        )
    print(f"Done in {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
