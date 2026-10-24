"""Focused dual-dominance search with safer NeuroSafe architecture."""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.run_fair_tune import run_agent_episodes, summarize
from neurosafe_cbo.config import CFG

OUT = os.path.join(ROOT, "results", "day7_dual2")

GRID = {
    "NS_ALPHA": [0.06, 0.08, 0.10],
    "SAFE_Z_MARGIN": [0.0, 0.2, 0.4],
    "LAMBDA_CVAR": [0.0, 0.4],
    "SCORE_MODE": ["mean"],
    "DECOUPLE_HARM": [True],
    "SAFE_WARMUP": [True],
    "SAFE_INTERSECT": [False, True],
    "NEURO_GAIN": [0.0],
    "UTIL_TIE_EPS": [0.05, 0.15],
}


def main():
    os.makedirs(OUT, exist_ok=True)
    base = dict(CFG)
    base["ALPHA"] = 0.10
    base["BASE_BETA"] = 0.8
    t0 = time.time()

    df_c = run_agent_episodes("Chance", base, [0.15], [0.15], 6, 12, "tune-Chance")
    csum = summarize(df_c)
    print("Chance tune", csum)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    rows = []
    for vals in combos:
        ov = dict(zip(keys, vals))
        cfg = dict(base)
        cfg.update(ov)
        tag = (
            f"a={ov['NS_ALPHA']},m={ov['SAFE_Z_MARGIN']},L={ov['LAMBDA_CVAR']},"
            f"I={ov['SAFE_INTERSECT']},tie={ov['UTIL_TIE_EPS']}"
        )
        df = run_agent_episodes("NeuroSafeCBO", cfg, [0.15], [0.15], 6, 12, tag)
        s = summarize(df)
        s.update(ov)
        s["tag"] = tag
        s["dual"] = (s["mean_return"] >= csum["mean_return"]) and (
            s["mean_viol"] <= csum["mean_viol"]
        )
        rows.append(s)
        print(("DUAL" if s["dual"] else "    "), tag, f"ret={s['mean_return']:.2f}", f"viol={s['mean_viol']:.4f}")

    tune = pd.DataFrame(rows)
    tune.to_csv(os.path.join(OUT, "sweep_tuning.csv"), index=False)
    duals = tune[tune.dual]
    print(f"\nDual on tune: {len(duals)}/{len(tune)}")

    if len(duals):
        best = duals.sort_values(["mean_return", "mean_viol"], ascending=[False, True]).iloc[0].to_dict()
        rule = "dual_on_tune"
    else:
        # minimize viol gap then maximize return gap
        tune = tune.assign(
            dv=(tune.mean_viol - csum["mean_viol"]).clip(lower=0),
            dr=tune.mean_return - csum["mean_return"],
        )
        best = tune.sort_values(["dv", "dr"], ascending=[True, False]).iloc[0].to_dict()
        rule = "closest"

    best["selection_rule"] = rule
    best["chance_tune"] = csum
    with open(os.path.join(OUT, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, default=float)
    print("Selected", rule, {k: best[k] for k in keys}, "ret", best["mean_return"], "viol", best["mean_viol"])

    cfg_best = dict(base)
    for k in keys:
        cfg_best[k] = best[k]

    frames = []
    for name in ["Chance", "NeuroSafeCBO", "RiskNeutral", "CVaR", "Greedy", "Random"]:
        cfg = cfg_best if name == "NeuroSafeCBO" else base
        frames.append(
            run_agent_episodes(name, cfg, base["SHIFT_MAGS"], base["OBS_NOISES"], 8, 15, f"final-{name}")
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
        )
        .sort_values("mean_viol")
    )
    summary.to_csv(os.path.join(OUT, "summary_final.csv"), index=False)
    print(summary.to_string(index=False))

    ch = summary[summary.agent == "Chance"].iloc[0]
    ns = summary[summary.agent == "NeuroSafeCBO"].iloc[0]
    dual = (ns.mean_return >= ch.mean_return) and (ns.mean_viol <= ch.mean_viol)
    verdict = {
        "dual_final": bool(dual),
        "chance": {"return": float(ch.mean_return), "viol": float(ch.mean_viol)},
        "neurosafe": {"return": float(ns.mean_return), "viol": float(ns.mean_viol)},
        "best": {k: best[k] for k in keys},
        "rule": rule,
        "n_dual_tune": int(len(duals)),
    }
    with open(os.path.join(OUT, "dual_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print("\nDUAL FINAL:", dual)
    print(f"Done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
