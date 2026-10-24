"""Fair hyperparameter sweep for NeuroSafe-CBO only.

Protocol
--------
1) Freeze Chance / other baselines at default CFG.
2) Tune ONLY NeuroSafe-CBO knobs on a held-out difficulty
   (shift=0.15, noise=0.15), smaller seed/episode budget.
3) Selection rule (pre-registered):
      - prefer configs with viol <= Chance_viol + 0.005
      - among them, maximize return
      - if none, maximize return - 40 * viol
4) Re-evaluate winners + all baselines on the FULL difficulty grid.
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

from neurosafe_cbo.agent import make_agent
from neurosafe_cbo.config import ACTIONS, CFG
from neurosafe_cbo.env import NonstationarySCMEnv
from neurosafe_cbo.metrics import episode_metrics

OUT = os.path.join(ROOT, "results", "day7_tuned")

# Held-out tuning slice (not the hardest setting)
TUNE_SM, TUNE_ON = 0.15, 0.15
TUNE_SEEDS, TUNE_EPS = 4, 8

# Final eval (same as main paper table)
FINAL_SEEDS, FINAL_EPS = 8, 12

# NeuroSafe-only grid (Chance ignores neuromod/CVaR; ALPHA override is NS-only)
GRID = {
    "NEURO_GAIN": [0.0, 0.2, 0.4, 0.6],
    "LAMBDA_CVAR": [0.8, 1.2, 1.6],
    "ALPHA": [0.05, 0.08, 0.10],  # applied only to NeuroSafeCBO cfg
}


def run_agent_episodes(name, cfg, shift_mags, obs_noises, n_seeds, n_eps, desc):
    rows = []
    total = len(shift_mags) * len(obs_noises) * n_seeds * n_eps
    pbar = tqdm(total=total, desc=desc)
    for sm in shift_mags:
        for on in obs_noises:
            for seed in range(n_seeds):
                for ep in range(n_eps):
                    env = NonstationarySCMEnv(
                        cfg["SHIFT_AT"], sm, on, seed=1000 * seed + ep, harm_mode="tradeoff"
                    )
                    agent = make_agent(name, seed=seed, cfg=cfg, actions=ACTIONS)
                    obs = env.reset()
                    U_hist, H_hist = [], []
                    for _ in range(cfg["HORIZON"]):
                        a, z = agent.act(obs)
                        obs, U, H, _, _ = env.step(a)
                        agent.observe(a, z, U, H)
                        U_hist.append(U)
                        H_hist.append(H)
                    m = episode_metrics(U_hist, H_hist, cfg["TAU_HARM"])
                    rows.append(
                        {
                            "agent": name,
                            "shift_mag": sm,
                            "obs_noise": on,
                            "seed": seed,
                            "episode": ep,
                            **m,
                        }
                    )
                    pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "mean_return": float(df["return"].mean()),
        "std_return": float(df["return"].std()),
        "mean_viol": float(df["viol"].mean()),
        "std_viol": float(df["viol"].std()),
        "mean_cvar": float(df["cvar"].mean()),
    }


def select_best(rows, chance_viol):
    """Pre-registered selection rule."""
    tol = 0.005
    feasible = [r for r in rows if r["mean_viol"] <= chance_viol + tol]
    if feasible:
        best = max(feasible, key=lambda r: r["mean_return"])
        rule = f"max return among viol <= Chance+{tol}"
    else:
        best = max(rows, key=lambda r: r["mean_return"] - 40.0 * r["mean_viol"])
        rule = "max return - 40*viol (no config met Chance+tol)"
    best = dict(best)
    best["selection_rule"] = rule
    best["chance_viol_tune"] = chance_viol
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    base = dict(CFG)
    base["COMPUTE_REGRET"] = False
    t0 = time.time()

    # --- Stage 1: Chance reference on tuning slice (frozen defaults) ---
    print("=== Tuning slice: Chance (frozen) ===")
    df_chance = run_agent_episodes(
        "Chance",
        base,
        [TUNE_SM],
        [TUNE_ON],
        TUNE_SEEDS,
        TUNE_EPS,
        "tune-Chance",
    )
    chance_sum = summarize(df_chance)
    print("Chance tune:", chance_sum)

    # --- Stage 2: NeuroSafe grid on same slice ---
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"=== Tuning NeuroSafeCBO over {len(combos)} configs ===")
    tune_rows = []
    for vals in combos:
        overrides = dict(zip(keys, vals))
        cfg_ns = dict(base)
        cfg_ns.update(overrides)
        tag = ",".join(f"{k}={overrides[k]}" for k in keys)
        df = run_agent_episodes(
            "NeuroSafeCBO",
            cfg_ns,
            [TUNE_SM],
            [TUNE_ON],
            TUNE_SEEDS,
            TUNE_EPS,
            f"tune-NS[{tag}]",
        )
        s = summarize(df)
        s.update(overrides)
        s["tag"] = tag
        tune_rows.append(s)
        print(
            f"  {tag} -> ret={s['mean_return']:.2f} viol={s['mean_viol']:.4f}"
        )

    tune_df = pd.DataFrame(tune_rows)
    tune_df.to_csv(os.path.join(OUT, "sweep_tuning.csv"), index=False)
    best = select_best(tune_rows, chance_sum["mean_viol"])
    with open(os.path.join(OUT, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "chance_tune": chance_sum}, f, indent=2)
    print("\nSelected:", best)

    # --- Stage 3: full-grid final eval ---
    print("\n=== Final eval on full difficulty grid ===")
    cfg_best = dict(base)
    for k in keys:
        cfg_best[k] = best[k]

    final_frames = []
    # Baselines with frozen base cfg
    for name in ["Random", "Greedy", "RiskNeutral", "Chance", "CVaR"]:
        df = run_agent_episodes(
            name,
            base,
            base["SHIFT_MAGS"],
            base["OBS_NOISES"],
            FINAL_SEEDS,
            FINAL_EPS,
            f"final-{name}",
        )
        final_frames.append(df)

    # NeuroSafe with tuned cfg only
    df_ns = run_agent_episodes(
        "NeuroSafeCBO",
        cfg_best,
        base["SHIFT_MAGS"],
        base["OBS_NOISES"],
        FINAL_SEEDS,
        FINAL_EPS,
        "final-NeuroSafeCBO",
    )
    final_frames.append(df_ns)

    final = pd.concat(final_frames, ignore_index=True)
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

    # Pareto plot
    plt.figure(figsize=(8, 5))
    plt.scatter(summary["mean_viol"], summary["mean_return"], s=70)
    for _, r in summary.iterrows():
        plt.text(r["mean_viol"] + 0.002, r["mean_return"], r["agent"], fontsize=8)
    plt.xlabel("Mean violation rate")
    plt.ylabel("Mean return")
    plt.title("Tuned NeuroSafe-CBO vs frozen baselines")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pareto_final.png"), dpi=160)
    plt.close()

    # Sensitivity heatmap-ish table already in sweep_tuning.csv
    print(f"\nDone in {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
