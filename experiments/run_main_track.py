"""A* main-track experiment runner (Month 1).

Usage:
  python experiments/run_main_track.py --profile smoke
  python experiments/run_main_track.py --profile week1 --tasks regret,scm
  python experiments/run_main_track.py --profile scm50 --tasks scm --seed-start 0 --seed-end 50
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.main_track_common import (  # noqa: E402
    MAIN_AGENTS,
    OUT_ROOT,
    apply_profile,
    frozen_base,
    run_scm_grid,
    run_episode_tracked,
    save_json,
    seed_stats_vs_chance,
    summarize_agents,
)
from neurosafe_cbo.agent import make_agent  # noqa: E402
from neurosafe_cbo.env_ihdp_bandit import IHDPSafeBanditEnv  # noqa: E402


def task_regret(cfg, out_dir, seeds, n_eps):
    os.makedirs(out_dir, exist_ok=True)
    agents = ["Chance", "NeuroSafeCBO"]
    ep_df, step_df = run_scm_grid(cfg, agents, seeds, n_eps, desc="regret")
    ep_df.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
    step_df.to_csv(os.path.join(out_dir, "regret_by_step.csv"), index=False)

    if not step_df.empty:
        curve = (
            step_df.groupby(["agent", "step"], as_index=False)
            .agg(mean_cum_regret=("cum_regret", "mean"), se=("cum_regret", "std"))
        )
        curve.to_csv(os.path.join(out_dir, "regret_curve.csv"), index=False)
        plt.figure(figsize=(7, 4))
        for name, g in curve.groupby("agent"):
            g = g.sort_values("step")
            plt.plot(g["step"], g["mean_cum_regret"], label=name)
        plt.xlabel("Environment step")
        plt.ylabel("Mean cumulative constrained regret")
        plt.title("Constrained regret vs horizon")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "regret_curve.png"), dpi=160)
        plt.close()

    stats = seed_stats_vs_chance(ep_df)
    save_json(os.path.join(out_dir, "seed_stats.json"), stats)
    summary = summarize_agents(ep_df)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("REGRET TASK:", stats)
    return ep_df


def task_scm(cfg, out_dir, seeds, n_eps):
    os.makedirs(out_dir, exist_ok=True)
    ep_df, step_df = run_scm_grid(cfg, MAIN_AGENTS, seeds, n_eps, desc="scm")
    ep_df.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
    if not step_df.empty:
        step_df.to_csv(os.path.join(out_dir, "regret_by_step.csv"), index=False)
    summary = summarize_agents(ep_df)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    stats = seed_stats_vs_chance(ep_df)
    save_json(os.path.join(out_dir, "seed_stats.json"), stats)
    print(summary.to_string(index=False))
    print("SEED STATS:", stats)
    return ep_df


def task_pareto(cfg, out_dir, seeds, n_eps):
    ep_df = task_scm(cfg, out_dir, seeds, min(n_eps, 5))
    g = ep_df.groupby("agent", as_index=False).agg(
        mean_viol=("viol", "mean"), mean_return=("return", "mean")
    )
    plt.figure(figsize=(6, 4))
    for _, r in g.iterrows():
        plt.scatter(r["mean_viol"], r["mean_return"], s=60)
        plt.text(r["mean_viol"], r["mean_return"], r["agent"], fontsize=8)
    plt.xlabel("Mean violation")
    plt.ylabel("Mean return")
    plt.title("Safety–utility Pareto (SCM)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto.png"), dpi=160)
    plt.close()


def task_kstudy(cfg, out_dir, seeds, n_eps):
    os.makedirs(out_dir, exist_ok=True)
    k_values = cfg.get("K_VALUES", [5, 11, 21, 41])
    rows = []
    for k in k_values:
        kcfg = dict(cfg)
        kcfg["N_ACTIONS"] = int(k)
        ep_df, _ = run_scm_grid(
            kcfg,
            ["Chance", "NeuroSafeCBO"],
            seeds,
            n_eps,
            desc=f"K={k}",
        )
        ep_df = ep_df.assign(k=int(k))
        rows.append(ep_df)
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, "kstudy_episodes.csv"), index=False)
    summary = (
        all_df.groupby(["k", "agent"], as_index=False)
        .agg(
            mean_return=("return", "mean"),
            mean_viol=("viol", "mean"),
            mean_creg=("avg_constrained_regret", "mean"),
        )
    )
    summary.to_csv(os.path.join(out_dir, "kstudy_summary.csv"), index=False)
    print(summary.to_string(index=False))
    return all_df


def task_ihdp(cfg, out_dir, seeds, n_eps):
    os.makedirs(out_dir, exist_ok=True)
    actions = np.array([0.0, 1.0])
    rows = []
    agents = ["Chance", "NeuroSafeCBO", "RiskNeutral", "Greedy", "Random"]
    total = len(seeds) * n_eps * len(agents)
    from tqdm import tqdm

    pbar = tqdm(total=total, desc="ihdp-bandit")
    for seed in seeds:
        for ep in range(n_eps):
            for name in agents:
                env = IHDPSafeBanditEnv(seed=1000 * int(seed) + ep, horizon=int(cfg["HORIZON"]))
                agent = make_agent(name, seed=int(seed), cfg=cfg, actions=actions)
                m, _ = run_episode_tracked(
                    env, agent, actions, cfg, compute_regret=False
                )
                rows.append({"agent": name, "seed": int(seed), "episode": ep, **m})
                pbar.update(1)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ihdp_episodes.csv"), index=False)
    summary = summarize_agents(df)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(summary.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke", choices=list(__import__(
        "experiments.main_track_common", fromlist=["PROFILES"]
    ).PROFILES.keys()))
    parser.add_argument(
        "--tasks",
        default="regret,scm",
        help="Comma-separated: regret, scm, pareto, kstudy, ihdp",
    )
    parser.add_argument("--out", default=OUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-end", type=int, default=None)
    args = parser.parse_args()

    cfg = apply_profile(frozen_base(), args.profile)
    n_seeds = int(cfg["SEEDS"])
    seed_start = 0 if args.seed_start is None else args.seed_start
    seed_end = n_seeds if args.seed_end is None else args.seed_end
    seeds = list(range(seed_start, seed_end))
    n_eps = int(cfg["EPISODES"])

    subdir = args.profile if args.seed_start is None else f"{args.profile}_s{seed_start}_{seed_end}"
    out_dir = os.path.join(args.out, subdir)
    os.makedirs(out_dir, exist_ok=True)
    save_json(
        os.path.join(out_dir, "config.json"),
        {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))},
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task == "regret":
            task_regret(cfg, os.path.join(out_dir, "regret"), seeds, n_eps)
        elif task == "scm":
            task_scm(cfg, out_dir, seeds, n_eps)
        elif task == "pareto":
            task_pareto(cfg, os.path.join(out_dir, "pareto"), seeds, n_eps)
        elif task == "kstudy":
            task_kstudy(cfg, os.path.join(out_dir, "kstudy"), seeds, n_eps)
        elif task == "ihdp":
            task_ihdp(cfg, os.path.join(out_dir, "ihdp"), seeds, n_eps)
        else:
            raise ValueError(f"Unknown task: {task}")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
