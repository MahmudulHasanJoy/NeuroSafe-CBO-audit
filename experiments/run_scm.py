"""Run nonstationary SCM benchmark with baselines and ablations."""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running as script from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurosafe_cbo.agent import AGENT_SPECS, make_agent
from neurosafe_cbo.config import ACTIONS, CFG
from neurosafe_cbo.env import NonstationarySCMEnv
from neurosafe_cbo.metrics import constrained_regret, episode_metrics


MAIN_AGENTS = [
    "Random",
    "Greedy",
    "RiskNeutral",
    "Chance",
    "CVaR",
    "NeuroSafeCBO",
]

ABLATION_AGENTS = [
    "NeuroSafeCBO",
    "NS_noPC",
    "NS_noNeuro",
    "NS_noChance",
    "NS_noCVaR",
]


def run_episode(env, agent, cfg, compute_regret: bool = True):
    obs = env.reset()
    U_hist, H_hist, regrets = [], [], []
    regret_every = int(cfg.get("REGRET_EVERY", 10))
    for t in range(cfg["HORIZON"]):
        a, z = agent.act(obs)
        if (
            compute_regret
            and hasattr(agent, "actions")
            and (t % regret_every == 0)
        ):
            mu_u, p_viol = env.oracle_action_stats(
                ACTIONS, cfg["ORACLE_MC"], cfg["TAU_HARM"]
            )
            chosen_idx = int(np.argmin(np.abs(ACTIONS - a)))
            regrets.append(constrained_regret(mu_u, p_viol, chosen_idx, cfg["ALPHA"]))
        obs, U, H, _, _ = env.step(a)
        agent.observe(a, z, U, H)
        U_hist.append(U)
        H_hist.append(H)
    m = episode_metrics(U_hist, H_hist, cfg["TAU_HARM"])
    m["avg_constrained_regret"] = float(np.mean(regrets)) if regrets else np.nan
    return m


def run_benchmark(agent_names, out_dir, cfg=None, tag="main"):
    cfg = cfg if cfg is not None else CFG
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    total = (
        len(cfg["SHIFT_MAGS"])
        * len(cfg["OBS_NOISES"])
        * cfg["SEEDS"]
        * cfg["EPISODES"]
        * len(agent_names)
    )
    pbar = tqdm(total=total, desc=f"SCM-{tag}")
    t0 = time.time()

    for sm in cfg["SHIFT_MAGS"]:
        for on in cfg["OBS_NOISES"]:
            for seed in range(cfg["SEEDS"]):
                for ep in range(cfg["EPISODES"]):
                    for name in agent_names:
                        env = NonstationarySCMEnv(
                            cfg["SHIFT_AT"],
                            sm,
                            on,
                            seed=1000 * seed + ep,
                            harm_mode=cfg.get("HARM_MODE", "tradeoff"),
                        )
                        agent = make_agent(name, seed=seed, cfg=cfg, actions=ACTIONS)
                        # Oracle MC is expensive; enable for main agents only on subset
                        use_oracle = bool(cfg.get("COMPUTE_REGRET", True)) and name in MAIN_AGENTS
                        m = run_episode(env, agent, cfg, compute_regret=use_oracle)
                        rows.append(
                            {
                                "tag": tag,
                                "shift_mag": sm,
                                "obs_noise": on,
                                "seed": seed,
                                "episode": ep,
                                "agent": name,
                                **m,
                            }
                        )
                        pbar.update(1)
    pbar.close()
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"results_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} in {time.time() - t0:.1f}s")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("agent", as_index=False)
        .agg(
            mean_return=("return", "mean"),
            std_return=("return", "std"),
            mean_viol=("viol", "mean"),
            std_viol=("viol", "std"),
            mean_cvar=("cvar", "mean"),
            mean_creg=("avg_constrained_regret", "mean"),
        )
        .sort_values("mean_viol")
    )
    return g


def plot_pareto(df: pd.DataFrame, out_dir: str, tag: str):
    for (sm, on), g_all in df.groupby(["shift_mag", "obs_noise"]):
        g = (
            g_all.groupby("agent", as_index=False)
            .agg(mean_viol=("viol", "mean"), mean_return=("return", "mean"))
        )
        plt.figure(figsize=(8, 5))
        plt.scatter(g["mean_viol"], g["mean_return"], s=60)
        for _, r in g.iterrows():
            plt.text(r["mean_viol"] + 0.002, r["mean_return"], r["agent"], fontsize=8)
        plt.xlabel("Mean violation rate")
        plt.ylabel("Mean return")
        plt.title(f"Pareto | shift={sm}, noise={on} | {tag}")
        plt.tight_layout()
        fn = os.path.join(out_dir, f"pareto_{tag}_sm{sm}_on{on}.png")
        plt.savefig(fn, dpi=160)
        plt.close()


def plot_bars(summary: pd.DataFrame, out_dir: str, tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    agents = summary["agent"].tolist()
    x = np.arange(len(agents))
    axes[0].bar(x, summary["mean_return"], yerr=summary["std_return"].fillna(0), capsize=3)
    axes[0].set_title("Return (↑)")
    axes[1].bar(x, summary["mean_viol"], yerr=summary["std_viol"].fillna(0), capsize=3)
    axes[1].set_title("Violation (↓)")
    axes[2].bar(x, summary["mean_cvar"])
    axes[2].set_title("Harm CVaR (↓)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=8)
    fig.suptitle(f"NeuroSafe-CBO SCM benchmark ({tag})")
    fig.tight_layout()
    fn = os.path.join(out_dir, f"bars_{tag}.png")
    fig.savefig(fn, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "day7"))
    parser.add_argument("--quick", action="store_true", help="Smaller config for smoke test")
    parser.add_argument("--no-regret", action="store_true")
    args = parser.parse_args()

    cfg = dict(CFG)
    if args.quick:
        cfg.update(
            {
                "SEEDS": 2,
                "EPISODES": 3,
                "HORIZON": 20,
                "ORACLE_MC": 8,
                "SHIFT_MAGS": [0.35],
                "OBS_NOISES": [0.15],
            }
        )
    if args.no_regret:
        cfg["COMPUTE_REGRET"] = False

    out = args.out
    os.makedirs(out, exist_ok=True)

    df_main = run_benchmark(MAIN_AGENTS, out, cfg, tag="main")
    sum_main = summarize(df_main)
    sum_main.to_csv(os.path.join(out, "summary_main.csv"), index=False)
    plot_pareto(df_main, out, "main")
    plot_bars(sum_main, out, "main")
    print("\n=== MAIN SUMMARY ===")
    print(sum_main.to_string(index=False))

    df_ab = run_benchmark(ABLATION_AGENTS, out, cfg, tag="ablation")
    sum_ab = summarize(df_ab)
    sum_ab.to_csv(os.path.join(out, "summary_ablation.csv"), index=False)
    plot_pareto(df_ab, out, "ablation")
    plot_bars(sum_ab, out, "ablation")
    print("\n=== ABLATION SUMMARY ===")
    print(sum_ab.to_string(index=False))

    # By difficulty for main
    by = (
        df_main.groupby(["shift_mag", "obs_noise", "agent"], as_index=False)
        .agg(mean_return=("return", "mean"), mean_viol=("viol", "mean"))
    )
    by.to_csv(os.path.join(out, "summary_main_by_difficulty.csv"), index=False)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
