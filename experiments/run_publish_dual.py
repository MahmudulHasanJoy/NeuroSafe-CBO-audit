"""Publishable weak-dual package vs Chance (frozen knobs)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from neurosafe_cbo.agent import make_agent
from neurosafe_cbo.config import ACTIONS, CFG
from neurosafe_cbo.env import NonstationarySCMEnv
from neurosafe_cbo.metrics import episode_metrics

OUT = os.path.join(ROOT, "results", "day7_publish")


def frozen_base():
    base = dict(CFG)
    base.update(
        {
            "ALPHA": 0.15,
            "NS_ALPHA": 0.15,
            "LAMBDA_CVAR": 0.0,
            "NEURO_GAIN": 0.0,
            "SAFE_FLOOR_BONUS": 0.0,
            "DOMINANCE_UTIL_EPS": 0.05,
            "PC_UTIL_MIX": 0.35,
            "DECOUPLE_HARM": True,
            "SCORE_MODE": "mean",
            "ANCHOR_WITHOUT_PC": False,
        }
    )
    return base


def dual_flags(ns_r, ns_v, ch_r, ch_v):
    return {
        "dual_weak": bool((ns_r >= ch_r - 1e-12) and (ns_v <= ch_v + 1e-12)),
        "dual_strict": bool((ns_r > ch_r) and (ns_v < ch_v)),
        "dR": float(ns_r - ch_r),
        "dV": float(ns_v - ch_v),
    }


def run_agents_seedlist(name, cfg, seeds, n_eps, desc):
    rows = []
    total = len(cfg["SHIFT_MAGS"]) * len(cfg["OBS_NOISES"]) * len(seeds) * n_eps
    pbar = tqdm(total=total, desc=desc)
    for sm in cfg["SHIFT_MAGS"]:
        for on in cfg["OBS_NOISES"]:
            for seed in seeds:
                for ep in range(n_eps):
                    env = NonstationarySCMEnv(
                        cfg["SHIFT_AT"],
                        sm,
                        on,
                        seed=1000 * int(seed) + ep,
                        harm_mode="tradeoff",
                    )
                    agent = make_agent(name, seed=int(seed), cfg=cfg, actions=ACTIONS)
                    obs = env.reset()
                    U_hist, H_hist = [], []
                    for _ in range(int(cfg["HORIZON"])):
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
                            "seed": int(seed),
                            "episode": ep,
                            **m,
                        }
                    )
                    pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def means(df, agent):
    sub = df[df["agent"] == agent]
    return float(sub["return"].mean()), float(sub["viol"].mean())


def bootstrap_ci(x, n_boot=5000, alpha=0.05):
    rng = np.random.RandomState(0)
    x = np.asarray(x, dtype=float)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(x.mean()), float(lo), float(hi)


def seed_level_stats(df):
    g = df.groupby(["agent", "seed"], as_index=False).agg(
        ret=("return", "mean"), viol=("viol", "mean")
    )
    ch = g[g.agent == "Chance"].set_index("seed").sort_index()
    ns = g[g.agent == "NeuroSafeCBO"].set_index("seed").sort_index()
    common = ch.index.intersection(ns.index)
    dR = (ns.loc[common, "ret"] - ch.loc[common, "ret"]).to_numpy()
    dV = (ns.loc[common, "viol"] - ch.loc[common, "viol"]).to_numpy()
    mean_dR, lo_dR, hi_dR = bootstrap_ci(dR)
    mean_dV, lo_dV, hi_dV = bootstrap_ci(dV)
    try:
        w_stat, w_p = stats.wilcoxon(dR, zero_method="wilcox", alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    t_stat, t_p = stats.ttest_rel(ns.loc[common, "ret"], ch.loc[common, "ret"])
    t_p_one = float(t_p / 2.0) if t_stat > 0 else float(1.0 - t_p / 2.0)
    by_seed = pd.DataFrame(
        {
            "seed": list(common),
            "ch_return": ch.loc[common, "ret"].to_numpy(),
            "ns_return": ns.loc[common, "ret"].to_numpy(),
            "ch_viol": ch.loc[common, "viol"].to_numpy(),
            "ns_viol": ns.loc[common, "viol"].to_numpy(),
            "dR": dR,
            "dV": dV,
        }
    )
    summary = {
        "n_seeds": int(len(common)),
        "mean_dR": mean_dR,
        "dR_ci95": [lo_dR, hi_dR],
        "mean_dV": mean_dV,
        "dV_ci95": [lo_dV, hi_dV],
        "wilcoxon_dR_greater": {"stat": w_stat, "pvalue": w_p},
        "ttest_rel_dR_greater": {"stat": float(t_stat), "pvalue": t_p_one},
        "chance_return": float(ch.loc[common, "ret"].mean()),
        "ns_return": float(ns.loc[common, "ret"].mean()),
        "chance_viol": float(ch.loc[common, "viol"].mean()),
        "ns_viol": float(ns.loc[common, "viol"].mean()),
        **dual_flags(
            float(ns.loc[common, "ret"].mean()),
            float(ns.loc[common, "viol"].mean()),
            float(ch.loc[common, "ret"].mean()),
            float(ch.loc[common, "viol"].mean()),
        ),
    }
    return {"by_seed": by_seed, "summary": summary}


def main():
    os.makedirs(OUT, exist_ok=True)
    base = frozen_base()

    train_seeds = list(range(0, 5))
    test_seeds = list(range(5, 10))
    held = {
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "cfg": {
            "SAFE_FLOOR_BONUS": 0.0,
            "DOMINANCE_UTIL_EPS": 0.05,
            "PC_UTIL_MIX": 0.35,
            "ALPHA": 0.15,
        },
    }
    for split, seeds in [("train", train_seeds), ("test", test_seeds)]:
        df = pd.concat(
            [
                run_agents_seedlist(n, base, seeds, 20, f"heldout-{split}-{n}")
                for n in ["Chance", "NeuroSafeCBO"]
            ],
            ignore_index=True,
        )
        df.to_csv(os.path.join(OUT, f"heldout_{split}_episodes.csv"), index=False)
        ch_r, ch_v = means(df, "Chance")
        ns_r, ns_v = means(df, "NeuroSafeCBO")
        held[split] = {
            "chance": {"return": ch_r, "viol": ch_v},
            "neurosafe": {"return": ns_r, "viol": ns_v},
            **dual_flags(ns_r, ns_v, ch_r, ch_v),
        }
        print(f"HELDOUT {split}:", held[split])
    with open(os.path.join(OUT, "heldout_frozen.json"), "w", encoding="utf-8") as f:
        json.dump(held, f, indent=2)

    main_seeds = list(range(20))
    main_agents = ["Chance", "NeuroSafeCBO", "RiskNeutral", "CVaR", "Greedy", "Random"]
    main_df = pd.concat(
        [run_agents_seedlist(n, base, main_seeds, 20, f"main20-{n}") for n in main_agents],
        ignore_index=True,
    )
    main_df.to_csv(os.path.join(OUT, "main20_episodes.csv"), index=False)
    summary = (
        main_df.groupby("agent", as_index=False)
        .agg(
            mean_return=("return", "mean"),
            std_return=("return", "std"),
            mean_viol=("viol", "mean"),
            std_viol=("viol", "std"),
            mean_cvar=("cvar", "mean"),
        )
        .sort_values("mean_viol")
    )
    summary.to_csv(os.path.join(OUT, "summary_main20.csv"), index=False)
    print(summary.to_string(index=False))

    ch_r, ch_v = means(main_df, "Chance")
    ns_r, ns_v = means(main_df, "NeuroSafeCBO")
    verdict = {
        "chance": {"return": ch_r, "viol": ch_v},
        "neurosafe": {"return": ns_r, "viol": ns_v},
        **dual_flags(ns_r, ns_v, ch_r, ch_v),
        "n_seeds": 20,
        "n_episodes": 20,
        "n_cells": 4,
    }
    with open(os.path.join(OUT, "dual_verdict_main20.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print("MAIN20:", verdict)

    pack = seed_level_stats(main_df[main_df.agent.isin(["Chance", "NeuroSafeCBO"])])
    pack["by_seed"].to_csv(os.path.join(OUT, "summary_by_seed.csv"), index=False)
    with open(os.path.join(OUT, "seed_stats.json"), "w", encoding="utf-8") as f:
        json.dump(pack["summary"], f, indent=2)
    print("SEED STATS:", pack["summary"])

    abl_seeds = list(range(10))
    abl_specs = [
        ("NeuroSafeCBO", "NeuroSafeCBO", dict(base)),
        ("NS_noPC", "NS_noPC", {**base, "PC_UTIL_MIX": 0.0, "ANCHOR_WITHOUT_PC": True}),
        ("NS_noUtilFloor", "NeuroSafeCBO", {**base, "DOMINANCE_UTIL_EPS": 0.0}),
        ("NS_ChanceOnly", "NeuroSafeCBO", {**base, "DOMINANCE_UTIL_EPS": 1e9, "PC_UTIL_MIX": 0.0}),
        ("NS_uncalibratedHarm", "NeuroSafeCBO", {**base, "DECOUPLE_HARM": False}),
        ("Chance", "Chance", dict(base)),
        ("RiskNeutral", "RiskNeutral", dict(base)),
        ("CVaR", "CVaR", dict(base)),
        ("Greedy", "Greedy", dict(base)),
        ("Random", "Random", dict(base)),
    ]
    abl_rows = []
    abl_frames = []
    ch_r_ref = ch_v_ref = None
    for label, agent_name, cfg in abl_specs:
        df = run_agents_seedlist(agent_name, cfg, abl_seeds, 20, f"abl-{label}")
        df = df.assign(label=label)
        abl_frames.append(df)
        r, v = float(df["return"].mean()), float(df["viol"].mean())
        if label == "Chance":
            ch_r_ref, ch_v_ref = r, v
        abl_rows.append({"label": label, "mean_return": r, "mean_viol": v})
    for row in abl_rows:
        row.update(dual_flags(row["mean_return"], row["mean_viol"], ch_r_ref, ch_v_ref))
    abl_tab = pd.DataFrame(abl_rows)
    pd.concat(abl_frames, ignore_index=True).to_csv(
        os.path.join(OUT, "ablation_episodes.csv"), index=False
    )
    abl_tab.to_csv(os.path.join(OUT, "ablation_table.csv"), index=False)
    with open(os.path.join(OUT, "ablation_table.json"), "w", encoding="utf-8") as f:
        json.dump(abl_rows, f, indent=2)
    print(abl_tab.to_string(index=False))

    report = (
        "# Publishable weak dual package\n\n"
        "## Frozen knobs\n"
        "- ALPHA = NS_ALPHA = 0.15\n"
        "- DOMINANCE_UTIL_EPS = 0.05\n"
        "- SAFE_FLOOR_BONUS = 0.0\n"
        "- PC_UTIL_MIX = 0.35\n"
        "- DECOUPLE_HARM = True\n\n"
        "## Held-out (no retune)\n"
        f"- train seeds {train_seeds}: dual_weak={held['train']['dual_weak']} "
        f"dR={held['train']['dR']:.4f} dV={held['train']['dV']:.6f}\n"
        f"- test seeds {test_seeds}: dual_weak={held['test']['dual_weak']} "
        f"dR={held['test']['dR']:.4f} dV={held['test']['dV']:.6f}\n\n"
        "## Main (20 seeds x 20 eps x 4 cells)\n"
        f"- Chance: return={ch_r:.4f}, viol={ch_v:.6f}\n"
        f"- NeuroSafeCBO: return={ns_r:.4f}, viol={ns_v:.6f}\n"
        f"- dual_weak={verdict['dual_weak']} dual_strict={verdict['dual_strict']}\n\n"
        "## Seed-level\n"
        "See seed_stats.json / summary_by_seed.csv\n\n"
        "## Ablations\n"
        "See ablation_table.csv\n"
    )
    with open(os.path.join(OUT, "PUBLISH_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
