"""Shared utilities for A* main-track experiments."""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from neurosafe_cbo.agent import make_agent
from neurosafe_cbo.config import CFG
from neurosafe_cbo.env import NonstationarySCMEnv
from neurosafe_cbo.metrics import constrained_regret, episode_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT = os.path.join(ROOT, "results", "main_track")

MAIN_AGENTS = ["Chance", "NeuroSafeCBO", "RiskNeutral", "CVaR", "Greedy", "Random"]

PROFILES = {
    "smoke": {
        "SEEDS": 2,
        "EPISODES": 2,
        "HORIZON": 20,
        "ORACLE_MC": 8,
        "REGRET_EVERY": 5,
        "SHIFT_MAGS": [0.35],
        "OBS_NOISES": [0.15],
    },
    "week1": {
        "SEEDS": 10,
        "EPISODES": 10,
        "HORIZON": 40,
        "ORACLE_MC": 12,
        "REGRET_EVERY": 5,
    },
    "scm50": {
        "SEEDS": 50,
        "EPISODES": 40,
        "HORIZON": 40,
        "ORACLE_MC": 12,
        "REGRET_EVERY": 10,
    },
    "ihdp": {
        "SEEDS": 10,
        "EPISODES": 20,
        "HORIZON": 40,
        "ORACLE_MC": 0,
        "REGRET_EVERY": 0,
    },
    "kstudy": {
        "SEEDS": 10,
        "EPISODES": 10,
        "HORIZON": 40,
        "ORACLE_MC": 12,
        "REGRET_EVERY": 10,
        "K_VALUES": [5, 11, 21, 41],
    },
}


def frozen_base() -> dict:
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
            "COMPUTE_REGRET": True,
            "HARM_MODE": "tradeoff",
        }
    )
    return base


def apply_profile(cfg: dict, profile: str) -> dict:
    if profile not in PROFILES:
        raise KeyError(f"Unknown profile {profile}; choose from {list(PROFILES)}")
    cfg = dict(cfg)
    cfg.update(PROFILES[profile])
    return cfg


def actions_for_k(k: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(k))


def run_episode_tracked(
    env,
    agent,
    actions: np.ndarray,
    cfg: dict,
    *,
    compute_regret: bool = True,
) -> tuple[dict, list[dict]]:
    """Run one episode; return summary metrics + per-checkpoint regret rows."""
    obs = env.reset()
    U_hist, H_hist = [], []
    step_rows: list[dict] = []
    regret_every = int(cfg.get("REGRET_EVERY", 10))
    oracle_mc = int(cfg.get("ORACLE_MC", 12))

    for t in range(int(cfg["HORIZON"])):
        a, z = agent.act(obs)
        if (
            compute_regret
            and regret_every > 0
            and hasattr(env, "oracle_action_stats")
            and (t % regret_every == 0)
        ):
            mu_u, p_viol = env.oracle_action_stats(actions, oracle_mc, cfg["TAU_HARM"])
            chosen_idx = int(np.argmin(np.abs(actions - a)))
            r_inst = constrained_regret(mu_u, p_viol, chosen_idx, cfg["ALPHA"])
            step_rows.append(
                {
                    "step": t,
                    "instant_regret": r_inst,
                    "cum_regret": float(
                        step_rows[-1]["cum_regret"] + r_inst if step_rows else r_inst
                    ),
                }
            )
        obs, U, H, _, _ = env.step(a)
        agent.observe(a, z, U, H)
        U_hist.append(U)
        H_hist.append(H)

    m = episode_metrics(U_hist, H_hist, cfg["TAU_HARM"])
    if step_rows:
        m["avg_constrained_regret"] = float(np.mean([r["instant_regret"] for r in step_rows]))
        m["final_cum_regret"] = float(step_rows[-1]["cum_regret"])
    else:
        m["avg_constrained_regret"] = float("nan")
        m["final_cum_regret"] = float("nan")
    return m, step_rows


def run_scm_grid(
    cfg: dict,
    agents: Iterable[str],
    seeds: Iterable[int],
    n_eps: int,
    desc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = actions_for_k(int(cfg["N_ACTIONS"]))
    ep_rows, step_rows = [], []
    total = len(cfg["SHIFT_MAGS"]) * len(cfg["OBS_NOISES"]) * len(list(seeds)) * n_eps * len(list(agents))
    pbar = tqdm(total=total, desc=desc)

    for sm in cfg["SHIFT_MAGS"]:
        for on in cfg["OBS_NOISES"]:
            for seed in seeds:
                for ep in range(n_eps):
                    for name in agents:
                        env = NonstationarySCMEnv(
                            cfg["SHIFT_AT"],
                            sm,
                            on,
                            seed=1000 * int(seed) + ep,
                            harm_mode=cfg.get("HARM_MODE", "tradeoff"),
                        )
                        agent = make_agent(name, seed=int(seed), cfg=cfg, actions=actions)
                        use_regret = bool(cfg.get("COMPUTE_REGRET", True)) and name in MAIN_AGENTS
                        m, steps = run_episode_tracked(
                            env, agent, actions, cfg, compute_regret=use_regret
                        )
                        ep_rows.append(
                            {
                                "agent": name,
                                "shift_mag": sm,
                                "obs_noise": on,
                                "seed": int(seed),
                                "episode": ep,
                                "n_actions": len(actions),
                                **m,
                            }
                        )
                        for s in steps:
                            step_rows.append(
                                {
                                    "agent": name,
                                    "shift_mag": sm,
                                    "obs_noise": on,
                                    "seed": int(seed),
                                    "episode": ep,
                                    **s,
                                }
                            )
                        pbar.update(1)
    pbar.close()
    return pd.DataFrame(ep_rows), pd.DataFrame(step_rows)


def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05):
    rng = np.random.RandomState(0)
    x = np.asarray(x, dtype=float)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(x.mean()), float(lo), float(hi)


def seed_stats_vs_chance(df: pd.DataFrame) -> dict:
    g = df.groupby(["agent", "seed"], as_index=False).agg(
        ret=("return", "mean"),
        viol=("viol", "mean"),
        creg=("avg_constrained_regret", "mean"),
    )
    ch = g[g.agent == "Chance"].set_index("seed").sort_index()
    ns = g[g.agent == "NeuroSafeCBO"].set_index("seed").sort_index()
    common = ch.index.intersection(ns.index)
    dR = (ns.loc[common, "ret"] - ch.loc[common, "ret"]).to_numpy()
    dV = (ns.loc[common, "viol"] - ch.loc[common, "viol"]).to_numpy()
    dC = (ns.loc[common, "creg"] - ch.loc[common, "creg"]).to_numpy()
    mean_dR, lo_dR, hi_dR = bootstrap_ci(dR)
    try:
        _, w_p = stats.wilcoxon(dR, zero_method="wilcox", alternative="greater")
    except ValueError:
        w_p = float("nan")
    return {
        "n_seeds": int(len(common)),
        "mean_dR": mean_dR,
        "dR_ci95": [lo_dR, hi_dR],
        "mean_dV": float(np.mean(dV)),
        "mean_dCreg": float(np.mean(dC)),
        "wilcoxon_dR_p": float(w_p),
        "chance_return": float(ch.loc[common, "ret"].mean()),
        "ns_return": float(ns.loc[common, "ret"].mean()),
        "chance_viol": float(ch.loc[common, "viol"].mean()),
        "ns_viol": float(ns.loc[common, "viol"].mean()),
        "chance_creg": float(ch.loc[common, "creg"].mean()),
        "ns_creg": float(ns.loc[common, "creg"].mean()),
    }


def summarize_agents(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("agent", as_index=False)
        .agg(
            mean_return=("return", "mean"),
            se_return=("return", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            mean_viol=("viol", "mean"),
            se_viol=("viol", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            mean_creg=("avg_constrained_regret", "mean"),
            se_creg=(
                "avg_constrained_regret",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
            ),
        )
        .sort_values("mean_viol")
    )


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
