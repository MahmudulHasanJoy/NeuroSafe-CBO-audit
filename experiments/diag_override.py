"""Diagnostic: how often is the anchored override pool non-empty, and does it change the action?

This answers the reviewer question the paper never answers:
  "If NeuroSafe-CBO ties Chance everywhere, what is the anchored override actually doing?"

Usage:
    python experiments/diag_override.py --harm-mode tradeoff --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.main_track_common import actions_for_k, frozen_base
from neurosafe_cbo.agent import make_agent
from neurosafe_cbo.env import NonstationarySCMEnv


def run_diag(cfg: dict, seeds, n_eps: int, harm_mode: str) -> pd.DataFrame:
    actions = actions_for_k(int(cfg["N_ACTIONS"]))
    rows = []
    for sm in cfg["SHIFT_MAGS"]:
        for on in cfg["OBS_NOISES"]:
            for seed in seeds:
                agg = None
                for ep in range(n_eps):
                    env = NonstationarySCMEnv(
                        cfg["SHIFT_AT"], sm, on,
                        seed=1000 * int(seed) + ep, harm_mode=harm_mode,
                    )
                    agent = make_agent("NeuroSafeCBO", seed=int(seed), cfg=cfg, actions=actions)
                    obs = env.reset()
                    for _t in range(int(cfg["HORIZON"])):
                        a, z = agent.act(obs)
                        obs, U, H, _, _ = env.step(a)
                        agent.observe(a, z, U, H)
                    d = agent.diag
                    agg = d if agg is None else {k: agg[k] + d[k] for k in d}
                n = max(agg["decisions"], 1)
                rows.append({
                    "shift_mag": sm, "obs_noise": on, "seed": int(seed),
                    "decisions": agg["decisions"],
                    "mean_feasible_size": agg["feasible_size_sum"] / n,
                    "mean_pool_size": agg["pool_size_sum"] / n,
                    "pct_feasible_gt1": 100.0 * agg["feasible_gt1"] / n,
                    "pct_pool_nonempty": 100.0 * agg["pool_nonempty"] / n,
                    "pct_override_fired": 100.0 * agg["override_fired"] / n,
                    "mean_util_gap_when_fired": (
                        agg["util_gap_sum"] / agg["override_fired"] if agg["override_fired"] else 0.0
                    ),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harm-mode", default="tradeoff")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--util-eps", type=float, default=None)
    ap.add_argument("--ctx-mode", default="pc")
    ap.add_argument("--ctx-gain", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = frozen_base()
    cfg["CTX_MODE"] = args.ctx_mode
    cfg["CTX_GAIN"] = args.ctx_gain
    cfg["HORIZON"] = 40
    if args.alpha is not None:
        cfg["ALPHA"] = cfg["NS_ALPHA"] = args.alpha
    if args.util_eps is not None:
        cfg["DOMINANCE_UTIL_EPS"] = args.util_eps

    df = run_diag(cfg, range(args.seeds), args.episodes, args.harm_mode)
    summary = {
        "harm_mode": args.harm_mode,
        "alpha": cfg["ALPHA"],
        "util_eps": cfg["DOMINANCE_UTIL_EPS"],
        "mean_feasible_size": float(df["mean_feasible_size"].mean()),
        "mean_pool_size": float(df["mean_pool_size"].mean()),
        "pct_feasible_gt1": float(df["pct_feasible_gt1"].mean()),
        "pct_pool_nonempty": float(df["pct_pool_nonempty"].mean()),
        "pct_override_fired": float(df["pct_override_fired"].mean()),
        "mean_util_gap_when_fired": float(df["mean_util_gap_when_fired"].mean()),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(args.out.replace(".csv", "_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
