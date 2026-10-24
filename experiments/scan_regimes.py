"""Scan environment regimes to find where (a) the harm constraint actually binds and
(b) safe agents can differentiate.

The published SCM-50 setup has safe agents at ~0.2% violation -- the constraint is
essentially never active, so there is nothing for a 'safe' method to be better at.
This scans TAU_HARM x SHIFT_MAG to locate regimes where the constraint is live.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.main_track_common import actions_for_k, frozen_base
from neurosafe_cbo.agent import make_agent
from neurosafe_cbo.env import NonstationarySCMEnv
from neurosafe_cbo.metrics import episode_metrics


def run_cell(cfg, agents, seeds, n_eps, sm, on, harm_mode):
    actions = actions_for_k(int(cfg["N_ACTIONS"]))
    rows = []
    for seed in seeds:
        for ep in range(n_eps):
            for name in agents:
                env = NonstationarySCMEnv(
                    cfg["SHIFT_AT"], sm, on, seed=1000 * int(seed) + ep, harm_mode=harm_mode
                )
                agent = make_agent(name, seed=int(seed), cfg=cfg, actions=actions)
                obs = env.reset()
                U_hist, H_hist = [], []
                for _t in range(int(cfg["HORIZON"])):
                    a, z = agent.act(obs)
                    obs, U, H, _, _ = env.step(a)
                    agent.observe(a, z, U, H)
                    U_hist.append(U)
                    H_hist.append(H)
                m = episode_metrics(U_hist, H_hist, cfg["TAU_HARM"])
                fired = getattr(agent, "diag", {}).get("override_fired", 0)
                dec = max(getattr(agent, "diag", {}).get("decisions", 0), 1)
                rows.append({
                    "agent": name, "seed": int(seed), "episode": ep,
                    "tau": cfg["TAU_HARM"], "shift_mag": sm, "obs_noise": on,
                    "pct_override": 100.0 * fired / dec,
                    **m,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35, 0.50])
    ap.add_argument("--shifts", type=float, nargs="+", default=[0.35, 1.0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--harm-mode", default="tradeoff")
    ap.add_argument("--ctx-mode", default="pc")
    ap.add_argument("--ctx-gain", type=float, default=0.35)
    ap.add_argument("--dual-update", type=int, default=1)
    ap.add_argument("--out", default="results/diag/regime_scan.csv")
    args = ap.parse_args()

    agents = ["Chance", "NeuroSafeCBO"]
    all_rows = []
    for tau in args.taus:
        for sm in args.shifts:
            cfg = frozen_base()
            cfg["CTX_MODE"] = args.ctx_mode
            cfg["CTX_GAIN"] = args.ctx_gain
            cfg["CTX_DUAL_UPDATE"] = bool(args.dual_update)
            cfg["HORIZON"] = 40
            cfg["TAU_HARM"] = tau
            rows = run_cell(cfg, agents, range(args.seeds), args.episodes, sm, 0.15, args.harm_mode)
            all_rows.extend(rows)
            df = pd.DataFrame(rows)
            g = df.groupby("agent").agg(ret=("return", "mean"), viol=("viol", "mean"),
                                        ovr=("pct_override", "mean"))
            ch, ns = g.loc["Chance"], g.loc["NeuroSafeCBO"]
            print(f"tau={tau:<5} shift={sm:<5} | Chance R={ch.ret:8.3f} V={ch.viol:.4f} | "
                  f"NS R={ns.ret:8.3f} V={ns.viol:.4f} | dR={ns.ret-ch.ret:+.4f} "
                  f"dV={ns.viol-ch.viol:+.5f} | override={ns.ovr:.1f}%", flush=True)

    out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
