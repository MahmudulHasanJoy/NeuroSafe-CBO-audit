"""Seeded evaluation on the real MIMIC-IV adverse-event domain.

Same statistical protocol as run_ctx_eval.py (paired per-seed deltas,
bootstrap CI, Wilcoxon), swapped onto neurosafe_cbo.env_mimic_bandit instead
of the synthetic SCM. See env_mimic_bandit.py's module docstring for the
causal-identification caveat before treating any result here as a clinical
claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.main_track_common import actions_for_k, bootstrap_ci, frozen_base
from neurosafe_cbo.agent import make_agent
from neurosafe_cbo.env_mimic_bandit import MimicSafeBanditEnv
from neurosafe_cbo.metrics import episode_metrics

CTX_CFG = dict(
    CTX_MODE="kalman",
    CTX_DUAL_UPDATE=False,
    RFF_SIGMA_U=0.30,
    RFF_SIGMA_Z=1.0,
    CTX_WEIGHT=0.7,
    CTX_HARM_BUCKETS=True,
    CTX_EDGES=(0.45, 0.52, 0.61),  # domain-calibrated: mimic z ~ [0.16, 1.08], tercile edges
)


def run(agents, seeds, eps, H, tau, extra=None):
    acts = actions_for_k(11)
    rows = []
    for name in agents:
        cfg = frozen_base()
        cfg.update({"TAU_HARM": tau, "HORIZON": H})
        if extra and name in extra:
            cfg.update(extra[name])
        for seed in seeds:
            for ep in range(eps):
                env = MimicSafeBanditEnv(
                    seed=1000 * int(seed) + ep, horizon=H, tau_harm=tau
                )
                ag = make_agent(name, seed=int(seed), cfg=cfg, actions=acts)
                obs = env.reset()
                U, Hh = [], []
                for _t in range(H):
                    a, z = ag.act(obs)
                    obs, u, h, _, _ = env.step(a)
                    ag.observe(a, z, u, h)
                    U.append(u)
                    Hh.append(h)
                m = episode_metrics(U, Hh, tau)
                rows.append(
                    {
                        "agent": name,
                        "seed": int(seed),
                        "episode": ep,
                        "return_per_step": m["return"] / H,
                        "viol": m["viol"],
                    }
                )
    return pd.DataFrame(rows)


def paired(df, a_new, a_base):
    g = df.groupby(["agent", "seed"], as_index=False).agg(
        ret=("return_per_step", "mean"), viol=("viol", "mean")
    )
    b = g[g.agent == a_base].set_index("seed").sort_index()
    n = g[g.agent == a_new].set_index("seed").sort_index()
    common = b.index.intersection(n.index)
    dR = (n.loc[common, "ret"] - b.loc[common, "ret"]).to_numpy()
    dV = (n.loc[common, "viol"] - b.loc[common, "viol"]).to_numpy()
    mR, loR, hiR = bootstrap_ci(dR)
    mV, loV, hiV = bootstrap_ci(dV)
    try:
        _, p = stats.wilcoxon(dR, alternative="greater")
    except ValueError:
        p = float("nan")
    return {
        "n_seeds": int(len(common)),
        "base_return": float(b.loc[common, "ret"].mean()),
        "new_return": float(n.loc[common, "ret"].mean()),
        "mean_dR": mR,
        "dR_ci95": [loR, hiR],
        "base_viol": float(b.loc[common, "viol"].mean()),
        "new_viol": float(n.loc[common, "viol"].mean()),
        "mean_dV": mV,
        "dV_ci95": [loV, hiV],
        "wilcoxon_dR_p": float(p),
        "seeds_improved": int(np.sum(dR > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--out", default="results/mimic_eval")
    args = ap.parse_args()

    agents = ["Chance", "NeuroSafeCBO", "NeuroSafeCtx"]
    extra = {"NeuroSafeCtx": CTX_CFG}
    df = run(agents, range(args.seeds), args.episodes, args.horizon, args.tau, extra)

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "episodes.csv"), index=False)

    res = {
        "config": {
            "seeds": args.seeds,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "tau": args.tau,
            "domain": "mimic_iv_real",
        },
        "NeuroSafeCtx_vs_Chance": paired(df, "NeuroSafeCtx", "Chance"),
        "NeuroSafeCBO_vs_Chance": paired(df, "NeuroSafeCBO", "Chance"),
    }
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    for k in ("NeuroSafeCBO_vs_Chance", "NeuroSafeCtx_vs_Chance"):
        r = res[k]
        print(f"\n=== {k} (n={r['n_seeds']} seeds) ===")
        print(
            f"  return/step : {r['base_return']:.4f} -> {r['new_return']:.4f}"
            f"   dR={r['mean_dR']:+.4f}"
        )
        print(
            f"  dR 95% CI   : [{r['dR_ci95'][0]:+.4f}, {r['dR_ci95'][1]:+.4f}]"
            f"   Wilcoxon p={r['wilcoxon_dR_p']:.2e}"
        )
        print(
            f"  violation   : {r['base_viol']:.5f} -> {r['new_viol']:.5f}"
            f"   dV={r['mean_dV']:+.5f}  CI [{r['dV_ci95'][0]:+.5f}, {r['dV_ci95'][1]:+.5f}]"
        )
        print(f"  seeds improved: {r['seeds_improved']}/{r['n_seeds']}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
