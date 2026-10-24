"""Numerical check of Theorem 1 (filtered-context safety-regret floor).

Instance: latent c_t ~ N(0,1) i.i.d., observed o_t = c_t + N(0, s^2).
Actions {0, 1}. a=1 has utility Delta=1 but is harmful iff c_t > 0; a=0 is
always safe with utility 0. Theorem 1: for ANY policy,
    (R_n / Delta + 2 V_n) / n  >=  arccos(rho) / pi,   rho = corr(c, o),
with equality for the Bayes rule a_t = 1{E[c_t | o_t] <= 0}.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurosafe_cbo.v3 import NeuroSafeCtxV3  # noqa: E402


def run_policy(policy, s, n, seed):
    rng = np.random.RandomState(seed)
    mist = R = V = 0.0
    agent = None
    if policy == "ours":
        agent = NeuroSafeCtxV3(np.array([0.0, 1.0]), seed, tau=0.5, alpha=0.15,
                               beta_h=1.0, beta_u=0.5, basis="quad5", context="raw")
    for _ in range(n):
        c = rng.normal()
        o = c + rng.normal(0, s)
        if policy == "bayes":
            a = 1.0 if o <= 0 else 0.0
        elif policy == "always_safe":
            a = 0.0
        else:
            a, z = agent.act(np.array([o, 0, 0, 0]))
        H = 1.0 if (a == 1.0 and c > 0) else 0.0
        U = a + rng.normal(0, 0.1)
        if agent is not None:
            agent.observe(a, z, U, H)
        a_star = 1.0 if c <= 0 else 0.0
        mist += a != a_star
        R += a_star - a
        V += H
    return dict(M=mist / n, R=R / n, V=V / n, combo=(R + 2 * V) / n)


if __name__ == "__main__":
    out = []
    for s in [0.4, 0.72, 1.0, 1.73]:
        rho = 1.0 / np.sqrt(1.0 + s * s)
        floor = float(np.arccos(rho) / np.pi)
        row = dict(noise_sd=s, rho=float(rho), floor=floor)
        for pol in ["bayes", "ours", "always_safe"]:
            rs = [run_policy(pol, s, 2000 if pol != "ours" else 600, k) for k in range(8)]
            row[pol] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        out.append(row)
        print(f"rho={rho:.3f} floor={floor:.4f} | bayes {row['bayes']['combo']:.4f} "
              f"| ours {row['ours']['combo']:.4f} (R={row['ours']['R']:.3f}, V={row['ours']['V']:.3f})"
              f" | always-safe {row['always_safe']['combo']:.4f}")
    os.makedirs("results/v3", exist_ok=True)
    json.dump(out, open("results/v3/floor_check.json", "w"), indent=2)
