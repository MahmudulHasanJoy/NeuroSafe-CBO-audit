"""Empirically check the margin condition (M) from paper/theory_steps_17_18.md
item C:

    P( |p(a, z) - alpha| <= t )  <=  C t^beta      for all t > 0, all a.

Flagged there as "a cheap experiment we have not run." This script estimates
beta by regressing log(empirical CDF of |p - alpha| near 0) on log(t) across
three environments: the decoupled synthetic SCM (true harm function), and the
two real domains' *fitted* harm models (env_lending_bandit.py,
env_mimic_bandit.py) as the best available surrogate for the true p(a, x).

A finite, well-defined beta with no atom at t=0 (F(0+) ~ 0) supports using
(M) as stated in the theory draft. This does not "prove" the condition (that
would require the true data-generating p(a,x) for the real domains, which is
unknown/unidentified -- see the causal-identification caveat in both env
files); it checks the condition against the same fitted surfaces the paper's
theory already treats as ground truth for the harm channel.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosafe_cbo.env import NonstationarySCMEnv
from neurosafe_cbo.env_lending_bandit import _load_and_fit as _load_lending
from neurosafe_cbo.env_mimic_bandit import _load_and_fit as _load_mimic

ACTIONS = np.linspace(0.0, 1.0, 11)


def margin_beta(p_values: np.ndarray, alpha: float, t_grid=None):
    t_grid = t_grid if t_grid is not None else np.geomspace(0.002, 0.2, 12)
    d = np.abs(p_values - alpha)
    F = np.array([np.mean(d <= t) for t in t_grid])
    F0 = float(np.mean(d <= 1e-4))
    mask = F > 0
    if mask.sum() < 3:
        return {"beta": None, "r2": None, "F_at_0": F0, "n": len(p_values)}
    x = np.log(t_grid[mask])
    y = np.log(F[mask])
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    beta, logC = coef
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "beta": float(beta),
        "C": float(np.exp(logC)),
        "r2": float(r2),
        "F_at_0": F0,
        "n": len(p_values),
    }


def check_scm(alpha=0.15, n_context=4000, seed=0):
    rng = np.random.RandomState(seed)
    env = NonstationarySCMEnv(shift_at=100, shift_mag=0.35, obs_noise=0.15, seed=seed, harm_mode="decoupled")
    p_values = []
    for _ in range(n_context):
        env.C = float(rng.normal())
        env.R = float(rng.normal() * 0.5)
        env.s = int(rng.rand() < 0.5)
        for a in ACTIONS:
            gain = env.DEC["risk_gain_post"] if env.s == 1 else env.DEC["risk_gain_pre"]
            side = gain * a + env.DEC["risk_rho"] * env.R + env.DEC["risk_ctx"] * env.C
            p = 1.0 / (1.0 + np.exp(-(side - env.DEC["risk_center"]) / env.DEC["risk_scale"]))
            p_values.append(p)
    return margin_beta(np.array(p_values), alpha)


def check_real(loader, feature_cols, alpha, seed=0, n_x=2000):
    loaded = loader(seed=seed)
    model = loaded["model"]
    X = loaded["covariates"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(n_x, X.shape[0]), replace=False)
    p_values = []
    for i in idx:
        for a in ACTIONS:
            row = np.concatenate([X[i], [a]]).reshape(1, -1)
            p = model.predict_proba(row)[0, 1]
            p_values.append(p)
    return margin_beta(np.array(p_values), alpha)


def main():
    results = {
        "decoupled_scm": {"alpha": 0.15, **check_scm(alpha=0.15)},
        "lending_real_fitted": {"alpha": 0.25, **check_real(_load_lending, None, alpha=0.25)},
        "mimic_real_fitted": {"alpha": 0.03, **check_real(_load_mimic, None, alpha=0.03)},
    }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "margin_check")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "margin_beta.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    for name, r in results.items():
        print(f"{name}: alpha={r['alpha']}  beta~{r['beta']:.3f}  R2={r['r2']:.3f}"
              f"  F(t<1e-4)={r['F_at_0']:.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
