"""Semi-synthetic IHDP policy evaluation for safe-optimal treatment rules."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurosafe_cbo.config import CFG

OUT_DIR = os.path.join(ROOT, "results", "day7")
DATA_DIR = os.path.join(ROOT, "q1_out")


def download(urls, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return np.load(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last = None
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                open(path, "wb").write(r.content)
                return np.load(path)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not download {path}. Last error: {last}")


def fit_outcomes(X, T, Y):
    m0 = HistGradientBoostingRegressor(max_depth=3, max_iter=100).fit(X[T == 0], Y[T == 0])
    m1 = HistGradientBoostingRegressor(max_depth=3, max_iter=100).fit(X[T == 1], Y[T == 1])
    return m0, m1


def harm_from_outcome(y):
    """Map outcomes to [0,1] harm via rank (lower outcome => higher harm)."""
    y = np.asarray(y, dtype=float)
    order = y.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(1.0, 0.0, len(y))  # best outcome -> harm 0
    return ranks


def evaluate_policies(mu0, mu1, tau_harm):
    """Policies map estimates to treat/not-treat; harm is outcome-rank based."""
    tau_hat = mu1 - mu0
    h0 = harm_from_outcome(mu0)
    h1 = harm_from_outcome(mu1)

    policies = {
        "all": np.ones_like(tau_hat),
        "none": np.zeros_like(tau_hat),
        "pos_effect": (tau_hat > 0).astype(float),
        "top_quartile": (tau_hat >= np.quantile(tau_hat, 0.75)).astype(float),
        "max_outcome": (mu1 >= mu0).astype(float),
        # Safe-optimal: improve utility AND keep treatment-harm within budget
        "safe_optimal": ((tau_hat > 0) & (h1 <= tau_harm)).astype(float),
    }

    rows = []
    for name, pi in policies.items():
        yhat = pi * mu1 + (1.0 - pi) * mu0
        harm = pi * h1 + (1.0 - pi) * h0
        rows.append(
            {
                "policy": name,
                "util": float(yhat.mean()),
                "harm": float(harm.mean()),
                "viol": float((harm > tau_harm).mean()),
                "treat_rate": float(pi.mean()),
            }
        )
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = download(
        [
            "https://raw.githubusercontent.com/Osier-Yi/SITE/master/data/ihdp_npci_1-100.train.npz",
            "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
        ],
        os.path.join(DATA_DIR, "train.npz"),
    )
    test = download(
        [
            "https://raw.githubusercontent.com/Osier-Yi/SITE/master/data/ihdp_npci_1-100.test.npz",
            "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
        ],
        os.path.join(DATA_DIR, "test.npz"),
    )

    Xtr, Ttr, Ytr = train["x"], train["t"], train["yf"]
    Xte = test["x"]
    tau = 0.50  # operating point on rank-harm scale for IHDP study
    rows = []
    n_rep = min(20, Xtr.shape[-1])
    for f in range(n_rep):
        m0, m1 = fit_outcomes(Xtr[:, :, f], Ttr[:, f], Ytr[:, f])
        mu0 = m0.predict(Xte[:, :, f])
        mu1 = m1.predict(Xte[:, :, f])
        for r in evaluate_policies(mu0, mu1, tau):
            r["replication"] = f
            r["tau_harm"] = tau
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "results_ihdp.csv"), index=False)
    summary = (
        df.groupby("policy", as_index=False)
        .agg(
            mean_util=("util", "mean"),
            mean_harm=("harm", "mean"),
            mean_viol=("viol", "mean"),
            mean_treat=("treat_rate", "mean"),
        )
        .sort_values("mean_util", ascending=False)
    )
    summary.to_csv(os.path.join(OUT_DIR, "summary_ihdp.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Saved IHDP results to {OUT_DIR}")


if __name__ == "__main__":
    main()
