"""Episode metrics and constrained regret helpers."""

from __future__ import annotations

import numpy as np

from .config import CFG


def estimate_cvar(samples, beta: float = None) -> float:
    beta = CFG["BETA_CVAR"] if beta is None else beta
    s = np.sort(np.asarray(samples, dtype=float))
    k = max(1, int(np.ceil(beta * len(s))))
    return float(np.mean(s[k - 1 :]))


def episode_metrics(U_hist, H_hist, tau: float = None) -> dict:
    tau = CFG["TAU_HARM"] if tau is None else tau
    U = np.asarray(U_hist, dtype=float)
    H = np.asarray(H_hist, dtype=float)
    return {
        "return": float(U.sum()),
        "mean_u": float(U.mean()),
        "viol": float(np.mean(H > tau)),
        "mean_h": float(H.mean()),
        "cvar": estimate_cvar(H),
    }


def constrained_regret(mu_u: np.ndarray, p_viol: np.ndarray, chosen_idx: int, alpha: float) -> float:
    """Instantaneous regret vs best alpha-feasible action (oracle).

    If no action is feasible, compare against the lowest-violation action.
    """
    feasible = p_viol <= alpha
    if np.any(feasible):
        best = float(np.max(mu_u[feasible]))
    else:
        best = float(mu_u[int(np.argmin(p_viol))])
    return float(best - mu_u[chosen_idx])
