"""German Credit sequential safe bandit environment (real adverse-event domain).

Rebuilt independently: the lending/German-Credit result referenced in
SESSION_REPORT.md has no surviving source in this snapshot (not in the
diagnostics tar.gz; the git bundle is a thin bundle missing its prerequisite
commit and cannot be unbundled standalone). This is a fresh implementation
against the public UCI German Credit dataset (`statlog/german`, n=1000,
1994), mirroring env_mimic_bandit.py's pattern, so results here are
independently verifiable rather than inherited from an unreproducible run.

--- Action / utility / harm ---

`a in [0,1]` is a loan-sizing decision (how large a loan to extend, using
the applicant's actual historical `amount` as the ordinal signal a response
model is fit against). Utility `U(a) = UTIL_SCALE * a + noise`: a larger
loan is more profitable to the lender if repaid. Harm `H(a, x)` is a
gradient-boosted classifier's `P(default | risk_score, duration, a)`,
fit on the real observed default outcome (`credit_risk`). Covariates:
`risk_score` (data-driven ordinal blend of checking-account status and
credit history, each ranked by their own empirical default rate) and
`duration_norm` (loan term).

--- Important limitation (read before citing any result from this env) ---

Same identification caveat as env_mimic_bandit.py, if anything sharper here:
`amount` in the raw data is the loan the bank *actually chose to extend*
after underwriting -- it already reflects the bank's own risk assessment
(confounding by indication / selection on the outcome's own predictor).
Querying the fitted model at a counterfactual `a` for a given applicant
does not recover a causal loan-sizing effect. Treat this as a methods
stress-test on a real financial adverse-event signal, not a validated
underwriting policy.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "german_credit")
RAW_PATH = os.path.join(DATA_DIR, "raw", "german_credit.csv")
PREPPED_PATH = os.path.join(DATA_DIR, "prepped.csv.gz")

UTIL_SCALE = 1.0
DURATION_SCALE = 72.0
AMOUNT_Q_CLIP = 0.99  # clip extreme loan amounts before min-max scaling

_CACHE: dict = {}


def _ordinal_risk_map(df: pd.DataFrame, col: str) -> dict:
    """Rank a categorical column's levels by its own empirical default rate,
    low risk -> 0, high risk -> 1. Same data-driven-ordinal approach used
    for admission_type acuity in env_mimic_bandit.py."""
    rates = df.groupby(col)["default"].mean().sort_values()
    levels = list(rates.index)
    return {lvl: i / (len(levels) - 1) for i, lvl in enumerate(levels)}


def _prep_table(raw_path: str = RAW_PATH) -> pd.DataFrame:
    if os.path.exists(PREPPED_PATH):
        return pd.read_csv(PREPPED_PATH)
    df = pd.read_csv(raw_path)
    df["default"] = (df["credit_risk"] == 0).astype(int)
    status_map = _ordinal_risk_map(df, "status")
    history_map = _ordinal_risk_map(df, "credit_history")
    df["risk_status"] = df["status"].map(status_map)
    df["risk_history"] = df["credit_history"].map(history_map)
    df["risk_score"] = 0.5 * (df["risk_status"] + df["risk_history"])
    df["duration_norm"] = np.clip(df["duration"] / DURATION_SCALE, 0, 1)
    hi = df["amount"].quantile(AMOUNT_Q_CLIP)
    df["amount_action"] = np.clip(df["amount"], 0, hi) / hi
    keep = ["risk_score", "duration_norm", "amount_action", "default"]
    df = df[keep].dropna()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PREPPED_PATH, index=False)
    return df


def _fit_harm_model(df: pd.DataFrame, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = df[["risk_score", "duration_norm", "amount_action"]].to_numpy(dtype=float)
    y = df["default"].to_numpy(dtype=int)
    # v3: monotone constraints. Unconstrained, the tree ensemble produced a
    # jagged dose-response (mean P(default) 0.25 -> 0.20 -> 0.41 -> 0.22 as the
    # loan grows), i.e. "a larger loan lowers default risk" on half the grid --
    # an artifact of 1000 sparse points, not a property of lending. We impose the
    # domain-knowledge constraint that, holding the other covariates fixed,
    # default risk is non-decreasing in the risk score and in the loan amount.
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=150, random_state=seed, monotonic_cst=[1, 0, 1]
    )
    model.fit(X, y)
    return model


def _load_and_fit(raw_path: str = RAW_PATH, seed: int = 0):
    key = (raw_path, seed)
    if key in _CACHE:
        return _CACHE[key]
    df = _prep_table(raw_path)
    model = _fit_harm_model(df, seed=seed)
    covariates = df[["risk_score", "duration_norm"]].to_numpy(dtype=float)
    result = {"df": df, "model": model, "covariates": covariates}
    _CACHE[key] = result
    return result


class LendingSafeBanditEnv:
    """Sequential bandit over real German Credit applicants with a fitted
    default-risk model. `a` is the loan-sizing action in [0, 1]. `H` is the
    fitted P(default | risk_score, duration, a). `U = UTIL_SCALE * a +
    noise` rewards larger loans, creating a safety/utility tension: the
    utility-best action (a=1, largest loan) is not the safest for
    high-risk applicants. See module docstring for the identification
    caveat.
    """

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 40,
        tau_harm: float = 0.25,
        obs_noise: float = 0.05,
        raw_path: str = RAW_PATH,
        model_seed: int = 0,
    ):
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.tau_harm = float(tau_harm)
        self.obs_noise = float(obs_noise)
        self.rng = np.random.RandomState(seed)
        loaded = _load_and_fit(raw_path, seed=model_seed)
        self.model = loaded["model"]
        self.X = loaded["covariates"]
        self.n_patients = self.X.shape[0]
        self.t = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.last_y = 0.0
        return self._next_obs()

    def _next_obs(self) -> np.ndarray:
        idx = int(self.rng.randint(0, self.n_patients))
        x = self.X[idx]
        noisy = x + self.rng.normal(0, self.obs_noise, size=x.shape)
        c = float(np.mean(noisy))
        x1 = float(noisy[0])
        x2 = float(noisy[1])
        y = float(self.last_y)
        self._cur_idx = idx
        return np.array([c, x1, x2, y], dtype=float)

    def step(self, a: float):
        idx = self._cur_idx
        a = float(np.clip(a, 0.0, 1.0))
        risk_score, duration_norm = self.X[idx]
        query = np.array([[risk_score, duration_norm, a]], dtype=float)
        if a < 0.05:
            # v3: a = 0 is "decline the application" -- no loan, no profit, no
            # default exposure. Previously the smallest action was still a loan,
            # so for the ~40% of applicants with no acceptable loan size every
            # agent was forced into a violation (realised rates ~0.36-0.44
            # against alpha = 0.15). Declining is the natural safe fallback.
            p_default = 0.0
            H = 0.0
            U = float(self.rng.normal(0, 0.05))
        else:
            p_default = float(self.model.predict_proba(query)[0, 1])
            H = float(np.clip(p_default + self.rng.normal(0, 0.01), 0.0, 1.0))
            U = float(UTIL_SCALE * a + self.rng.normal(0, 0.05))
        self.last_y = U
        self.t += 1
        done = self.t >= self.horizon
        obs = self._next_obs() if not done else np.array([0.0, 0.0, 0.0, self.last_y], dtype=float)
        return obs, U, H, done, {"applicant": idx, "p_default": p_default}
