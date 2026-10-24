"""MIMIC-IV sequential safe bandit environment (real adverse-event domain).

Mirrors env_ihdp_bandit.py's pattern: fit a response surface on real data,
then let the agent query it at a chosen action to get a per-step outcome.

Data: MIMIC-IV `hosp/admissions.csv.gz`, `hosp/patients.csv.gz`,
`hosp/diagnoses_icd.csv.gz` (see data/mimic_iv/raw/). Requires PhysioNet
credentialing + the MIMIC-IV Data Use Agreement; the raw files are not
redistributed with this repo.

--- What the action/utility/harm mean here ---

There is no naturally repeated "treatment dial" in MIMIC-IV the way loan
amount is one in the lending domain. We use `admission_type` as an ordinal
proxy for triage/admission intensity (0 = ambulatory/observation, 1 =
urgent/emergency), and let the agent's action `a in [0,1]` stand in for
"how aggressively to admit/observe this patient." Harm is real:
`hospital_expire_flag` (died during this admission), regressed on
(age, comorbidity count, intensity) with a gradient-boosted classifier.
Utility is a resource-cost proxy, `U(a) = UTIL_SCALE * (1 - a)`: lower
intensity is cheaper/higher-utility, higher intensity is safer.

--- Important limitation (read before citing any result from this env) ---

The fitted P(death | age, n_diag, a) is OBSERVATIONAL, not causal. In the
raw data, higher-acuity admission types have HIGHER death rates (0.01% at
AMBULATORY OBSERVATION vs 4.5% at URGENT) -- this is confounding by
indication (sicker patients get triaged more aggressively), not evidence
that aggressive care is harmful. Querying the fitted model at a
counterfactual `a` for a given patient does not recover the true causal
effect of that triage decision. This is the same identification caveat
already flagged for the lending domain's dose-response in SESSION_REPORT.md
Section 8.3 -- it applies here with more force, since MIMIC has no
plausible near-random arm (lending at least has some rate variation not
fully explained by risk). Treat results from this environment as a
methods stress-test on a real, high-stakes adverse-event signal, not as a
validated clinical claim.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "mimic_iv")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PREPPED_PATH = os.path.join(DATA_DIR, "prepped.csv.gz")

# Ordinal acuity scale, low -> high, fixed by *observed mortality rank* in the
# admissions table (see docstring: this is a proxy, not a resource-intensity
# measurement). Index / (len - 1) gives the [0, 1] intensity value.
ACUITY_ORDER = [
    "AMBULATORY OBSERVATION",
    "EU OBSERVATION",
    "DIRECT OBSERVATION",
    "SURGICAL SAME DAY ADMISSION",
    "ELECTIVE",
    "OBSERVATION ADMIT",
    "DIRECT EMER.",
    "EW EMER.",
    "URGENT",
]
ACUITY_LEVEL = {name: i / (len(ACUITY_ORDER) - 1) for i, name in enumerate(ACUITY_ORDER)}

UTIL_SCALE = 5.0
AGE_SCALE = 100.0
NDIAG_SCALE = 20.0  # clip/scale comorbidity count into a roughly [0, 1+] range

_CACHE: dict = {}


def _prep_table(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    if os.path.exists(PREPPED_PATH):
        return pd.read_csv(PREPPED_PATH)
    adm = pd.read_csv(
        os.path.join(raw_dir, "admissions.csv.gz"),
        usecols=["subject_id", "hadm_id", "admission_type", "hospital_expire_flag"],
    )
    pts = pd.read_csv(
        os.path.join(raw_dir, "patients.csv.gz"), usecols=["subject_id", "anchor_age"]
    )
    diag = pd.read_csv(os.path.join(raw_dir, "diagnoses_icd.csv.gz"), usecols=["hadm_id"])
    ndiag = diag.groupby("hadm_id").size().rename("n_diag")
    df = adm.merge(pts, on="subject_id", how="left").merge(ndiag, on="hadm_id", how="left")
    df["n_diag"] = df["n_diag"].fillna(0)
    df = df[df["admission_type"].isin(ACUITY_LEVEL)].copy()
    df["acuity"] = df["admission_type"].map(ACUITY_LEVEL)
    df = df.dropna(subset=["anchor_age", "n_diag", "acuity", "hospital_expire_flag"])
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PREPPED_PATH, index=False)
    return df


def _fit_harm_model(df: pd.DataFrame, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.stack(
        [
            df["anchor_age"].to_numpy(dtype=float) / AGE_SCALE,
            np.clip(df["n_diag"].to_numpy(dtype=float) / NDIAG_SCALE, 0, 3),
            df["acuity"].to_numpy(dtype=float),
        ],
        axis=1,
    )
    y = df["hospital_expire_flag"].to_numpy(dtype=int)
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=seed)
    model.fit(X, y)
    return model


def _load_and_fit(raw_dir: str = RAW_DIR, seed: int = 0):
    key = (raw_dir, seed)
    if key in _CACHE:
        return _CACHE[key]
    df = _prep_table(raw_dir)
    model = _fit_harm_model(df, seed=seed)
    covariates = np.stack(
        [
            df["anchor_age"].to_numpy(dtype=float) / AGE_SCALE,
            np.clip(df["n_diag"].to_numpy(dtype=float) / NDIAG_SCALE, 0, 3),
        ],
        axis=1,
    )
    result = {"df": df, "model": model, "covariates": covariates}
    _CACHE[key] = result
    return result


class MimicSafeBanditEnv:
    """Sequential bandit over real MIMIC-IV admissions with a fitted harm model.

    `a` is the chosen triage/admission intensity in [0, 1]. `H` is the
    fitted P(death | age, n_diag, a) at that action for a randomly drawn
    real patient. `U = UTIL_SCALE * (1 - a) + noise` rewards lower
    intensity (cheaper care), creating a genuine safety/utility tension:
    the utility-best action (a=0) is not the safest one for high-severity
    patients. See module docstring for the causal-identification caveat.
    """

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 40,
        tau_harm: float = 0.03,
        obs_noise: float = 0.05,
        raw_dir: str = RAW_DIR,
        model_seed: int = 0,
    ):
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.tau_harm = float(tau_harm)
        self.obs_noise = float(obs_noise)
        self.rng = np.random.RandomState(seed)
        loaded = _load_and_fit(raw_dir, seed=model_seed)
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
        age_n, ndiag_n = self.X[idx]
        query = np.array([[age_n, ndiag_n, a]], dtype=float)
        p_death = float(self.model.predict_proba(query)[0, 1])
        H = float(np.clip(p_death + self.rng.normal(0, 0.002), 0.0, 1.0))
        U = float(UTIL_SCALE * (1.0 - a) + self.rng.normal(0, 0.15))
        self.last_y = U
        self.t += 1
        done = self.t >= self.horizon
        obs = self._next_obs() if not done else np.array([0.0, 0.0, 0.0, self.last_y], dtype=float)
        return obs, U, H, done, {"patient": idx, "p_death": p_death}
