"""MIMIC-IV norepinephrine dosing as a safe contextual bandit (v3 interface).

Same interface as IHDPBindingEnv: reset() -> obs, step(a) -> (obs, U, H, done, info),
oracle(alpha) -> action in [0, 1].

Fixed before any v3 MIMIC run:
- one simulator per process, fitted on the FIT half of the hypotensive on-drug cohort
  (stays shuffled with RandomState(0), first 50%), random_state=0; the held-out half
  is reserved for off-policy evaluation
- utility U = (next MAP - 70) / 10  (affine rescale; report ret * 10 + 70 in mmHg)
- harm H = new arrhythmia in the next window (binary); violation is H > 0.5
- observation [MAP - 70, FEATS[1] - 70, 0, last U]: the previous dose is the clinician's
  own past action and is hidden from every agent (the simulator still uses it)
- 5 dose bins; action a in [0, 1] -> bin round(4a)
Patient-level data never leaves this machine; nothing here is written to disk.
NOT used for any reported result: the MIMIC audit (experiments/mimic_audit.py) uses
cross-validated models and real-data off-policy evaluation. This adapter keeps the
original, overfitted harm model and exists only for exploratory simulator runs.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

K = 5
PKL = "results/mimic_windows.pkl"
_CACHE = {}


def _load():
    if "sim" not in _CACHE:
        from experiments.run_mimic_domain import FEATS
        df = pd.read_pickle(PKL)
        h = df[df.hypotensive & (df.prev_dbin > 0)].dropna(subset=FEATS + ["map_now"]).copy()
        h["new_arr"] = h.new_arr.astype(float)
        stays = np.array(sorted(h.stay_id.unique())); r = np.random.RandomState(0); r.shuffle(stays)
        cut = int(.5 * len(stays))
        fit = h[h.stay_id.isin(set(stays[:cut]))]; ope = h[h.stay_id.isin(set(stays[cut:]))]
        S = fit[FEATS].to_numpy(float)
        X = np.column_stack([S, fit.dbin.to_numpy(int)])
        y = fit.map_now.to_numpy(float)
        m_map = HistGradientBoostingRegressor(max_depth=4, max_iter=250, random_state=0).fit(X, y)
        m_arr = HistGradientBoostingClassifier(max_depth=4, max_iter=250, random_state=0).fit(X, fit.new_arr.to_numpy(float))
        _CACHE["sim"] = dict(S=S, m_map=m_map, m_arr=m_arr, res_sd=float(np.std(y - m_map.predict(X))),
                             fit=fit, ope=ope, feats=FEATS)
    return _CACHE["sim"]


class MimicBandit:
    def __init__(self, seed=0, horizon=400, tau_harm=0.5):
        c = _load()
        self.S, self.m_map, self.m_arr, self.res_sd = c["S"], c["m_map"], c["m_arr"], c["res_sd"]
        self.rng = np.random.RandomState(seed)
        self.horizon, self.tau_harm = horizon, tau_harm
        self.reset()

    def reset(self):
        self.t, self.last_u = 0, 0.0
        return self._next_obs()

    def _next_obs(self):
        self._cur = int(self.rng.randint(len(self.S)))
        s = self.S[self._cur]
        return np.array([s[0] - 70.0, s[1] - 70.0, 0.0, self.last_u])

    def oracle(self, alpha):
        X = np.column_stack([np.repeat(self.S[self._cur][None, :], K, axis=0), np.arange(K)])
        p = self.m_arr.predict_proba(X)[:, 1]; m = self.m_map.predict(X)
        ok = p <= alpha
        k = int(np.argmax(np.where(ok, m, -np.inf))) if ok.any() else int(np.argmin(p))
        return k / (K - 1)

    def step(self, a):
        k = int(np.clip(np.round(float(a) * (K - 1)), 0, K - 1))
        x = np.concatenate([self.S[self._cur], [k]])[None, :]
        mp = float(self.m_map.predict(x)[0]) + self.rng.normal(0, self.res_sd)
        H = 1.0 if self.rng.rand() < float(self.m_arr.predict_proba(x)[0, 1]) else 0.0
        U = (mp - 70.0) / 10.0
        self.last_u = U; self.t += 1
        return self._next_obs(), U, H, self.t >= self.horizon, {}
