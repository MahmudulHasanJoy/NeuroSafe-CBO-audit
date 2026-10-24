"""Off-policy evaluation for the MIMIC vasopressor domain."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BehaviourPolicy:
    def __init__(self, clip=1e-3, seed=0):
        self.clip = clip; self.seed = seed
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, C=1.0)
        self.diag = {}
    def fit(self, S, a):
        Xtr, Xte, ytr, yte = train_test_split(S, a, test_size=0.25, random_state=self.seed, stratify=a)
        self.scaler.fit(Xtr); self.model.fit(self.scaler.transform(Xtr), ytr)
        p = self.model.predict_proba(self.scaler.transform(Xte))
        idx = np.searchsorted(self.model.classes_, yte)
        ll = float(np.mean(np.log(np.clip(p[np.arange(len(yte)), idx], 1e-12, 1))))
        base = np.bincount(ytr, minlength=len(self.model.classes_)) / len(ytr)
        ll0 = float(np.mean(np.log(np.clip(base[idx], 1e-12, 1))))
        self.diag = {"heldout_loglik": ll, "marginal_loglik": ll0,
                     "pseudo_r2": 1 - ll / ll0 if ll0 else None,
                     "accuracy": float((self.model.predict(self.scaler.transform(Xte)) == yte).mean()),
                     "n_fit": int(len(ytr)), "min_propensity": float(p.min())}
        self.scaler.fit(S); self.model.fit(self.scaler.transform(S), a)
        return self
    def proba(self, S):
        return np.clip(self.model.predict_proba(self.scaler.transform(S)), self.clip, 1.0)

def weights(pi_e, pi_b, a, clip_ratio=20.0):
    n = len(a)
    w = pi_e[np.arange(n), a] / np.maximum(pi_b[np.arange(n), a], 1e-12)
    return np.minimum(w, clip_ratio), int((w > clip_ratio).sum())

def ess(w):
    s1, s2 = w.sum(), (w ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0

def wis(r, w):
    sw = w.sum()
    return float((w * r).sum() / sw) if sw > 0 else float("nan")

def boot_ci(r, w, B=2000, seed=0):
    rng = np.random.RandomState(seed); n = len(r); out = np.empty(B)
    for b in range(B):
        i = rng.randint(0, n, n); out[b] = wis(r[i], w[i])
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))

class QModel:
    def __init__(self, n_actions, seed=0):
        self.K = n_actions
        self.models = [HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=seed)
                       for _ in range(n_actions)]
        self.fitted = np.zeros(n_actions, dtype=bool); self.fallback = 0.0
    def fit(self, S, a, r):
        self.fallback = float(np.mean(r))
        for k in range(self.K):
            m = a == k
            if m.sum() >= 50:
                self.models[k].fit(S[m], r[m]); self.fitted[k] = True
        return self
    def predict_all(self, S):
        out = np.full((len(S), self.K), self.fallback)
        for k in range(self.K):
            if self.fitted[k]:
                out[:, k] = self.models[k].predict(S)
        return out

def dr(r, S, a, pi_e, pi_b, q, clip_ratio=20.0):
    w, _ = weights(pi_e, pi_b, a, clip_ratio)
    Q = q.predict_all(S); n = len(a)
    return float(np.mean((pi_e * Q).sum(axis=1) + w * (r - Q[np.arange(n), a])))

def evaluate(policy_probs, S, a, reward, harm, pi_b, q_r=None, q_h=None, clip_ratio=20.0, seed=0):
    w, n_clip = weights(policy_probs, pi_b, a, clip_ratio)
    e = ess(w)
    out = {"n_decisions": int(len(a)), "ess": e, "ess_frac": e / len(a),
           "weight_max": float(w.max()), "weight_mean": float(w.mean()),
           "n_weights_clipped": n_clip, "wis_return": wis(reward, w), "wis_harm": wis(harm, w),
           "behaviour_return": float(reward.mean()), "behaviour_harm": float(harm.mean())}
    lo, hi = boot_ci(reward, w, seed=seed); out["wis_return_ci"] = [lo, hi]
    lo, hi = boot_ci(harm, w, seed=seed); out["wis_harm_ci"] = [lo, hi]
    if q_r is not None: out["dr_return"] = dr(reward, S, a, policy_probs, pi_b, q_r, clip_ratio)
    if q_h is not None: out["dr_harm"] = dr(harm, S, a, policy_probs, pi_b, q_h, clip_ratio)
    out["reliable"] = bool(e >= 200 and e / len(a) >= 0.01)
    return out
