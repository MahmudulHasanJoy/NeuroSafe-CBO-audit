"""IHDP safe bandit with a binding harm constraint (v3).

Why this replaces env_ihdp_bandit.py for safety evaluation
-----------------------------------------------------------
The original environment used *rank-based* harm: within each arm, harm is
the patient's rank on that arm's outcome scale, linearly spaced from 1 to 0.
Ranks are uniform on [0,1] by construction, so P(H > 0.5) ~= 0.5 under
*any* policy. Every agent therefore violated at ~0.49 against alpha = 0.15,
and the safety axis could not distinguish methods.

Here utility is unchanged (fitted IHDP response surfaces), and harm is an
explicit adverse-event channel:
    treat    -> H ~ Bernoulli(p1(x)),  p1(x) = sigmoid((s(x) - q) / w)
    no-treat -> H ~ Bernoulli(p0),     p0 = 0.02
with s(x) the noise-free context signal (mean covariate, the same channel the
agent observes with noise), q the 70th percentile of s over patients, and
w = 0.1 * sd(s). IHDP has no adverse-event variable, so this channel is
CONSTRUCTED; what it fixes is that the constraint now binds for roughly the
riskiest ~30% of patients and treatment (utility-favoured for most patients)
is genuinely unsafe for them. Violation is H > tau with tau = 0.5 (H binary).
"""
from __future__ import annotations

import numpy as np

from .env_ihdp_bandit import _fit_surfaces, _load_ihdp_surfaces

_CACHE = {}


def _surfaces(rep):
    """Load and fit once per process (previously refit at every episode)."""
    if rep not in _CACHE:
        train, test = _load_ihdp_surfaces()
        m0, m1 = _fit_surfaces(train, rep=rep)
        X = test["x"][:, :, rep]
        _CACHE[rep] = (X, m0.predict(X), m1.predict(X))
    return _CACHE[rep]


class IHDPBindingEnv:
    def __init__(self, seed=0, horizon=200, tau_harm=0.5, obs_noise=0.05, rep=0,
                 p0=0.02, q_frac=0.70):
        self.rng = np.random.RandomState(seed)
        self.horizon, self.tau_harm, self.obs_noise = horizon, tau_harm, obs_noise
        self.X, self.mu0, self.mu1 = _surfaces(rep)
        s = self.X.mean(axis=1)
        q = np.quantile(s, q_frac)
        w = 0.1 * s.std()
        self.p1 = 1.0 / (1.0 + np.exp(-(s - q) / w))
        self.p0 = p0
        self.reset()

    def reset(self):
        self.t = 0
        self.last_y = 0.0
        return self._next_obs()

    def _next_obs(self):
        idx = int(self.rng.randint(0, len(self.X)))
        noisy = self.X[idx] + self.rng.normal(0, self.obs_noise, size=self.X.shape[1])
        self._cur = idx
        return np.array([float(noisy.mean()), float(noisy[0]), float(noisy[1]), self.last_y])

    def oracle(self, alpha):
        """Oracle knowing x: treat iff p1(x) <= alpha and mu1 > mu0."""
        i = self._cur
        return 1.0 if (self.p1[i] <= alpha and self.mu1[i] > self.mu0[i]) else 0.0

    def step(self, a):
        i = self._cur
        treat = float(a) >= 0.5
        U = float((self.mu1[i] if treat else self.mu0[i]) + self.rng.normal(0, 0.5))
        H = float(self.rng.rand() < (self.p1[i] if treat else self.p0))
        self.last_y = U
        self.t += 1
        return self._next_obs(), U, H, self.t >= self.horizon, {}
