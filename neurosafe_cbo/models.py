"""Predictive coding, RFF features, and online Bayesian linear models."""
from __future__ import annotations

import math

import numpy as np


def norm_cdf(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


class PredictiveCoder:
    def __init__(self, cfg: dict, seed: int = 0):
        r = np.random.RandomState(seed)
        self.W = r.normal(0, 0.3, (3, 1))
        self.z = np.zeros(1)
        self.steps = int(cfg["PC_STEPS"])
        self.lr_z = float(cfg["PC_LR_Z"])
        self.lr_w = float(cfg["PC_LR_W"])

    def infer(self, c: float, y: float, harm: float) -> float:
        o = np.array(
            [np.clip(c, -3, 3), np.clip(y / 2.0, -3, 3), np.clip(harm * 3.0, -3, 3)],
            dtype=float,
        )
        err = o - (self.W @ self.z).ravel()
        for _ in range(self.steps):
            err = o - (self.W @ self.z).ravel()
            self.z = self.z + self.lr_z * (self.W.T @ err).ravel()
            self.z *= 0.98
        self.W = self.W + self.lr_w * np.outer(err, self.z)
        return float(self.z[0])


class KalmanContext:
    """Correctly-specified predictive-coding context for an AR(1) latent.

    The original `PredictiveCoder` uses an untrained 1-D bottleneck whose latent
    correlates with the true confounder at r~0.09, i.e. it destroys the context
    signal that is present in the observation at r~0.93. Predictive coding for a
    linear-Gaussian latent *is* a Kalman filter (predict the next observation,
    correct by the prediction residual), so this is a faithful implementation of
    the cited mechanism rather than a different one.
    """

    def __init__(self, cfg: dict, seed: int = 0):
        self.rho = float(cfg.get("CTX_RHO", 0.80))
        self.q = float(cfg.get("CTX_Q", 0.06))
        self.r = float(cfg.get("CTX_R", 0.03))
        self.m = 0.0
        self.P = 1.0
        self.last_pred = 0.0

    def infer(self, c: float, y: float, harm: float) -> float:
        # predict
        m_pred = self.rho * self.m
        P_pred = self.rho * self.rho * self.P + self.q
        self.last_pred = m_pred
        # correct by prediction error on the observed context channel
        resid = float(np.clip(c, -5.0, 5.0)) - m_pred
        K = P_pred / (P_pred + self.r)
        self.m = m_pred + K * resid
        self.P = (1.0 - K) * P_pred
        return float(np.clip(self.m, -3.0, 3.0))


def make_context(cfg: dict, seed: int = 0):
    """Context encoder factory; 'pc' keeps the published behavior."""
    mode = str(cfg.get("CTX_MODE", "pc")).lower()
    if mode in ("kalman", "filter", "pc_fixed"):
        return KalmanContext(cfg, seed)
    return PredictiveCoder(cfg, seed)


class BucketedActionStats:
    """Context-conditional harm via per-(context-bucket, action) statistics.

    Keeps the calibration properties of ActionStats (which is why the project
    moved to it in the first place -- the RFF harm head was mis-scaled, mu_H~7
    against tau=0.2) while letting the feasible set depend on the current
    context instead of being marginalised over it.
    """

    def __init__(self, n_actions: int, edges, **kw):
        self.edges = np.asarray(edges, dtype=float)
        self.nb = len(self.edges) + 1
        self.banks = [ActionStats(n_actions, **kw) for _ in range(self.nb)]
        self.pooled = ActionStats(n_actions, **kw)

    def bucket(self, z: float) -> int:
        return int(np.searchsorted(self.edges, float(z)))

    def update(self, idx: int, y: float, z: float = 0.0) -> None:
        self.banks[self.bucket(z)].update(idx, y)
        self.pooled.update(idx, y)

    def predict(self, z: float = 0.0, min_n: float = 4.0):
        b = self.banks[self.bucket(z)]
        mu_b, s_b = b.predict()
        mu_p, s_p = self.pooled.predict()
        # shrink to the pooled estimate where the bucket is data-poor
        w = np.clip(b.n / (b.n + min_n), 0.0, 1.0)
        return w * mu_b + (1.0 - w) * mu_p, np.sqrt(w * s_b ** 2 + (1.0 - w) * s_p ** 2)


class RFF:
    """Random Fourier features with optional per-dimension lengthscales.

    The published code used one isotropic lengthscale for both the action and the
    context dimension. The utility surface here is narrow in `a` (peak width ~0.1)
    but smooth in `z`, so a single lengthscale either cannot resolve the peak or
    makes the fit wildly high-frequency in context. `sigma_z` decouples the two.
    """

    def __init__(self, dim: int, sigma: float, seed: int = 0, sigma_z: float | None = None):
        r = np.random.RandomState(seed)
        s_a = max(float(sigma), 1e-6)
        s_z = max(float(sigma_z if sigma_z is not None else sigma), 1e-6)
        self.W = np.stack(
            [r.normal(0, 1.0 / s_a, dim), r.normal(0, 1.0 / s_z, dim)], axis=1
        )
        self.b = r.uniform(0, 2.0 * np.pi, dim)
        self.scale = math.sqrt(2.0 / dim)

    def phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return self.scale * np.cos(self.W @ x + self.b)
        return self.scale * np.cos(x @ self.W.T + self.b)


class BayesLin:
    def __init__(self, dim: int, noise: float, forget: float):
        self.m = np.zeros(dim)
        self.p = np.ones(dim)
        self.noise2 = float(noise) ** 2
        self.forget = float(forget)

    def predict(self, phi: np.ndarray):
        phi = np.asarray(phi, dtype=float)
        if phi.ndim == 1:
            mu = float(phi @ self.m)
            var = self.noise2 + float(np.sum((phi ** 2) / self.p))
            return mu, math.sqrt(max(var, 1e-12))
        mu = phi @ self.m
        var = self.noise2 + np.sum((phi ** 2) / self.p, axis=1)
        return mu, np.sqrt(np.maximum(var, 1e-12))

    def update(self, phi: np.ndarray, y: float) -> None:
        phi = np.asarray(phi, dtype=float)
        self.p = self.forget * self.p + (1.0 - self.forget)
        self.m = self.forget * self.m
        inv = 1.0 / self.noise2
        p_new = self.p + (phi ** 2) * inv
        self.m = (self.p * self.m + phi * float(y) * inv) / p_new
        self.p = p_new


class ActionStats:
    def __init__(
        self,
        n_actions: int,
        *,
        prior_mu: float = 0.0,
        prior_var: float = 0.05,
        prior_n: float = 2.0,
        forget: float = 0.995,
        obs_var: float = 0.01,
    ):
        self.n = np.full(n_actions, float(prior_n))
        self.mu = np.full(n_actions, float(prior_mu))
        self.ss = np.full(n_actions, float(prior_var) * prior_n)
        self.forget = float(forget)
        self.obs_var = float(obs_var)

    def update(self, idx: int, y: float) -> None:
        idx = int(idx)
        self.n *= self.forget
        self.ss *= self.forget
        n = self.n[idx]
        mu = self.mu[idx]
        self.n[idx] = n + 1.0
        delta = float(y) - mu
        self.mu[idx] = mu + delta / self.n[idx]
        self.ss[idx] = self.ss[idx] + delta * (float(y) - self.mu[idx])

    def predict(self):
        var = self.ss / np.maximum(self.n, 1e-6)
        var = np.maximum(var, self.obs_var)
        s = np.sqrt(var / np.maximum(self.n, 1e-6) + self.obs_var)
        return self.mu.copy(), s
