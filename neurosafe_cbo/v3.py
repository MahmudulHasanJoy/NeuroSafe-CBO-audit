"""NeuroSafe-Ctx (v3): the method the paper describes, implemented as described.

Components
----------
1. ``AdaptiveKalman``   -- moment-matched AR(1)+noise filter on the observed
   context channel. The AR coefficient, state variance and observation noise
   are estimated online from lag-1/lag-2 autocovariances, so no per-domain
   tuning is needed. If the context is (close to) i.i.d. the filter is the
   identity, because an AR(1) latent is not identifiable from i.i.d. data and
   any monotone shrinkage would not change the decision rule anyway.
2. ``DiscountedLogisticGLM`` -- Bayesian logistic regression of the Bernoulli
   target 1{H > tau} on phi(a, z) = [1, a, z, a z, a^2], with exponential
   forgetting (weights gamma^age), a Gaussian prior, and a full-covariance
   Laplace approximation refit by warm-started Newton steps every round.
   Feasibility is certified optimistically-for-safety:
       p_up(a) = sigmoid(theta^T phi + beta_h * ||phi||_{Sigma}) <= alpha.
3. ``BucketedUtility`` -- per-(context-bucket, action) utility means with
   shrinkage toward the per-action pooled mean, weight n/(n+m). Bucket edges
   are empirical quantiles of the context history (adaptive; no hand-set
   edges -- this removes the CTX_EDGES bug class entirely).

Baselines
---------
* ``SafeGPUCB``  -- contextual safe GP-UCB in the style of SafeOpt/StageOpt
  (GP on harm certifies a safe set, GP-UCB on utility selects inside it).
  THIS IS OUR REIMPLEMENTATION, not the reference `safeopt` package (which
  is not installable offline); it is disclosed as such in the paper.
* ``ConservativeUCB`` -- conservative linear UCB (Kazerouni et al. 2017 /
  Wu et al. 2016 style) with the safest action as the reference policy.

All agents expose act(obs) -> (a, z) and observe(a, z, U, H).
"""
from __future__ import annotations

import numpy as np

from .config import ACTIONS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
class AdaptiveKalman:
    def __init__(self, min_hist: int = 30, rho_floor: float = 0.2):
        self.hist: list[float] = []
        self.min_hist = min_hist
        self.rho_floor = rho_floor
        self.m = 0.0
        self.P = 1.0
        self.params = None  # (rho, q, r, mean)

    def _fit(self):
        o = np.asarray(self.hist[-400:], dtype=float)
        mu = o.mean()
        d = o - mu
        v0 = float(np.mean(d * d))
        c1 = float(np.mean(d[1:] * d[:-1]))
        c2 = float(np.mean(d[2:] * d[:-2]))
        if c1 <= 1e-9 or c1 / max(v0, 1e-9) < self.rho_floor:
            return None
        rho = float(np.clip(c2 / c1, 0.05, 0.99)) if c2 > 0 else float(np.clip(c1 / v0, 0.05, 0.99))
        s2 = min(c1 / rho, v0 * 0.999)
        r = max(v0 - s2, 1e-4)
        q = max(s2 * (1.0 - rho * rho), 1e-5)
        return rho, q, r, mu

    def infer(self, c: float) -> float:
        self.hist.append(float(c))
        n = len(self.hist)
        if n < self.min_hist:
            return float(c)
        if n == self.min_hist or n % 25 == 0:
            self.params = self._fit()
        if self.params is None:
            return float(c)
        rho, q, r, mu = self.params
        m_pred = mu + rho * (self.m - mu)
        P_pred = rho * rho * self.P + q
        K = P_pred / (P_pred + r)
        self.m = m_pred + K * (float(c) - m_pred)
        self.P = (1.0 - K) * P_pred
        return float(self.m)


class RawContext:
    def infer(self, c: float) -> float:
        return float(c)


class NoContext:
    """Ablation: context-blind (z is constant), isolates context-conditional feasibility."""

    def infer(self, c: float) -> float:
        return 0.0


class _Standardizer:
    """Running standardization of z so features are domain-scale free."""

    def __init__(self):
        self.n = 0
        self.mu = 0.0
        self.m2 = 0.0

    def update(self, z):
        self.n += 1
        d = z - self.mu
        self.mu += d / self.n
        self.m2 += d * (z - self.mu)

    def __call__(self, z):
        sd = np.sqrt(self.m2 / self.n) if self.n > 5 else 1.0
        return (np.asarray(z) - self.mu) / max(sd, 1e-6)


# ---------------------------------------------------------------------------
# Harm: discounted Bayesian logistic GLM (full-covariance Laplace)
# ---------------------------------------------------------------------------
def glm_features(a, zs, basis="quad5"):
    """quad5: [1, a, z, az, a^2] (d=5).  cubic8: [1,a,a^2,a^3] (x) [1,z] (d=8)."""
    a = np.asarray(a, dtype=float)
    zs = np.broadcast_to(np.asarray(zs, dtype=float), a.shape)
    if basis == "cubic8":
        P = [np.ones_like(a), a, a * a, a ** 3]
        return np.stack(P + [p * zs for p in P], axis=-1)
    return np.stack([np.ones_like(a), a, zs, a * zs, a * a], axis=-1)


class DiscountedLogisticGLM:
    def __init__(self, dim: int = 5, prior_var: float = 4.0, gamma: float = 0.995,
                 newton_iters: int = 3):
        self.d = dim
        self.lam = 1.0 / prior_var
        self.gamma = gamma
        self.iters = newton_iters
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.theta = np.zeros(dim)
        self.Sigma = np.eye(dim) * prior_var

    def update(self, x, y):
        self.X.append(np.asarray(x, dtype=float))
        self.y.append(float(y))
        X = np.asarray(self.X)
        Y = np.asarray(self.y)
        w = self.gamma ** np.arange(len(Y) - 1, -1, -1)
        th = self.theta.copy()
        for _ in range(self.iters):
            p = _sigmoid(X @ th)
            g = X.T @ (w * (p - Y)) + self.lam * th
            Hs = (X * (w * p * (1 - p))[:, None]).T @ X + self.lam * np.eye(self.d)
            th = th - np.linalg.solve(Hs, g)
        p = _sigmoid(X @ th)
        Hs = (X * (w * p * (1 - p))[:, None]).T @ X + self.lam * np.eye(self.d)
        self.theta = th
        self.Sigma = np.linalg.inv(Hs)

    def upper(self, Phi, beta):
        mean = Phi @ self.theta
        width = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Phi, self.Sigma, Phi), 0.0))
        return _sigmoid(mean + beta * width), _sigmoid(mean)


# ---------------------------------------------------------------------------
# Harm ablation: per-(bucket, action) mean harm pushed through a Gaussian CDF
# ---------------------------------------------------------------------------
class GaussCellHarm:
    """The pre-v3 harm path: cell-wise mean/var of H, P(H<=tau) via Gaussian CDF."""

    def __init__(self, K, tau):
        self.K = K
        self.tau = tau
        self.rec: list[tuple[int, float, float]] = []

    def update(self, idx, zs, H):
        self.rec.append((idx, zs, H))

    def p_viol(self, zs, edges):
        from scipy.stats import norm
        if not self.rec:
            return np.full(self.K, 0.5)
        R = np.asarray(self.rec)
        idx = R[:, 0].astype(int)
        b_all = np.searchsorted(edges, R[:, 1])
        b = int(np.searchsorted(edges, zs))
        out = np.empty(self.K)
        for k in range(self.K):
            cell = R[(idx == k) & (b_all == b), 2]
            pool = R[idx == k, 2]
            n = len(cell)
            w = n / (n + 4.0)
            mu = (cell.mean() if n else 0.08) * w + (pool.mean() if len(pool) else 0.08) * (1 - w)
            var = (cell.var() if n > 1 else 0.02) * w + (pool.var() if len(pool) > 1 else 0.02) * (1 - w)
            sd = np.sqrt(var + 0.02 / (n + 3.0))
            out[k] = 1.0 - norm.cdf((self.tau - mu) / max(sd, 1e-6))
        return out


# ---------------------------------------------------------------------------
# Utility: context-bucketed table with pooled shrinkage
# ---------------------------------------------------------------------------
class BucketedUtility:
    def __init__(self, K, n_buckets=4, shrink=3.0):
        self.K = K
        self.B = n_buckets
        self.m = shrink
        self.a_idx: list[int] = []
        self.z: list[float] = []
        self.u: list[float] = []

    def edges(self):
        if len(self.z) < 4 * self.B:
            return np.array([])
        qs = np.linspace(0, 1, self.B + 1)[1:-1]
        return np.quantile(np.asarray(self.z), qs)

    def update(self, idx, zs, U):
        self.a_idx.append(int(idx))
        self.z.append(float(zs))
        self.u.append(float(U))

    def predict(self, zs, contextual=True):
        K = self.K
        if not self.u:
            return np.zeros(K), np.full(K, 10.0)
        A = np.asarray(self.a_idx)
        U = np.asarray(self.u)
        sig = max(float(np.std(U)), 0.05)
        n_p = np.bincount(A, minlength=K).astype(float)
        s_p = np.bincount(A, weights=U, minlength=K)
        mu_p = np.where(n_p > 0, s_p / np.maximum(n_p, 1), U.mean())
        if not contextual:
            return mu_p, sig / np.sqrt(n_p + 1.0)
        e = self.edges()
        b_all = np.searchsorted(e, np.asarray(self.z))
        b = int(np.searchsorted(e, zs))
        sel = b_all == b
        n_c = np.bincount(A[sel], minlength=K).astype(float)
        s_c = np.bincount(A[sel], weights=U[sel], minlength=K)
        mu = (s_c + self.m * mu_p) / (n_c + self.m)
        return mu, sig / np.sqrt(n_c + self.m)


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------
class NeuroSafeCtxV3:
    def __init__(self, actions=None, seed=0, tau=0.2, alpha=0.15, beta_h=1.0,
                 beta_u=0.5, gamma=0.995, n_buckets=4, shrink=3.0, warmup=10,
                 context="kalman", harm="glm", utility="bucket", basis="quad5"):
        self.actions = np.asarray(actions if actions is not None else ACTIONS, dtype=float)
        self.K = len(self.actions)
        self.rng = np.random.RandomState(seed)
        self.tau, self.alpha, self.beta_h, self.beta_u = tau, alpha, beta_h, beta_u
        self.warmup = warmup
        self.ctx = {"kalman": AdaptiveKalman, "raw": RawContext, "none": NoContext}[context]()
        self.std = _Standardizer()
        self.harm_mode = harm
        self.basis = basis
        self.glm = DiscountedLogisticGLM(dim=8 if basis == "cubic8" else 5, gamma=gamma)
        self.gauss = GaussCellHarm(self.K, tau)
        self.util = BucketedUtility(self.K, n_buckets, shrink)
        self.utility_mode = utility
        self.t = 0
        self.stats = {"decisions": 0, "empty_feasible": 0}

    def _p_viol(self, zs):
        if self.harm_mode == "glm":
            up, _ = self.glm.upper(glm_features(self.actions, zs, self.basis), self.beta_h)
            return up
        return self.gauss.p_viol(zs, self.util.edges())

    def act(self, obs):
        self.t += 1
        z = self.ctx.infer(float(obs[0]))
        self.std.update(z)
        if self.t <= self.warmup:
            return float(self.rng.choice(self.actions)), z
        zs = float(self.std(z))
        pv = self._p_viol(zs)
        feasible = pv <= self.alpha
        self.stats["decisions"] += 1
        if not feasible.any():
            self.stats["empty_feasible"] += 1
            return float(self.actions[int(np.argmin(pv))]), z
        mu, s = self.util.predict(zs, contextual=(self.utility_mode == "bucket"))
        score = np.where(feasible, mu + self.beta_u * s, -np.inf)
        return float(self.actions[int(np.argmax(score))]), z

    def observe(self, a, z, U, H):
        idx = int(np.argmin(np.abs(self.actions - float(a))))
        zs = float(self.std(z))
        self.glm.update(glm_features(np.array([self.actions[idx]]), zs, self.basis)[0],
                        float(H > self.tau))
        self.gauss.update(idx, zs, float(H))
        self.util.update(idx, zs, float(U))


# ---------------------------------------------------------------------------
# Baseline: contextual safe GP-UCB (SafeOpt/StageOpt-style reimplementation)
# ---------------------------------------------------------------------------
class _NumpyGP:
    def __init__(self, ls=(0.15, 0.6), noise=0.05, window=150):
        self.ls = np.asarray(ls, dtype=float)
        self.noise = noise
        self.window = window
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def _k(self, A, B):
        d = (A[:, None, :] - B[None, :, :]) / self.ls
        return np.exp(-0.5 * np.sum(d * d, axis=-1))

    def add(self, x, y):
        self.X.append(np.asarray(x, dtype=float))
        self.y.append(float(y))

    def predict(self, Xs):
        if not self.y:
            return np.zeros(len(Xs)), np.ones(len(Xs))
        X = np.asarray(self.X[-self.window:])
        y = np.asarray(self.y[-self.window:])
        ym, ysd = y.mean(), max(y.std(), 1e-3)
        yn = (y - ym) / ysd
        Kxx = self._k(X, X) + self.noise * np.eye(len(X))
        L = np.linalg.cholesky(Kxx)
        al = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        Ks = self._k(Xs, X)
        mu = Ks @ al
        v = np.linalg.solve(L, Ks.T)
        var = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-9)
        return mu * ysd + ym, np.sqrt(var) * ysd


class SafeGPUCB:
    """Safe set = {a : mu_H + beta*sd_H <= tau}; play GP-UCB utility inside it.
    Given the *observed* context channel directly (a concession in its favour)."""

    def __init__(self, actions=None, seed=0, tau=0.2, beta=2.0, beta_u=1.0, warmup=10):
        self.actions = np.asarray(actions if actions is not None else ACTIONS, dtype=float)
        self.rng = np.random.RandomState(seed)
        self.tau, self.beta, self.beta_u, self.warmup = tau, beta, beta_u, warmup
        self.std = _Standardizer()
        self.gu = _NumpyGP()
        self.gh = _NumpyGP()
        self.t = 0

    def act(self, obs):
        self.t += 1
        z = float(obs[0])
        self.std.update(z)
        if self.t <= self.warmup:
            return float(self.rng.choice(self.actions)), z
        Xs = np.stack([self.actions, np.full(len(self.actions), float(self.std(z)))], axis=1)
        mh, sh = self.gh.predict(Xs)
        safe = mh + self.beta * sh <= self.tau
        if not safe.any():
            return float(self.actions[int(np.argmin(mh + self.beta * sh))]), z
        mu, su = self.gu.predict(Xs)
        score = np.where(safe, mu + self.beta_u * su, -np.inf)
        return float(self.actions[int(np.argmax(score))]), z

    def observe(self, a, z, U, H):
        x = np.array([float(a), float(self.std(z))])
        self.gu.add(x, U)
        self.gh.add(x, H)


# ---------------------------------------------------------------------------
# Baseline: conservative linear UCB (Kazerouni et al. 2017 style)
# ---------------------------------------------------------------------------
class ConservativeUCB:
    """Linear UCB on [1,a,z,az,a^2]; deviates from the reference (safest) action
    only while the pessimistic cumulative reward stays >= (1-eta) x reference."""

    def __init__(self, actions=None, seed=0, eta=0.1, beta_u=1.0, warmup=10, lam=1.0):
        self.actions = np.asarray(actions if actions is not None else ACTIONS, dtype=float)
        self.rng = np.random.RandomState(seed)
        self.eta, self.beta_u, self.warmup = eta, beta_u, warmup
        self.std = _Standardizer()
        self.A = lam * np.eye(5)
        self.b = np.zeros(5)
        self.ref_idx = 0  # safest action (a = 0) is the reference policy
        self.cum_lcb = 0.0
        self.cum_ref = 0.0
        self.ref_sum = 0.0
        self.ref_n = 0
        self.t = 0

    def act(self, obs):
        self.t += 1
        z = float(obs[0])
        self.std.update(z)
        if self.t <= self.warmup:
            return float(self.rng.choice(self.actions)), z
        Phi = glm_features(self.actions, float(self.std(z)))
        Ainv = np.linalg.inv(self.A)
        th = Ainv @ self.b
        mu = Phi @ th
        w = np.sqrt(np.einsum("ij,jk,ik->i", Phi, Ainv, Phi))
        k = int(np.argmax(mu + self.beta_u * w))
        ref_mean = self.ref_sum / self.ref_n if self.ref_n else mu[self.ref_idx]
        if self.cum_lcb + (mu[k] - self.beta_u * w[k]) >= (1 - self.eta) * (self.cum_ref + ref_mean):
            self._pending = (mu[k] - self.beta_u * w[k], ref_mean)
            return float(self.actions[k]), z
        self._pending = (ref_mean, ref_mean)
        return float(self.actions[self.ref_idx]), z

    def observe(self, a, z, U, H):
        idx = int(np.argmin(np.abs(self.actions - float(a))))
        phi = glm_features(np.array([self.actions[idx]]), float(self.std(z)))[0]
        self.A += np.outer(phi, phi)
        self.b += phi * U
        if idx == self.ref_idx:
            self.ref_sum += U
            self.ref_n += 1
        if self.t > self.warmup and hasattr(self, "_pending"):
            lcb, ref = self._pending
            self.cum_lcb += lcb
            self.cum_ref += ref
