"""Safe UCB-GLM baseline in the style of Amani, Alizadeh & Thrampoulidis (ICASSP 2020).

Differences from NeuroSafe-Ctx, chosen to follow that paper's design:
  * frequentist confidence sets from the design matrix V_t = sum phi phi^T + lam I (not the Bayesian
    Laplace posterior), with ridge-regularised logistic MLE for the harm event 1{H > tau};
  * no forgetting; the raw observed context (no filter); a ridge-regression UCB for utility on the
    same features (no context buckets);
  * plays the most optimistic action among those whose harm upper bound
    sigmoid(theta^T phi + beta_h ||phi||_{V^-1}) <= alpha, else the safe seed a = 0.
Warm-up (10 random rounds) and features ([1,a,a^2,a^3] x [1,z]) match NeuroSafe-Ctx for comparability."""
import numpy as np
from neurosafe_cbo.v3 import glm_features, _Standardizer, _sigmoid

class SafeGLMUCB:
    def __init__(self, actions, seed=0, tau=0.2, alpha=0.15, beta_h=1.0, beta_u=0.5, lam=1.0, warmup=10, newton_iters=3):
        self.actions = np.asarray(actions, dtype=float); self.K = len(self.actions)
        self.rng = np.random.RandomState(seed)
        self.tau, self.alpha, self.beta_h, self.beta_u, self.lam, self.warmup = tau, alpha, beta_h, beta_u, lam, warmup
        self.iters = newton_iters; self.d = 8; self.std = _Standardizer(); self.t = 0
        self.X, self.yh, self.yu = [], [], []
        self.th = np.zeros(self.d); self.Vinv = np.eye(self.d) / lam; self.wu = np.zeros(self.d)
        self.stats = {"decisions": 0, "empty_feasible": 0}

    def act(self, obs):
        self.t += 1
        z = float(obs[0]); self.std.update(z)
        if self.t <= self.warmup:
            return float(self.rng.choice(self.actions)), z
        Phi = glm_features(self.actions, float(self.std(z)), "cubic8")
        w = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Phi, self.Vinv, Phi), 0.0))
        p_up = _sigmoid(Phi @ self.th + self.beta_h * w)
        ok = p_up <= self.alpha; self.stats["decisions"] += 1
        if not ok.any():
            self.stats["empty_feasible"] += 1
            return float(self.actions[0]), z
        score = np.where(ok, Phi @ self.wu + self.beta_u * w, -np.inf)
        return float(self.actions[int(np.argmax(score))]), z

    def observe(self, a, z, U, H):
        idx = int(np.argmin(np.abs(self.actions - float(a))))
        x = glm_features(np.array([self.actions[idx]]), float(self.std(z)), "cubic8")[0]
        self.X.append(x); self.yh.append(float(H > self.tau)); self.yu.append(float(U))
        X, Y, Uu = np.asarray(self.X), np.asarray(self.yh), np.asarray(self.yu)
        V = X.T @ X + self.lam * np.eye(self.d); self.Vinv = np.linalg.inv(V)
        self.wu = self.Vinv @ (X.T @ Uu)                      # ridge regression for utility
        th = self.th.copy()                                   # ridge-regularised logistic MLE (Newton)
        for _ in range(self.iters):
            p = _sigmoid(X @ th)
            g = X.T @ (p - Y) + self.lam * th
            Hs = (X * (p * (1 - p))[:, None]).T @ X + self.lam * np.eye(self.d)
            th = th - np.linalg.solve(Hs, g)
        self.th = th
