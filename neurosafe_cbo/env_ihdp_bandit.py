"""IHDP sequential safe bandit environment (Domain 2).

Each step samples a patient covariate vector; the agent chooses treat (1) or
no-treat (0). Outcomes are drawn from fitted IHDP response surfaces with noise;
harm is rank-based on the counterfactual outcome scale.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "q1_out")


def _download_npz(urls, path):
    import requests

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


def _load_ihdp_surfaces():
    train = _download_npz(
        [
            "https://raw.githubusercontent.com/Osier-Yi/SITE/master/data/ihdp_npci_1-100.train.npz",
            "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
        ],
        os.path.join(DATA_DIR, "train.npz"),
    )
    test = _download_npz(
        [
            "https://raw.githubusercontent.com/Osier-Yi/SITE/master/data/ihdp_npci_1-100.test.npz",
            "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
        ],
        os.path.join(DATA_DIR, "test.npz"),
    )
    return train, test


def _fit_surfaces(train, rep: int = 0):
    X = train["x"][:, :, rep]
    T = train["t"][:, rep]
    Y = train["yf"][:, rep]
    m0 = HistGradientBoostingRegressor(max_depth=3, max_iter=80, random_state=0).fit(
        X[T == 0], Y[T == 0]
    )
    m1 = HistGradientBoostingRegressor(max_depth=3, max_iter=80, random_state=1).fit(
        X[T == 1], Y[T == 1]
    )
    return m0, m1


class IHDPSafeBanditEnv:
    """Sequential bandit over IHDP patients with rank-based harm."""

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 40,
        tau_harm: float = 0.5,
        obs_noise: float = 0.05,
        rep: int = 0,
    ):
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.tau_harm = float(tau_harm)
        self.obs_noise = float(obs_noise)
        self.rng = np.random.RandomState(seed)
        train, test = _load_ihdp_surfaces()
        self.m0, self.m1 = _fit_surfaces(train, rep=rep)
        self.X = test["x"][:, :, rep]
        self.n_patients = self.X.shape[0]
        self.t = 0
        self._harm_cache: dict | None = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.last_y = 0.0
        return self._next_obs()

    def _rank_harm(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        order = y.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(1.0, 0.0, len(y))
        return ranks

    def _ensure_harm_cache(self):
        if self._harm_cache is not None:
            return
        mu0 = self.m0.predict(self.X)
        mu1 = self.m1.predict(self.X)
        self._harm_cache = {
            "h0": self._rank_harm(mu0),
            "h1": self._rank_harm(mu1),
            "mu0": mu0,
            "mu1": mu1,
        }

    def _next_obs(self) -> np.ndarray:
        idx = int(self.rng.randint(0, self.n_patients))
        x = self.X[idx]
        noisy = x + self.rng.normal(0, self.obs_noise, size=x.shape)
        # SCM-compatible obs: (C, x1, x2, y) for NeuroSafe agent / PC
        c = float(np.mean(noisy))
        x1 = float(noisy[0]) if len(noisy) > 0 else 0.0
        x2 = float(noisy[1]) if len(noisy) > 1 else 0.0
        y = float(self.last_y)
        self._cur_idx = idx
        return np.array([c, x1, x2, y], dtype=float)

    def step(self, a: float):
        self._ensure_harm_cache()
        idx = self._cur_idx
        treat = float(a) >= 0.5
        mu = self._harm_cache["mu1"][idx] if treat else self._harm_cache["mu0"][idx]
        U = float(mu + self.rng.normal(0, 0.5))
        H = float(self._harm_cache["h1"][idx] if treat else self._harm_cache["h0"][idx])
        self.last_y = U
        self.t += 1
        done = self.t >= self.horizon
        obs = self._next_obs() if not done else np.array([0.0, 0.0, 0.0, self.last_y], dtype=float)
        return obs, U, H, done, {"patient": idx, "treat": treat}
