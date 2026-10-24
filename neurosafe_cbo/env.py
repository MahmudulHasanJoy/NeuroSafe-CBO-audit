"""Nonstationary structural causal model environment."""

from __future__ import annotations

import copy

import numpy as np


class NonstationarySCMEnv:
    """Causal SCM with a distribution shift in treatment/confounder effects.

    Latent confounders (U, C) evolve as AR(1). Observed covariates are noisy
    functions of C and previous endogenous variables. Utility is outcome y;
    harm is a bounded transform of negative utility.
    """

    def __init__(self, shift_at: int, shift_mag: float, obs_noise: float, seed: int = 0, harm_mode: str = "tradeoff"):
        self.shift_at = int(shift_at)
        self.shift_mag = float(shift_mag)
        self.obs_noise = float(obs_noise)
        self.seed = int(seed)
        self.harm_mode = harm_mode
        self.rng = np.random.RandomState(seed)
        self.reset()

    # --- decoupled-regime constants (harm_mode="decoupled") ------------------
    # Harm rides a latent risk channel R that is structurally distinct from the
    # utility drivers, and the action->risk gain jumps at the shift so the
    # FEASIBLE SET itself moves. Utility gets a context-dependent peak located
    # by the latent confounder C, so locating the optimum requires inferring
    # context -- a context-blind chance-constrained agent cannot find it.
    DEC = {
        "risk_gain_pre": 2.2,    # action -> risk slope before shift
        "risk_gain_post": 4.0,   # after shift (feasible boundary contracts)
        "risk_rho": 0.30,        # unobservable residual risk loading
        "risk_ctx": 0.90,        # OBSERVABLE-context loading: safety depends on context
        "risk_center": 1.6,
        "risk_scale": 0.45,
        "peak_gain": 3.0,        # utility bump height at the context-located peak
        "peak_width": 0.10,
        "peak_base": 0.15,       # a* = peak_base + peak_slope * clip(C,-1,1)
        "peak_slope": 0.20,
    }

    def reset(self) -> np.ndarray:
        self.t = 0
        self.s = 0
        self.U = float(self.rng.normal())
        self.C = float(self.rng.normal())
        self.R = float(self.rng.normal() * 0.5)
        self.a_prev = 0.0
        self.x1 = self.x2 = self.y = 0.0
        return self._obs()

    def _peak_action(self) -> float:
        d = self.DEC
        return d["peak_base"] + d["peak_slope"] * float(np.clip(self.C, -1.0, 1.0))

    def _decoupled_harm(self, a: float) -> float:
        d = self.DEC
        gain = d["risk_gain_post"] if self.s == 1 else d["risk_gain_pre"]
        side = gain * float(a) + d["risk_rho"] * self.R + d["risk_ctx"] * self.C
        side += float(self.rng.normal(0, 0.04))
        return float(1.0 / (1.0 + np.exp(-(side - d["risk_center"]) / d["risk_scale"])))

    def get_state(self) -> dict:
        return {
            "t": self.t,
            "s": self.s,
            "U": self.U,
            "C": self.C,
            "R": getattr(self, "R", 0.0),
            "a_prev": self.a_prev,
            "x1": self.x1,
            "x2": self.x2,
            "y": self.y,
            "rng_state": self.rng.get_state(),
        }

    def set_state(self, state: dict) -> None:
        self.t = state["t"]
        self.s = state["s"]
        self.U = state["U"]
        self.C = state["C"]
        self.R = state.get("R", 0.0)
        self.a_prev = state["a_prev"]
        self.x1 = state["x1"]
        self.x2 = state["x2"]
        self.y = state["y"]
        self.rng.set_state(state["rng_state"])

    def clone(self) -> "NonstationarySCMEnv":
        env = NonstationarySCMEnv(
            self.shift_at, self.shift_mag, self.obs_noise, self.seed, harm_mode=self.harm_mode
        )
        env.set_state(copy.deepcopy(self.get_state()))
        return env

    def _coeffs(self) -> dict:
        b = {
            "a_x1": 0.85,
            "aprev_x1": 0.3,
            "u_x1": 0.55,
            "c_x1": 0.35,
            "a_x2": -0.45,
            "aprev_x2": -0.15,
            "x1_x2": 0.28,
            "u_x2": 0.45,
            "c_x2": 0.22,
            "x1_y": 1.15,
            "x2_y": -0.78,
            "a_y": 0.35,
            "aprev_y": 0.15,
            "u_y": 0.65,
            "c_y": 0.18,
        }
        if self.t >= self.shift_at:
            self.s = 1
            b["a_y"] += self.shift_mag
            b["u_y"] += 0.2
        return b

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self.C + self.rng.normal(0, self.obs_noise),
                self.x1 + self.rng.normal(0, self.obs_noise),
                self.x2 + self.rng.normal(0, self.obs_noise),
                self.y + self.rng.normal(0, self.obs_noise),
            ],
            dtype=float,
        )

    def step(self, a: float):
        b = self._coeffs()
        self.U = 0.85 * self.U + 0.15 * self.rng.normal()
        self.C = 0.75 * self.C + 0.25 * self.rng.normal()
        if not hasattr(self, "R"):
            self.R = 0.0
        self.R = 0.80 * self.R + 0.20 * self.rng.normal()

        x1 = (
            b["a_x1"] * a
            + b["aprev_x1"] * self.a_prev
            + b["u_x1"] * self.U
            + b["c_x1"] * self.C
            + self.rng.normal(0, 0.6)
        )
        x2 = (
            b["a_x2"] * a
            + b["aprev_x2"] * self.a_prev
            + b["x1_x2"] * x1
            + b["u_x2"] * self.U
            + b["c_x2"] * self.C
            + self.rng.normal(0, 0.6)
        )
        y = (
            b["x1_y"] * x1
            + b["x2_y"] * x2
            + b["a_y"] * a
            + b["aprev_y"] * self.a_prev
            + b["u_y"] * self.U
            + b["c_y"] * self.C
            + self.rng.normal(0, 0.6)
        )

        if self.harm_mode == "decoupled":
            d = self.DEC
            a_star = self._peak_action()
            y = y + d["peak_gain"] * float(
                np.exp(-((float(a) - a_star) ** 2) / (2.0 * d["peak_width"] ** 2))
            )

        self.a_prev = float(a)
        self.x1, self.x2, self.y = float(x1), float(x2), float(y)
        self.t += 1

        U = float(y)
        # Two harm channels:
        # - aligned: harm from negative utility (legacy)
        # - tradeoff: higher intensity a is riskier, especially after shift
        aligned = max(0.0, -y) / (1.0 + abs(y))
        side = 1.00 * float(a) + 0.10 * max(self.C, 0.0)
        if self.s == 1:
            side += 0.95 * float(a)
        side += float(self.rng.normal(0, 0.04))
        tradeoff = 1.0 / (1.0 + np.exp(-(side - 2.85)))
        mode = getattr(self, "harm_mode", "tradeoff")
        if mode == "aligned":
            H = float(aligned)
        elif mode == "decoupled":
            H = float(np.clip(self._decoupled_harm(a), 0.0, 1.0))
        else:
            H = float(0.15 * aligned + 0.85 * tradeoff)
        return self._obs(), U, H, False, {"shift": self.s, "aligned_h": aligned}

    def oracle_action_stats(self, actions: np.ndarray, n_mc: int, tau: float):
        """Monte-Carlo E[U|do(a)] and P(H>tau|do(a)] from the current latent state."""
        base = copy.deepcopy(self.get_state())
        mu_u = np.zeros(len(actions))
        p_viol = np.zeros(len(actions))
        for i, a in enumerate(actions):
            us, vs = [], []
            for k in range(n_mc):
                self.set_state(copy.deepcopy(base))
                self.rng = np.random.RandomState(10_000 + k + 17 * self.t)
                _, u, h, _, _ = self.step(float(a))
                us.append(u)
                vs.append(float(h > tau))
            mu_u[i] = np.mean(us)
            p_viol[i] = np.mean(vs)
        self.set_state(base)
        return mu_u, p_viol
