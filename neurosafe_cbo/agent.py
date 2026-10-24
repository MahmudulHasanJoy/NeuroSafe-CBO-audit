"""NeuroSafe-CBO agent and named baseline / ablation variants."""
from __future__ import annotations

import math

import numpy as np

from .config import ACTIONS, CFG, K_CVAR
from .models import (ActionStats, BayesLin, BucketedActionStats, PredictiveCoder, RFF,
                     make_context, norm_cdf)


class NeuroSafeCBO:
    """Optimal-among-safe causal controller (Chance-anchored dual path)."""

    def __init__(
        self,
        actions=None,
        cfg=None,
        *,
        use_pc: bool = True,
        use_neuromod: bool = True,
        use_chance: bool = True,
        use_cvar: bool = True,
        seed: int = 0,
    ):
        self.actions = np.asarray(actions if actions is not None else ACTIONS, dtype=float)
        self.cfg = dict(cfg if cfg is not None else CFG)
        self.use_pc = use_pc
        self.use_neuromod = use_neuromod
        self.use_chance = use_chance
        self.use_cvar = use_cvar
        self.full_stack = bool(
            use_chance and (use_pc or bool(self.cfg.get("ANCHOR_WITHOUT_PC", False)))
        )
        self.pc = make_context(self.cfg, seed) if use_pc else None
        self.rff = RFF(
            int(self.cfg["RFF_DIM"]),
            float(self.cfg.get("RFF_SIGMA_U", self.cfg["RFF_SIGMA"])),
            seed,
            sigma_z=self.cfg.get("RFF_SIGMA_Z"),
        )
        self.rff_h = RFF(
            int(self.cfg["RFF_DIM"]),
            float(self.cfg.get("RFF_SIGMA_H", 0.35)),
            seed + 7919,
            sigma_z=self.cfg.get("RFF_SIGMA_HZ", 1.0),
        )
        self.muU = BayesLin(
            int(self.cfg["RFF_DIM"]),
            max(float(self.cfg["OBS_NOISE_U"]), 0.75),
            float(self.cfg["FORGET"]),
        )
        self.muH = BayesLin(
            int(self.cfg["RFF_DIM"]),
            max(float(self.cfg["OBS_NOISE_H"]), 0.05),
            float(self.cfg["FORGET"]),
        )
        self.u_stats = ActionStats(
            len(self.actions),
            prior_mu=0.0,
            prior_var=1.0,
            prior_n=2.0,
            forget=float(self.cfg["FORGET"]),
            obs_var=0.25,
        )
        self.h_stats = ActionStats(
            len(self.actions),
            prior_mu=0.08,
            prior_var=0.02,
            prior_n=3.0,
            forget=float(self.cfg["FORGET"]),
            obs_var=0.01,
        )
        if bool(self.cfg.get("CTX_HARM_BUCKETS", False)):
            self.h_ctx = BucketedActionStats(
                len(self.actions),
                self.cfg.get("CTX_EDGES", (-0.6, 0.0, 0.6)),
                prior_mu=0.08, prior_var=0.02, prior_n=3.0,
                forget=float(self.cfg["FORGET"]), obs_var=0.01,
            )
        else:
            self.h_ctx = None
        self.t = 0
        self.harm_ema = 0.0
        self.rng = np.random.RandomState(seed)
        # --- anchored-override diagnostics (non-invasive; off the hot path) ---
        self.diag = {
            "decisions": 0,          # post-warmup decisions on the full-stack path
            "feasible_gt1": 0,       # steps where >1 action is alpha-feasible
            "pool_nonempty": 0,      # steps where the override pool is non-empty
            "override_fired": 0,     # steps where the played action != Chance's pick
            "feasible_size_sum": 0,  # to report mean feasible-set size
            "pool_size_sum": 0,
            "util_gap_sum": 0.0,     # utility gap the override captured (played - chance)
        }

    def _action_index(self, a: float) -> int:
        return int(np.argmin(np.abs(self.actions - float(a))))

    def _calibrated_harm(self) -> bool:
        return bool(self.cfg.get("DECOUPLE_HARM", True))

    def _predict(self, z_u: float, z_h: float = 0.0):
        muU_a, sU_a = self.u_stats.predict()
        if self._calibrated_harm():
            muH, sH = self.h_stats.predict()
        else:
            xz_h = np.stack([self.actions, np.full(len(self.actions), z_h)], axis=1)
            muH, sH = self.muH.predict(self.rff.phi(xz_h))
            muH = np.asarray(muH, dtype=float)
            sH = np.asarray(sH, dtype=float)
        if abs(float(z_u)) < 1e-12:
            return muU_a, sU_a, muH, sH
        xz = np.stack([self.actions, np.full(len(self.actions), z_u)], axis=1)
        mu_r, s_r = self.muU.predict(self.rff.phi(xz))
        xz0 = np.stack([self.actions, np.zeros(len(self.actions))], axis=1)
        mu0, _ = self.muU.predict(self.rff.phi(xz0))
        ctx_gain = float(self.cfg.get("CTX_GAIN", 0.35))
        muU = muU_a + ctx_gain * (np.asarray(mu_r) - np.asarray(mu0))
        sU = np.sqrt(np.maximum(sU_a ** 2 + 0.15 * (np.asarray(s_r) ** 2), 1e-8))
        return muU, sU, muH, sH

    def _chance_index(self, muU, sU, muH, sH):
        beta = float(self.cfg["BASE_BETA"])
        score = muU + beta * sU
        alpha = float(self.cfg["ALPHA"])
        p_safe = norm_cdf((self.cfg["TAU_HARM"] - muH) / np.maximum(sH, 1e-6))
        feasible = p_safe >= (1.0 - alpha)
        masked = np.asarray(score, dtype=float).copy()
        masked[~feasible] = -np.inf
        if not np.any(np.isfinite(masked)):
            return int(np.argmin(muH)), p_safe
        return int(np.argmax(masked)), p_safe

    def act(self, obs: np.ndarray):
        self.t += 1
        c, _x1, _x2, y = [float(v) for v in obs]
        z = self.pc.infer(c, y, self.harm_ema) if self.pc is not None else 0.0

        if self.t <= int(self.cfg["WARMUP"]):
            return float(self.rng.choice(self.actions)), z

        if not self.full_stack:
            muU, sU, muH, sH = self._predict(z, z)
            if (not self.use_chance) and (not self.use_cvar):
                beta = float(self.cfg["BASE_BETA"])
                return float(self.actions[int(np.argmax(muU + beta * sU))]), z
            if self.use_chance and (not self.use_cvar):
                idx, _ = self._chance_index(muU, sU, muH, sH)
                return float(self.actions[idx]), z
            beta = float(self.cfg["BASE_BETA"])
            score = muU + beta * sU - float(self.cfg.get("LAMBDA_CVAR", 0.0)) * (
                muH + sH * K_CVAR
            )
            if self.use_neuromod:
                diff = float(self.cfg["TAU_HARM"]) - float(np.min(muH))
                score = score + float(self.cfg.get("NEURO_GAIN", 0.0)) / (
                    1.0 + math.exp(-float(self.cfg.get("NEURO_K", 8.0)) * diff)
                )
            return float(self.actions[int(np.argmax(score))]), z

        muU0, sU0, muH0, sH0 = self._predict(0.0, 0.0)
        idx_c, p_safe0 = self._chance_index(muU0, sU0, muH0, sH0)
        safe_floor = float(p_safe0[idx_c]) + float(self.cfg.get("SAFE_FLOOR_BONUS", 0.0))
        alpha = float(self.cfg.get("NS_ALPHA", self.cfg["ALPHA"]))
        feasible = p_safe0 >= (1.0 - alpha)
        pool = feasible & (p_safe0 >= safe_floor - 1e-12)

        muU_pc, _, _, _ = self._predict(z, 0.0)
        pc_mix = float(self.cfg.get("PC_UTIL_MIX", 0.35))
        util = (1.0 - pc_mix) * muU0 + pc_mix * muU_pc
        util_floor = float(util[idx_c]) + float(self.cfg.get("DOMINANCE_UTIL_EPS", 0.01))
        pool = pool & (util >= util_floor - 1e-12)
        if not np.any(pool):
            idx = idx_c
        else:
            order = np.lexsort((-p_safe0, -util))
            idx = idx_c
            for j in order:
                if pool[j]:
                    idx = int(j)
                    break

        d = self.diag
        d["decisions"] += 1
        n_feas = int(np.sum(feasible))
        n_pool = int(np.sum(pool))
        d["feasible_size_sum"] += n_feas
        d["pool_size_sum"] += n_pool
        if n_feas > 1:
            d["feasible_gt1"] += 1
        if n_pool > 0:
            d["pool_nonempty"] += 1
        if idx != idx_c:
            d["override_fired"] += 1
            d["util_gap_sum"] += float(util[idx] - util[idx_c])
        return float(self.actions[idx]), z

    def observe(self, a: float, z: float, U: float, H: float) -> None:
        idx = self._action_index(a)
        self.u_stats.update(idx, U)
        self.h_stats.update(idx, H)
        if getattr(self, "h_ctx", None) is not None:
            self.h_ctx.update(idx, H, float(z))
        self.muU.update(self.rff.phi(np.array([a, float(z)], dtype=float)), U)
        self.muH.update(self.rff_h.phi(np.array([a, float(z)], dtype=float)), H)
        # The published code ALSO trained the same model at (a, 0) with the same
        # target, which teaches it utility is context-independent and cancels the
        # context signal. Kept default-on for reproducibility; disable to let the
        # context-conditional model actually learn f(a, z).
        if bool(self.cfg.get("CTX_DUAL_UPDATE", True)):
            self.muU.update(self.rff.phi(np.array([a, 0.0], dtype=float)), U)
            self.muH.update(self.rff_h.phi(np.array([a, 0.0], dtype=float)), H)
        self.harm_ema = 0.95 * self.harm_ema + 0.05 * float(H)


class NeuroSafeCtx(NeuroSafeCBO):
    """Context-conditional safe-optimal rule (the rule the paper actually states).

    Differences from the published `NeuroSafeCBO`:
      1. Context comes from a correctly-specified predictive filter (CTX_MODE),
         not the untrained bottleneck whose latent is ~uncorrelated with context.
      2. Utility is predicted CONDITIONAL on context, rather than blended 65/35
         with a context-free estimate.
      3. No Chance anchoring. The published anchor (play only actions at least as
         safe as Chance's pick AND >= its utility + eps) makes the pool empty ~97%
         of the time, which is why the published agent is statistically identical
         to its own baseline. The feasibility constraint is still hard -- safety
         is enforced by the chance constraint itself, not by copying Chance.
    """

    def act(self, obs: np.ndarray):
        self.t += 1
        c, _x1, _x2, y = [float(v) for v in obs]
        z = self.pc.infer(c, y, self.harm_ema) if self.pc is not None else 0.0

        if self.t <= int(self.cfg["WARMUP"]):
            return float(self.rng.choice(self.actions)), z

        # --- harm / feasibility
        if self.h_ctx is not None:
            muH, sH = self.h_ctx.predict(z)
        elif bool(self.cfg.get("CTX_HARM", False)):
            # context-conditional harm: feasible set is evaluated AT THE CURRENT
            # context, instead of marginalising over it like a context-blind agent.
            xz_h = np.stack([self.actions, np.full(len(self.actions), z)], axis=1)
            mh, sh = self.muH.predict(self.rff_h.phi(xz_h))
            muH = np.asarray(mh, dtype=float)
            sH = np.maximum(np.asarray(sh, dtype=float), 1e-3)
            mh0, _ = self.h_stats.predict()
            bw = float(self.cfg.get("CTX_HARM_BLEND", 0.5))
            muH = (1.0 - bw) * mh0 + bw * muH
        else:
            muH, sH = self.h_stats.predict()
        alpha = float(self.cfg.get("NS_ALPHA", self.cfg["ALPHA"]))
        p_safe = norm_cdf((float(self.cfg["TAU_HARM"]) - muH) / np.maximum(sH, 1e-6))
        feasible = p_safe >= (1.0 - alpha)

        # --- context-conditional utility
        muU_a, sU_a = self.u_stats.predict()
        xz = np.stack([self.actions, np.full(len(self.actions), z)], axis=1)
        mu_r, s_r = self.muU.predict(self.rff.phi(xz))
        w = float(self.cfg.get("CTX_WEIGHT", 0.7))
        util = (1.0 - w) * muU_a + w * np.asarray(mu_r, dtype=float)
        sU = np.sqrt(np.maximum((1.0 - w) * sU_a ** 2 + w * np.asarray(s_r) ** 2, 1e-8))
        score = util + float(self.cfg["BASE_BETA"]) * sU

        d = self.diag
        d["decisions"] += 1
        n_feas = int(np.sum(feasible))
        d["feasible_size_sum"] += n_feas
        d["pool_size_sum"] += n_feas
        if n_feas > 1:
            d["feasible_gt1"] += 1
        if n_feas > 0:
            d["pool_nonempty"] += 1

        if not np.any(feasible):
            return float(self.actions[int(np.argmin(muH))]), z
        masked = score.copy()
        masked[~feasible] = -np.inf
        idx = int(np.argmax(masked))

        # diagnostic: does context change the pick vs the context-free rule?
        masked0 = (muU_a + float(self.cfg["BASE_BETA"]) * sU_a).copy()
        masked0[~feasible] = -np.inf
        idx0 = int(np.argmax(masked0))
        if idx != idx0:
            d["override_fired"] += 1
            d["util_gap_sum"] += float(util[idx] - util[idx0])
        return float(self.actions[idx]), z


class RandomAgent:
    def __init__(self, actions=None, seed: int = 0, cfg=None, **_kwargs):
        self.actions = np.asarray(actions if actions is not None else ACTIONS, dtype=float)
        self.rng = np.random.RandomState(seed)

    def act(self, obs):
        return float(self.rng.choice(self.actions)), 0.0

    def observe(self, *args, **kwargs):
        return None


class GreedySafeAgent:
    def __init__(self, actions=None, seed: int = 0, cfg=None, **_kwargs):
        self.actions = np.asarray(actions if actions is not None else ACTIONS, dtype=float)

    def act(self, obs):
        y = float(obs[-1])
        a = self.actions[0] if y < -0.2 else self.actions[-1]
        return float(a), 0.0

    def observe(self, *args, **kwargs):
        return None


AGENT_SPECS = {
    "Random": dict(cls=RandomAgent),
    "Greedy": dict(cls=GreedySafeAgent),
    "RiskNeutral": dict(
        cls=NeuroSafeCBO, use_pc=False, use_neuromod=False, use_chance=False, use_cvar=False
    ),
    "Chance": dict(
        cls=NeuroSafeCBO, use_pc=False, use_neuromod=False, use_chance=True, use_cvar=False
    ),
    "CVaR": dict(
        cls=NeuroSafeCBO, use_pc=False, use_neuromod=False, use_chance=False, use_cvar=True
    ),
    "NeuroSafeCtx": dict(
        cls=NeuroSafeCtx, use_pc=True, use_neuromod=False, use_chance=True, use_cvar=True
    ),
    "NeuroSafeCBO": dict(
        cls=NeuroSafeCBO, use_pc=True, use_neuromod=True, use_chance=True, use_cvar=True
    ),
    "NS_noPC": dict(
        cls=NeuroSafeCBO, use_pc=False, use_neuromod=True, use_chance=True, use_cvar=True
    ),
    "NS_noNeuro": dict(
        cls=NeuroSafeCBO, use_pc=True, use_neuromod=False, use_chance=True, use_cvar=True
    ),
    "NS_noChance": dict(
        cls=NeuroSafeCBO, use_pc=True, use_neuromod=True, use_chance=False, use_cvar=True
    ),
    "NS_noCVaR": dict(
        cls=NeuroSafeCBO, use_pc=True, use_neuromod=True, use_chance=True, use_cvar=False
    ),
}


def make_agent(name: str, seed: int = 0, cfg=None, actions=None):
    if name not in AGENT_SPECS:
        raise KeyError(f"Unknown agent: {name}")
    spec = dict(AGENT_SPECS[name])
    cls = spec.pop("cls")
    return cls(actions=actions, cfg=cfg, seed=seed, **spec)
