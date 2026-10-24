"""Frozen experiment and algorithm configuration for NeuroSafe-CBO."""
from __future__ import annotations

import math

import numpy as np

SEED = 42

CFG = {
    "SEEDS": 10,
    "EPISODES": 20,
    "HORIZON": 40,
    "N_ACTIONS": 11,
    "SHIFT_AT": 20,
    "SHIFT_MAGS": [0.15, 0.35],
    "OBS_NOISES": [0.15, 0.30],
    "TAU_HARM": 0.20,
    "ALPHA": 0.15,
    "NS_ALPHA": 0.15,
    "BETA_CVAR": 0.90,
    "LAMBDA_CVAR": 0.0,
    "BASE_BETA": 0.8,
    "NEURO_K": 8.0,
    "NEURO_GAIN": 0.0,
    "PC_STEPS": 5,
    "PC_LR_Z": 0.08,
    "PC_LR_W": 0.02,
    "RFF_DIM": 128,
    "RFF_SIGMA": 0.7,
    "OBS_NOISE_U": 0.18,
    "OBS_NOISE_H": 0.10,
    "FORGET": 0.995,
    "WARMUP": 10,
    "ORACLE_MC": 12,
    "REGRET_EVERY": 10,
    "SAFE_FLOOR_BONUS": 0.0,
    "DOMINANCE_UTIL_EPS": 0.05,
    "DOMINANCE_SAFE_EPS": 0.0,
    "PC_UTIL_MIX": 0.35,
    "DECOUPLE_HARM": True,
    "SCORE_MODE": "mean",
    "ANCHOR_WITHOUT_PC": False,
}

ACTIONS = np.linspace(0.0, 1.0, CFG["N_ACTIONS"])


def gaussian_cvar_k(beta: float = CFG["BETA_CVAR"]) -> float:
    from scipy.stats import norm

    z = float(norm.ppf(beta))
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / (1.0 - beta)


K_CVAR = gaussian_cvar_k()
