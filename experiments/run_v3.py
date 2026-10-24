"""v3 evaluation harness.

Domains: scm (decoupled nonstationary SCM), lending (German Credit, real
default labels), ihdp (IHDP with a binding harm channel; needs the IHDP npz
files, which this script downloads on first use).

Usage
-----
  python experiments/run_v3.py main     --domain scm      --seeds 100:130
  python experiments/run_v3.py frontier --domain lending  --seeds 100:115
  python experiments/run_v3.py tune     --domain scm      --seeds 0:10
Every run writes episodes.csv; `summarize` turns them into JSON + LaTeX.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from neurosafe_cbo.agent import make_agent  # noqa: E402
from neurosafe_cbo.config import CFG  # noqa: E402
from neurosafe_cbo.v3 import ConservativeUCB, NeuroSafeCtxV3, SafeGPUCB  # noqa: E402

ACTS = np.linspace(0.0, 1.0, 11)
OUT = os.path.join(ROOT, "results", "v3")

DOMAINS = {
    "scm": dict(tau=0.20, H=200),
    "lending": dict(tau=0.25, H=200),
    "ihdp": dict(tau=0.50, H=200),
}

# Frozen after tuning on seeds 0-9 (see `tune`); evaluation uses seeds >= 100.
OURS = dict(beta_h=1.0, beta_u=0.5, gamma=0.995, n_buckets=4, shrink=3.0, basis="cubic8")


def make_env(domain, seed, H):
    if domain == "scm":
        from neurosafe_cbo.env import NonstationarySCMEnv
        return NonstationarySCMEnv(H // 2, 0.35, 0.15, seed=seed, harm_mode="decoupled")
    if domain == "lending":
        from neurosafe_cbo.env_lending_bandit import LendingSafeBanditEnv
        return LendingSafeBanditEnv(seed=seed, horizon=H, tau_harm=DOMAINS["lending"]["tau"])
    if domain == "ihdp":
        from neurosafe_cbo.env_ihdp_v3 import IHDPBindingEnv
        return IHDPBindingEnv(seed=seed, horizon=H)
    if domain == "mimic":
        from neurosafe_cbo.env_mimic_v3 import MimicBandit
        return MimicBandit(seed=seed, horizon=H)
    raise KeyError(domain)


def _legacy(name, seed, tau, alpha, H):
    ACTS = _domain_acts()
    cfg = dict(CFG)
    cfg.update({"TAU_HARM": tau, "ALPHA": alpha, "NS_ALPHA": alpha, "HORIZON": H,
                "SHIFT_AT": H // 2})
    return make_agent(name, seed=seed, cfg=cfg, actions=ACTS)


def make(name, seed, tau, knob, H, ours=None):
    ACTS = _domain_acts()
    o = dict(OURS if ours is None else ours)
    if name == "NeuroSafe-Ctx":
        return NeuroSafeCtxV3(ACTS, seed, tau, alpha=knob, **o)
    if name == "Ctx-noGLM":
        return NeuroSafeCtxV3(ACTS, seed, tau, alpha=knob, harm="gauss", **o)
    if name == "Ctx-BetaCell":
        from neurosafe_cbo.beta_cell import BetaCellHarm
        ag = NeuroSafeCtxV3(ACTS, seed, tau, alpha=knob, harm="gauss", **o)
        ag.gauss = BetaCellHarm(ag.K, tau)
        return ag
    if name == "Ctx-noBucket":
        return NeuroSafeCtxV3(ACTS, seed, tau, alpha=knob, utility="pooled", **o)
    if name == "Ctx-noKalman":
        return NeuroSafeCtxV3(ACTS, seed, tau, alpha=knob, context="raw", **o)
    if name == "Ctx-noContext":
        return NeuroSafeCtxV3(ACTS, seed, tau, alpha=knob, context="none", **o)
    if name == "SafeGP":
        return SafeGPUCB(ACTS, seed, tau, beta=knob)
    if name == "SafeOpt-ref":
        from neurosafe_cbo.safeopt_ref import SafeOptRef
        cfg = safeopt_config(DOMAIN_OF_RUN[0])
        return SafeOptRef(ACTS, seed, tau, beta=knob, **cfg)
    if name == "Conservative":
        return ConservativeUCB(ACTS, seed, eta=knob)
    if name in ("Chance", "NeuroSafeCBO", "RiskNeutral"):
        return _legacy(name, seed, tau, knob, H)
    raise KeyError(name)


FRONTIER_KNOBS = {"NeuroSafe-Ctx": [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35],
                  "Chance": [0.02, 0.05, 0.10, 0.15, 0.25, 0.35],
                  "SafeGP": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                  "Conservative": [0.0, 0.02, 0.05, 0.1, 0.3],
                  "SafeOpt-ref": [1.0, 2.0, 3.0, 4.0]}

DOMAINS["mimic"] = dict(tau=0.5, H=400)


def _domain_acts():
    """MIMIC has 5 dose bins; every other domain keeps the shared 11-point grid."""
    return np.linspace(0.0, 1.0, 5) if DOMAIN_OF_RUN[0] == "mimic" else ACTS

DEFAULT_KNOB = {"NeuroSafe-Ctx": 0.15, "Ctx-noGLM": 0.15, "Ctx-noBucket": 0.15,
                "Ctx-noKalman": 0.15, "Chance": 0.15, "NeuroSafeCBO": 0.15,
                "RiskNeutral": 0.15, "SafeGP": 2.0, "Conservative": 0.1,
                "SafeOpt-ref": 2.0}


def lending_oracle(env, tau):
    x = env.X[env._cur_idx]
    q = np.column_stack([np.full(len(ACTS), x[0]), np.full(len(ACTS), x[1]), ACTS])
    p = env.model.predict_proba(q)[:, 1]
    p[0] = 0.0  # a = 0 declines the loan
    ok = p <= tau - 0.0104  # P(p + N(0, .01) > tau) <= 0.15
    return float(ACTS[np.where(ok)[0].max()]) if ok.any() else float(ACTS[int(np.argmin(p))])


DOMAIN_OF_RUN = ["scm"]
_CTX_SCALE = {}


def context_scale(domain):
    """SD of the observed context channel on a separate env seed (99999)."""
    if domain not in _CTX_SCALE:
        env = make_env(domain, 99999, DOMAINS[domain]["H"])
        obs = env.reset(); cs = []
        for _ in range(200):
            cs.append(float(obs[0])); obs, *_ = env.step(0.0)
        _CTX_SCALE[domain] = max(float(np.std(cs)), 1e-3)
    return _CTX_SCALE[domain]


def safeopt_config(domain):
    """Hyperparameters chosen on tuning seeds by run_safeopt_ref.py tune (defaults otherwise)."""
    path = os.path.join(OUT, domain, "safeopt_ref_config.json")
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    cfg = {k: v for k, v in cfg.items() if k in ("ls_a", "ls_z", "u_var", "u_noise", "h_var", "h_noise")}
    cfg["ctx_scale"] = context_scale(domain)
    return cfg


def episode(domain, name, seed, knob, ours=None):
    DOMAIN_OF_RUN[0] = domain
    d = DOMAINS[domain]
    tau, H = d["tau"], d["H"]
    env = make_env(domain, seed, H)
    obs = env.reset()
    ag = None if name == "Oracle" else make(name, seed, tau, knob, H, ours)
    U, V = [], []
    for _ in range(H):
        if name == "Oracle":
            a = lending_oracle(env, tau) if domain == "lending" else env.oracle(0.15)
            z = 0.0
        else:
            a, z = ag.act(obs)
        obs, u, h, _, _ = env.step(a)
        if ag is not None:
            ag.observe(a, z, u, h)
        U.append(u)
        V.append(h > tau)
    extra = {}
    if isinstance(ag, NeuroSafeCtxV3) and ag.stats["decisions"]:
        extra["empty_frac"] = ag.stats["empty_feasible"] / ag.stats["decisions"]
    return dict(domain=domain, agent=name, knob=knob, seed=seed,
                ret=float(np.mean(U)), viol=float(np.mean(V)), **extra)


def run_many(domain, agents, seeds, knobs=None, tag="main", ours=None):
    rows = []
    t0 = time.time()
    for name in agents:
        for kn in (knobs or {}).get(name, [DEFAULT_KNOB.get(name, 0.15)]):
            for s in seeds:
                rows.append(episode(domain, name, int(s), kn, ours))
        print(f"  {domain}/{name} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, domain), exist_ok=True)
    path = os.path.join(OUT, domain, f"{tag}_episodes.csv")
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df = df.drop_duplicates(["agent", "knob", "seed"], keep="last")
    df.to_csv(path, index=False)
    return df


def boot_ci(x, B=5000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(df, new, base, knob_new=None, knob_base=None):
    a = df[(df.agent == new) & ((df.knob == knob_new) if knob_new is not None else True)]
    b = df[(df.agent == base) & ((df.knob == knob_base) if knob_base is not None else True)]
    a = a.set_index("seed")
    b = b.set_index("seed")
    common = a.index.intersection(b.index)
    dR = (a.loc[common, "ret"] - b.loc[common, "ret"]).to_numpy()
    dV = (a.loc[common, "viol"] - b.loc[common, "viol"]).to_numpy()

    def wil(x, alt):
        try:
            return float(stats.wilcoxon(x, alternative=alt).pvalue)
        except ValueError:
            return float("nan")

    mR, lR, hR = boot_ci(dR)
    mV, lV, hV = boot_ci(dV)
    return dict(n=int(len(common)),
                base_ret=float(b.loc[common, "ret"].mean()), new_ret=float(a.loc[common, "ret"].mean()),
                base_viol=float(b.loc[common, "viol"].mean()), new_viol=float(a.loc[common, "viol"].mean()),
                dR=mR, dR_ci=[lR, hR], p_dR_two=wil(dR, "two-sided"),
                dV=mV, dV_ci=[lV, hV], p_dV_two=wil(dV, "two-sided"),
                win_R=int(np.sum(dR > 0)), win_V=int(np.sum(dV < 0)))


def parse_seeds(s):
    a, b = s.split(":")
    return range(int(a), int(b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["main", "frontier", "ablation", "time"])
    ap.add_argument("--domain", default="scm")
    ap.add_argument("--seeds", default="100:130")
    ap.add_argument("--agents", default="")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    if args.mode == "main":
        ag = args.agents.split(",") if args.agents else [
            "NeuroSafe-Ctx", "Chance", "NeuroSafeCBO", "SafeGP", "Conservative", "RiskNeutral"]
        if args.domain in ("lending", "ihdp") and not args.agents:
            ag.append("Oracle")
        run_many(args.domain, ag, seeds, tag="main")
    elif args.mode == "ablation":
        run_many(args.domain, ["Ctx-noGLM", "Ctx-noBucket", "Ctx-noKalman"], seeds, tag="main")
    elif args.mode == "frontier":
        knobs = FRONTIER_KNOBS
        ag = args.agents.split(",") if args.agents else list(knobs)
        run_many(args.domain, ag, seeds, knobs={k: knobs[k] for k in ag}, tag="frontier")
    elif args.mode == "time":
        for n in ["NeuroSafe-Ctx", "Chance", "SafeGP", "Conservative", "NeuroSafeCBO"]:
            t = time.time()
            r = episode(args.domain, n, 0, DEFAULT_KNOB[n])
            print(n, f"{time.time()-t:.2f}s", r)

