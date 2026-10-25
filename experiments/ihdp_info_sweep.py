"""IHDP mechanism test: vary how much the observed context reveals about harm (rho = corr of the
observed context with the risk signal that drives harm) by changing observation noise. If the paper's
claim is right, NeuroSafe-Ctx's advantage over the context-blind Chance rule shrinks as rho falls.
Held-out seeds 100-149, same agents and settings as the main results. Writes results/v3/ihdp_info_sweep.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import experiments.run_v3 as rv
from neurosafe_cbo.env_ihdp_v3 import IHDPBindingEnv

SD_S, D = 0.0871, 25                      # sd of the risk signal across patients; covariates averaged
TARGETS = [0.9935, 0.95, 0.90, 0.70, 0.50, 0.30]   # 0.9935 = the paper's setting (obs_noise 0.05)
noise_for = lambda r: 0.05 if r > 0.99 else float(np.sqrt(D) * SD_S * np.sqrt(1 / r**2 - 1))
_orig = rv.make_env
def run_at(rho, agent, knob, seed):
    rv.make_env = lambda dom, s, H: IHDPBindingEnv(seed=s, horizon=H, obs_noise=noise_for(rho)) if dom == "ihdp" else _orig(dom, s, H)
    r = rv.episode("ihdp", agent, seed, knob); r["rho"] = rho; return r

def boot(x, B=2000, seed=0):
    rng = np.random.RandomState(seed); m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

rows = [run_at(r, a, k, s) for r in TARGETS
        for a, ks in [("NeuroSafe-Ctx", rv.FRONTIER_KNOBS["NeuroSafe-Ctx"]), ("Chance", rv.FRONTIER_KNOBS["Chance"])]
        for k in ks for s in range(100, 150)]
df = pd.DataFrame(rows); os.makedirs("results/v3", exist_ok=True); df.to_csv("results/v3/ihdp_info_sweep.csv", index=False)

print(" rho  | headline pair (ours a=0.15 vs Chance a=0.02): dR [95% CI]   dV [95% CI]      | frontier: ours above / comparable (mean gap) | ours min V")
for r in TARGETS:
    d = df[df.rho == r]
    a = d[(d.agent == "NeuroSafe-Ctx") & np.isclose(d.knob, 0.15)].set_index("seed")
    b = d[(d.agent == "Chance") & np.isclose(d.knob, 0.02)].set_index("seed")
    mR, lR, hR = boot((a.ret - b.ret).to_numpy()); mV, lV, hV = boot((a.viol - b.viol).to_numpy())
    f = d[d.agent == "NeuroSafe-Ctx"].groupby("knob")[["ret", "viol"]].mean().sort_values("viol")
    c = d[d.agent == "Chance"].groupby("knob")[["ret", "viol"]].mean()
    gaps = [float(np.interp(v, f.viol, f.ret) - rr) for rr, v in zip(c.ret, c.viol) if f.viol.min() <= v <= f.viol.max()]
    above = sum(g > 0 for g in gaps)
    gap_txt = f"{above}/{len(gaps)} ({np.mean(gaps):+.3f})" if gaps else "none comparable"
    print(f"{r:5.3f} | {mR:+.3f} [{lR:+.3f},{hR:+.3f}]  {mV:+.4f} [{lV:+.4f},{hV:+.4f}] | {gap_txt:>28s} | {f.viol.min():.4f}")
