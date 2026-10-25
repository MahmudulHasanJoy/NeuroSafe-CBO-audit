"""Why does NeuroSafe-Ctx exceed small alpha budgets? Decomposes realised violation into the
10 random warm-up rounds and the rest, and compares with (i) always playing the safest action a=0,
(ii) the context-seeing oracle (IHDP), and (iii) a variant whose warm-up plays the safest action.
Held-out seeds 100-149. Writes results/v3/small_alpha_check.csv
Usage: python experiments/small_alpha_check.py ihdp scm"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import experiments.run_v3 as rv
from neurosafe_cbo.v3 import NeuroSafeCtxV3

class SafeWarmup(NeuroSafeCtxV3):
    """Identical, except the warm-up rounds play the safest action (a = 0) instead of random actions."""
    def act(self, obs):
        if self.t < self.warmup:
            self.t += 1
            z = self.ctx.infer(float(obs[0])); self.std.update(z)
            return float(self.actions[0]), z
        return super().act(obs)

def run(domain, kind, seed, alpha):
    rv.DOMAIN_OF_RUN[0] = domain
    tau, H = rv.DOMAINS[domain]["tau"], rv.DOMAINS[domain]["H"]
    env = rv.make_env(domain, seed, H); obs = env.reset()
    ag = None
    if kind == "ours": ag = rv.make("NeuroSafe-Ctx", seed, tau, alpha, H)
    if kind == "ours_safe_warmup": ag = SafeWarmup(rv._domain_acts(), seed, tau, alpha=alpha, **rv.OURS)
    U, V = [], []
    for t in range(H):
        if kind == "always_safe": a, z = 0.0, 0.0
        elif kind == "oracle": a, z = env.oracle(alpha), 0.0
        else: a, z = ag.act(obs)
        obs, u, h, _, _ = env.step(a)
        if ag is not None: ag.observe(a, z, u, h)
        U.append(u); V.append(float(h > tau))
    V = np.array(V)
    return dict(domain=domain, kind=kind, alpha=alpha, seed=seed, ret=float(np.mean(U)), viol=float(V.mean()),
                viol_warmup=float(V[:10].sum() / H), viol_after=float(V[10:].sum() / H))

rows = []
for domain in sys.argv[1:] or ["ihdp", "scm"]:
    kinds = ["ours", "ours_safe_warmup", "always_safe"] + (["oracle"] if domain == "ihdp" else [])
    for alpha in [0.02, 0.03, 0.05, 0.075]:
        for kind in kinds:
            for seed in range(100, 150):
                rows.append(run(domain, kind, seed, alpha))
df = pd.DataFrame(rows)
os.makedirs("results/v3", exist_ok=True); df.to_csv("results/v3/small_alpha_check.csv", index=False)
g = df.groupby(["domain", "alpha", "kind"]).agg(ret=("ret", "mean"), viol=("viol", "mean"),
        from_warmup=("viol_warmup", "mean"), after_warmup=("viol_after", "mean"),
        viol_se=("viol", lambda x: x.std(ddof=1) / np.sqrt(len(x)))).reset_index()
pd.set_option("display.width", 160)
print(g.round(4).to_string(index=False))
