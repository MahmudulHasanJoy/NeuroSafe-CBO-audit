"""Safe UCB-GLM baseline: tune (beta_h, beta_u) on tuning seeds 0-9 with the same rule used for reference
SafeOpt (highest tuning-seed return among configurations whose tuning-seed violation at alpha = 0.15 is
<= 0.15), then run the held-out frontier (seeds 100-149) over the same alpha grid as NeuroSafe-Ctx.
Usage: python experiments/run_safe_glm.py ihdp   (also scm, lending). Writes results/v3/<domain>/safeglm_*."""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import experiments.run_v3 as rv
from neurosafe_cbo.safe_glm import SafeGLMUCB

def episode(domain, seed, alpha, cfg):
    rv.DOMAIN_OF_RUN[0] = domain
    tau, H = rv.DOMAINS[domain]["tau"], rv.DOMAINS[domain]["H"]
    env = rv.make_env(domain, seed, H); obs = env.reset()
    ag = SafeGLMUCB(rv._domain_acts(), seed, tau, alpha=alpha, **cfg); U, V = [], []
    for _ in range(H):
        a, z = ag.act(obs); obs, u, h, _, _ = env.step(a); ag.observe(a, z, u, h)
        U.append(u); V.append(h > tau)
    return dict(domain=domain, agent="SafeGLM", knob=alpha, seed=seed, ret=float(np.mean(U)), viol=float(np.mean(V)),
                empty_frac=ag.stats["empty_feasible"] / max(ag.stats["decisions"], 1))

domain = sys.argv[1]; out = os.path.join(rv.OUT, domain); os.makedirs(out, exist_ok=True)
grid = [dict(beta_h=bh, beta_u=bu) for bh, bu in itertools.product([0.5, 1.0, 2.0], [0.25, 0.5, 1.0])]
best, log = None, []
for cfg in grid:
    rs = [episode(domain, s, 0.15, cfg) for s in range(10)]
    ret, vio = np.mean([r["ret"] for r in rs]), np.mean([r["viol"] for r in rs])
    log.append(dict(cfg, ret=ret, viol=vio)); print(f"tune {cfg}: ret {ret:.4f} viol {vio:.4f}", flush=True)
    if vio <= 0.15 and (best is None or ret > best[0]): best = (ret, cfg)
cfg = best[1] if best else dict(beta_h=1.0, beta_u=0.5)
json.dump(dict(chosen=cfg, grid=log), open(os.path.join(out, "safeglm_config.json"), "w"), indent=2)
print("chosen", cfg, flush=True)
rows = [episode(domain, s, a, cfg) for a in rv.FRONTIER_KNOBS["NeuroSafe-Ctx"] for s in range(100, 150)]
pd.DataFrame(rows).to_csv(os.path.join(out, "safeglm_episodes.csv"), index=False)
print("wrote", os.path.join(out, "safeglm_episodes.csv"))
