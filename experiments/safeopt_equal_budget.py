"""Reference SafeOpt with an EQUAL tuning budget (18 configurations, as for NeuroSafe-Ctx).
Grid: action lengthscale {0.1,0.2,0.4} x utility noise {0.1,0.5} x context lengthscale {0.5,1,2}.
Same selection rule as run_safeopt_ref.py: highest tuning-seed (0-9) return among configurations whose
tuning-seed violation at beta=2 is <= 0.15. Then the held-out frontier (seeds 100-149, beta in 1,2,3,4).
Writes results/v3/<domain>/safeopt_ref18_config.json and safeopt_ref18_episodes.csv, and compares
with the original 6-configuration run and with NeuroSafe-Ctx.  Usage: python experiments/safeopt_equal_budget.py ihdp"""
import os, sys, json, itertools, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import experiments.run_v3 as R

domain = sys.argv[1]; out = os.path.join(R.OUT, domain); t0 = time.time()
def use(cfg):  # make() looks up safeopt_config at call time, so this swaps the hyperparameters in
    R.safeopt_config = lambda dom, cfg=cfg: dict(cfg, ctx_scale=R.context_scale(dom))
best, log = None, []
for ls_a, u_noise, ls_z in itertools.product([0.1, 0.2, 0.4], [0.1, 0.5], [0.5, 1.0, 2.0]):
    cfg = dict(ls_a=ls_a, u_noise=u_noise, ls_z=ls_z); use(cfg)
    rs = [R.episode(domain, "SafeOpt-ref", s, 2.0) for s in range(10)]
    ret, vio = float(np.mean([r["ret"] for r in rs])), float(np.mean([r["viol"] for r in rs]))
    log.append(dict(cfg, ret=ret, viol=vio)); print(f"tune {cfg}: ret {ret:.4f} viol {vio:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if vio <= 0.15 and (best is None or ret > best[0]): best = (ret, cfg)
cfg = best[1] if best else dict(ls_a=0.2, u_noise=0.5, ls_z=1.0)
json.dump(dict(chosen=cfg, grid=log), open(os.path.join(out, "safeopt_ref18_config.json"), "w"), indent=2)
print("chosen", cfg, flush=True); use(cfg)
rows = [R.episode(domain, "SafeOpt-ref", s, b) for b in R.FRONTIER_KNOBS["SafeOpt-ref"] for s in range(100, 150)]
new = pd.DataFrame(rows); new.to_csv(os.path.join(out, "safeopt_ref18_episodes.csv"), index=False)

old = pd.read_csv(os.path.join(out, "safeopt_ref_episodes.csv")); old = old[old.seed >= 100]
ours = pd.read_csv(os.path.join(out, "frontier_episodes.csv")); ours = ours[ours.agent == "NeuroSafe-Ctx"]
fo = ours.groupby("knob")[["ret", "viol"]].mean().sort_values("viol")
print("\nbeta | 6-config ret / viol | 18-config ret / viol")
for b in sorted(new.knob.unique()):
    o6, o18 = old[np.isclose(old.knob, b)], new[np.isclose(new.knob, b)]
    print(f"{b:4g} | {o6.ret.mean():.4f} / {o6.viol.mean():.4f} | {o18.ret.mean():.4f} / {o18.viol.mean():.4f}")
for name, d in [("6-config", old), ("18-config", new)]:
    f = d.groupby("knob")[["ret", "viol"]].mean()
    gaps = [float(np.interp(v, fo.viol, fo.ret) - r) for r, v in zip(f.ret, f.viol) if fo.viol.min() <= v <= fo.viol.max()]
    print(f"NeuroSafe-Ctx frontier above SafeOpt ({name}) at {sum(g > 0 for g in gaps)}/{len(gaps)} comparable points; gaps {np.round(gaps, 3).tolist()}")
print(f"done in {time.time()-t0:.0f}s")
