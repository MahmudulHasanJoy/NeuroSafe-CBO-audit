"""Reference-SafeOpt baseline: smoke test, tuning, held-out frontier.
  python experiments/run_safeopt_ref.py --smoke
  python experiments/run_safeopt_ref.py --domain scm      (tune on 0-9, then held-out 100-149)
Tuning grid (6 configs) is smaller than ours (18); choice rule: highest tuning-seed return
among configs whose tuning-seed violation at beta=2 is <= 0.15 (the nominal alpha)."""
import sys, os, json, itertools, argparse, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments import run_v3 as R

ap = argparse.ArgumentParser(); ap.add_argument("--domain", default="scm"); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    for dom in ["scm", "lending"]:
        R.DOMAINS[dom] = dict(R.DOMAINS[dom], H=30)
        print(dom, R.episode(dom, "SafeOpt-ref", 0, 2.0)); R.DOMAINS[dom]["H"] = 200
    print("smoke test OK"); sys.exit()
dom = a.domain; D = os.path.join(R.OUT, dom); os.makedirs(D, exist_ok=True)
cfgp = os.path.join(D, "safeopt_ref_config.json"); t0 = time.time()
marker = cfgp + ".tuning"
if os.path.exists(marker) and os.path.exists(cfgp):
    os.remove(cfgp)  # a previous tuning run crashed; its partial config must not be reused
if not os.path.exists(cfgp):
    open(marker, "w").close()
    best = None
    for ls_a, u_noise in itertools.product([0.1, 0.2, 0.4], [0.1, 0.5]):
        json.dump(dict(ls_a=ls_a, u_noise=u_noise), open(cfgp, "w"))
        rs = [R.episode(dom, "SafeOpt-ref", s, 2.0) for s in range(10)]
        ret, vio = np.mean([r["ret"] for r in rs]), np.mean([r["viol"] for r in rs])
        print(f"tune ls_a={ls_a} u_noise={u_noise}: ret {ret:.4f} viol {vio:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vio <= 0.15 and (best is None or ret > best[0]):
            best = (ret, dict(ls_a=ls_a, u_noise=u_noise))
    json.dump(best[1] if best else dict(ls_a=0.2, u_noise=0.5), open(cfgp, "w"))
    os.remove(marker)
print("config", open(cfgp).read())
R.run_many(dom, ["SafeOpt-ref"], list(range(0, 10)) + list(range(100, 150)),
           knobs={"SafeOpt-ref": R.FRONTIER_KNOBS["SafeOpt-ref"]}, tag="safeopt_ref")
print("done", time.time() - t0)
