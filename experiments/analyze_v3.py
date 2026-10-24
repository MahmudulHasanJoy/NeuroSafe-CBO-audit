"""Analyze v3 results for one domain -> JSON (+ printed summary)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_v3 import paired, OUT, DEFAULT_KNOB

dom = sys.argv[1]
D = os.path.join(OUT, dom)
rd = lambda t: pd.read_csv(os.path.join(D, f"{t}_episodes.csv")) if os.path.exists(os.path.join(D, f"{t}_episodes.csv")) else pd.DataFrame()
tune, fr, de, ab = rd("tuneseeds"), rd("frontier"), rd("defaults"), rd("ablation")
so = rd("safeopt_ref")
held = pd.concat([fr, de, ab] + ([so[so.seed >= 100]] if not so.empty else []), ignore_index=True)
if not so.empty:
    tune = pd.concat([tune, so[so.seed < 100]], ignore_index=True)

def summ(df):
    g = df.groupby(["agent", "knob"]).agg(ret=("ret", "mean"), ret_se=("ret", "sem"),
                                         viol=("viol", "mean"), viol_se=("viol", "sem"), n=("seed", "nunique")).reset_index()
    return g

F = summ(held)
res = {"domain": dom, "frontier": F.to_dict("records"), "matched": [], "ablation": [], "defaults": {}}

# matched operating points: alpha for ours chosen on TUNING seeds so that our
# tuning-seed violation <= the baseline's tuning-seed violation (closest from below)
T = summ(tune)
ours_t = T[T.agent == "NeuroSafe-Ctx"]
for base in ["Chance", "SafeGP", "SafeOpt-ref", "Conservative"]:
    for _, b in T[T.agent == base].iterrows():
        ok = ours_t[ours_t.viol <= b.viol]
        if ok.empty:
            continue
        a = float(ok.sort_values("viol").iloc[-1].knob)
        r = paired(held, "NeuroSafe-Ctx", base, knob_new=a, knob_base=b.knob)
        r.update(base=base, base_knob=float(b.knob), our_alpha=a)
        res["matched"].append(r)

# ablations at alpha = 0.15 and at each alpha (dominance count)
for abl in ["Ctx-noGLM", "Ctx-noBucket", "Ctx-noKalman", "Ctx-noContext"]:
    if ab.empty or abl not in set(ab.agent):
        continue
    r = paired(held, abl, "NeuroSafe-Ctx", knob_new=0.15, knob_base=0.15)
    r.update(variant=abl)
    res["ablation"].append(r)

for ag in set(de.agent) if not de.empty else []:
    x = de[de.agent == ag]
    res["defaults"][ag] = dict(ret=float(x.ret.mean()), ret_se=float(x.ret.sem()),
                               viol=float(x.viol.mean()), viol_se=float(x.viol.sem()))

json.dump(res, open(os.path.join(D, "summary.json"), "w"), indent=2)
pd.set_option("display.width", 200)
print(F.round(4).to_string())
print("\nMATCHED (our alpha chosen on tuning seeds)")
for r in res["matched"]:
    print(f"vs {r['base']:12s}@{r['base_knob']:<5} ours a={r['our_alpha']:<6} "
          f"ret {r['base_ret']:.3f}->{r['new_ret']:.3f} dR={r['dR']:+.3f} [{r['dR_ci'][0]:+.3f},{r['dR_ci'][1]:+.3f}] p={r['p_dR_two']:.1e} | "
          f"viol {r['base_viol']:.3f}->{r['new_viol']:.3f} dV={r['dV']:+.4f} [{r['dV_ci'][0]:+.4f},{r['dV_ci'][1]:+.4f}] p={r['p_dV_two']:.1e}")
print("\nABLATION (variant - full, alpha=0.15)")
for r in res["ablation"]:
    print(f"{r['variant']:14s} dR={r['dR']:+.3f} [{r['dR_ci'][0]:+.3f},{r['dR_ci'][1]:+.3f}] p={r['p_dR_two']:.1e} | dV={r['dV']:+.4f} [{r['dV_ci'][0]:+.4f},{r['dV_ci'][1]:+.4f}] p={r['p_dV_two']:.1e}")
print("\nDEFAULTS", json.dumps(res["defaults"], indent=0))
