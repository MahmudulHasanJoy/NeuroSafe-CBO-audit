"""MIMIC-IV gate on BOTH halves: the fit half (reproduces mimic_audit.py) and the held-out half
(answers the referee). Same models as the audit: logistic harm model, gradient-boosted MAP model,
both fitted on the fit half only. Writes aggregates only: results/v3/mimic/gate_holdout.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.getcwd())
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from neurosafe_cbo.env_mimic_v3 import _load, K

c = _load(); fit, hold, F = c["fit"], c["ope"], c["feats"]
def design(d, oh):
    S, a = d[F].to_numpy(float), d.dbin.to_numpy(int)
    return np.column_stack([S, np.eye(K)[a]]) if oh else np.column_stack([S, a])
def dmat(S, k, oh):
    return np.column_stack([S, np.tile(np.eye(K)[k], (len(S), 1))]) if oh else np.column_stack([S, np.full(len(S), k)])
mh = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(design(fit, True), fit.new_arr.to_numpy(float))
mm = HGR(max_depth=4, max_iter=250, random_state=0).fit(design(fit, False), fit.map_now.to_numpy(float))

def gate(df):
    sub = df.sample(n=min(8000, len(df)), random_state=1)
    S, acl = sub[F].to_numpy(float), sub.dbin.to_numpy(int); ix = np.arange(len(S))
    P = np.column_stack([mh.predict_proba(dmat(S, k, True))[:, 1] for k in range(K)])
    M = np.column_stack([mm.predict(dmat(S, k, False)) for k in range(K)])
    clin, bu = M[ix, acl], M.argmax(1)
    rows = []
    for al in [0.015, 0.02, 0.03, 0.05, 0.075, 0.10]:
        ok = P <= al; binds = ~ok[ix, bu]
        cb = np.where(ok.any(1), np.where(ok, M, -np.inf).max(1), M[ix, P.argmin(1)])
        r = dict(alpha=al, choice=float((ok.sum(1) >= 2).mean()), binds=float(binds.mean()),
                 cost=float((M.max(1) - cb)[binds].mean()) if binds.any() else 0.0, headroom=float((cb - clin).mean()))
        r["passed"] = bool(r["choice"] >= 0.80 and r["binds"] >= 0.10 and r["headroom"] >= 0.5)
        rows.append(r)
    return dict(n=int(len(sub)), unconstrained_headroom=float((M.max(1) - clin).mean()), gate=rows)

out = {"fit_half": gate(fit), "heldout_half": gate(hold)}
os.makedirs("results/v3/mimic", exist_ok=True)
json.dump(out, open("results/v3/mimic/gate_holdout.json", "w"), indent=2)
for half, res in out.items():
    print(f"\n{half} (n={res['n']}, unconstrained headroom {res['unconstrained_headroom']:+.2f} mmHg)")
    print("  alpha  >=2 safe  best unsafe  cost  headroom  gate")
    for r in res["gate"]:
        print(f"  {r['alpha']:<6g} {r['choice']:7.3f}  {r['binds']:10.3f}  {r['cost']:5.2f}  {r['headroom']:+7.2f}  {'pass' if r['passed'] else 'fail'}")
h = out["heldout_half"]; best = max(h["gate"], key=lambda r: r["headroom"])
print(f"\nVERDICT: held-out gate {'PASSES somewhere - tell the authors' if any(r['passed'] for r in h['gate']) else 'fails at every alpha'}; "
      f"best constrained headroom {best['headroom']:+.2f} mmHg (alpha={best['alpha']:g}), unconstrained {h['unconstrained_headroom']:+.2f} mmHg")
