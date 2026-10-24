"""MIMIC audit: simulator validation, pre-registered gate, real-data OPE confirmation,
exploratory harm finding. Writes aggregates only (no patient-level data)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from neurosafe_cbo.env_mimic_v3 import _load, K
from neurosafe_cbo import ope as O

c = _load(); fit, hold, F = c["fit"], c["ope"], c["feats"]
A = {"n_fit_windows": len(fit), "n_hold_windows": len(hold), "n_hold_stays": int(hold.stay_id.nunique())}
def design(d, oh):
    S, a = d[F].to_numpy(float), d.dbin.to_numpy(int)
    return np.column_stack([S, np.eye(K)[a]]) if oh else np.column_stack([S, a])
def dmat(S, k, oh):
    return np.column_stack([S, np.tile(np.eye(K)[k], (len(S), 1))]) if oh else np.column_stack([S, np.full(len(S), k)])
REG = dict(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
HARM = {"hgb_current": (lambda: HGC(max_depth=4, max_iter=250, random_state=0), False),
        "hgb_regularised": (lambda: HGC(**REG), False),
        "logistic": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)), True)}
MAPM = {"hgb_current": (lambda: HGR(max_depth=4, max_iter=250, random_state=0), False),
        "hgb_regularised": (lambda: HGR(**REG), False),
        "ridge": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), True)}
g, yh, ym = fit.stay_id.to_numpy(), fit.new_arr.to_numpy(float), fit.map_now.to_numpy(float)
gkf = GroupKFold(n_splits=5); A["cv"] = {"harm": {}, "map": {}}
for n, (mk, oh) in HARM.items():
    X = design(fit, oh); pr = np.zeros(len(fit))
    for tr, te in gkf.split(X, yh, g):
        pr[te] = mk().fit(X[tr], yh[tr]).predict_proba(X[te])[:, 1]
    A["cv"]["harm"][n] = dict(logloss=log_loss(yh, pr), auc=roc_auc_score(yh, pr), mean_pred=pr.mean())
for n, (mk, oh) in MAPM.items():
    X = design(fit, oh); pr = np.zeros(len(fit))
    for tr, te in gkf.split(X, ym, g):
        pr[te] = mk().fit(X[tr], ym[tr]).predict(X[te])
    A["cv"]["map"][n] = dict(r2=r2_score(ym, pr))
bh = min(A["cv"]["harm"], key=lambda n: A["cv"]["harm"][n]["logloss"])
bm = max(A["cv"]["map"], key=lambda n: A["cv"]["map"][n]["r2"])
A["chosen_harm"], A["chosen_map"] = bh, bm
oh0 = HARM["hgb_current"][0]().fit(design(fit, False), yh); om0 = MAPM["hgb_current"][0]().fit(design(fit, False), ym)
A["insample_original"] = dict(auc=roc_auc_score(yh, oh0.predict_proba(design(fit, False))[:, 1]),
                              r2=r2_score(ym, om0.predict(design(fit, False))))
hk, hoh = HARM[bh]; mk_, moh = MAPM[bm]
mh = hk().fit(design(fit, hoh), yh); mm = mk_().fit(design(fit, moh), ym)
A["heldout_chosen"] = dict(auc=roc_auc_score(hold.new_arr, mh.predict_proba(design(hold, hoh))[:, 1]),
                           r2=r2_score(hold.map_now, mm.predict(design(hold, moh))))

sub = fit.sample(n=min(8000, len(fit)), random_state=1)
S, acl = sub[F].to_numpy(float), sub.dbin.to_numpy(int); ix = np.arange(len(S))
P = np.column_stack([mh.predict_proba(dmat(S, k, hoh))[:, 1] for k in range(K)])
M = np.column_stack([mm.predict(dmat(S, k, moh)) for k in range(K)])
A["dose"] = dict(harm=P.mean(0).tolist(), map=M.mean(0).tolist())
clin, bu = M[ix, acl], M.argmax(1)
A["sim_clinician_map"], A["sim_unconstrained_best"] = clin.mean(), M.max(1).mean()
A["gate"] = []
for al in [0.015, 0.02, 0.03, 0.05, 0.075, 0.10]:
    ok = P <= al; binds = ~ok[ix, bu]
    cb = np.where(ok.any(1), np.where(ok, M, -np.inf).max(1), M[ix, P.argmin(1)])
    row = dict(alpha=al, choice=(ok.sum(1) >= 2).mean(), binds=binds.mean(),
               cost=(M.max(1) - cb)[binds].mean() if binds.any() else 0.0, headroom=(cb - clin).mean())
    row["passed"] = bool(row["choice"] >= 0.80 and row["binds"] >= 0.10 and row["headroom"] >= 0.5)
    A["gate"].append(row)

Sf, af = fit[F].to_numpy(float), fit.dbin.to_numpy(int)
Sh, ah = hold[F].to_numpy(float), hold.dbin.to_numpy(int)
r, hz = hold.map_now.to_numpy(float), hold.new_arr.to_numpy(float)
Ph = np.column_stack([mh.predict_proba(dmat(Sh, k, hoh))[:, 1] for k in range(K)])
Mh = np.column_stack([mm.predict(dmat(Sh, k, moh)) for k in range(K)])
bp = O.BehaviourPolicy(seed=0); bp.fit(Sf, af); pib = bp.proba(Sh)
okm = Ph <= 0.02
kp = np.where(okm.any(1), np.where(okm, Mh, -np.inf).argmax(1), Ph.argmin(1))
def soft(eps):
    pe = np.full((len(kp), K), eps / K); pe[np.arange(len(kp)), kp] += 1 - eps
    return pe
u, inv = np.unique(hold.stay_id.to_numpy(), return_inverse=True); NS = len(u)
ps = lambda x: np.bincount(inv, weights=x, minlength=NS)
def cdiff(pe, y, clip, B=2000, seed=0):
    w, _ = O.weights(pe, pib, ah, clip)
    sw, swy, sy, cnt = ps(w), ps(w * y), ps(y), ps(np.ones_like(y))
    point = swy.sum() / sw.sum() - sy.sum() / cnt.sum()
    rng = np.random.RandomState(seed); d = np.empty(B)
    for b in range(B):
        m = np.bincount(rng.randint(NS, size=NS), minlength=NS)
        d[b] = (m @ swy) / (m @ sw) - (m @ sy) / (m @ cnt)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(d=point, lo=lo, hi=hi)
qh = O.QModel(K, seed=0); qh.fit(Sf, af, yh)
qr = O.QModel(K, seed=0); qr.fit(Sf, af, ym)
res = O.evaluate(soft(0.05), Sh, ah, r, hz, pib, q_r=qr, q_h=qh, seed=0)
A["ope"] = dict(clin_map=r.mean(), clin_harm=hz.mean(), wis_map=res["wis_return"], wis_harm=res["wis_harm"],
                dr_map=res["dr_return"], dr_harm=res["dr_harm"], ess=res["ess"], ess_frac=res["ess_frac"],
                n_clipped=res["n_weights_clipped"], reliable=bool(res["reliable"]),
                map_diff=cdiff(soft(0.05), r, 20.0), harm_diff=cdiff(soft(0.05), hz, 20.0),
                policy_mix=(np.bincount(kp, minlength=K) / len(kp)).tolist(),
                clinician_mix=(np.bincount(ah, minlength=K) / len(ah)).tolist())
A["sensitivity"] = [dict(eps=e, clip=cl, **cdiff(soft(e), hz, cl)) for e in (0.02, 0.05, 0.10) for cl in (10.0, 20.0, 50.0)]

os.makedirs("results/v3/mimic", exist_ok=True)
json.dump(A, open("results/v3/mimic/audit.json", "w"), indent=2, default=float)
o = A["ope"]
print(f"chosen harm={bh} map={bm} | held-out AUC {A['heldout_chosen']['auc']:.3f} R2 {A['heldout_chosen']['r2']:.3f}")
print("gate at alpha=0.02:", {k: round(float(v), 3) for k, v in A["gate"][1].items()})
print(f"MAP diff {o['map_diff']['d']:+.3f} [{o['map_diff']['lo']:+.3f}, {o['map_diff']['hi']:+.3f}] | "
      f"harm diff {o['harm_diff']['d']:+.4f} [{o['harm_diff']['lo']:+.4f}, {o['harm_diff']['hi']:+.4f}] | "
      f"DR MAP {o['dr_map'] - o['clin_map']:+.3f}, DR harm {o['dr_harm'] - o['clin_harm']:+.4f}")
print("wrote results/v3/mimic/audit.json")
