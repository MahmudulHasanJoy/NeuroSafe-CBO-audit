"""MIMIC-IV vasopressor domain: agent comparison + off-policy evaluation."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurosafe_cbo import ope as OPE
from neurosafe_cbo.v3 import ConservativeUCB, NeuroSafeCtxV3, SafeGPUCB
K = 5
ACTS = np.arange(K, dtype=float) / (K - 1)
FEATS = ["map_prev","map_min_prev","prev_dose","hr_prev","fluid_prev","n_other_prev","anchor_age","n_dx"]
TAU = 0.5
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
class MimicSim:
    def __init__(self, df, seed=0):
        self.rng = np.random.RandomState(seed)
        self.S = df[FEATS].to_numpy(dtype=float); self.a = df.dbin.to_numpy(dtype=int)
        self.map_next = df.map_now.to_numpy(dtype=float); self.harm = df.new_arr.to_numpy(dtype=float)
        self.n = len(df); X = np.column_stack([self.S, self.a])
        self.m_map = HistGradientBoostingRegressor(max_depth=4, max_iter=250, random_state=seed).fit(X, self.map_next)
        self.m_arr = HistGradientBoostingClassifier(max_depth=4, max_iter=250, random_state=seed).fit(X, self.harm)
        self.res_sd = float(np.std(self.map_next - self.m_map.predict(X))); self.t = 0
    def diagnostics(self):
        from sklearn.metrics import roc_auc_score
        X = np.column_stack([self.S, self.a])
        return {"map_resid_sd": self.res_sd,
                "map_r2": float(1 - np.var(self.map_next - self.m_map.predict(X)) / np.var(self.map_next)),
                "arr_auc": float(roc_auc_score(self.harm, self.m_arr.predict_proba(X)[:,1])),
                "arr_base_rate": float(self.harm.mean())}
    def reset(self): self.t = 0; return self._obs()
    def _obs(self):
        self.i = int(self.rng.randint(self.n)); s = self.S[self.i]
        return np.array([s[0]-70.0, s[1]-70.0, s[2], 0.0])
    def step(self, a):
        k = int(np.clip(np.round(float(a)*(K-1)), 0, K-1))
        x = np.concatenate([self.S[self.i], [k]])[None, :]
        mp = float(self.m_map.predict(x)[0]) + self.rng.normal(0, self.res_sd)
        p = float(self.m_arr.predict_proba(x)[0,1])
        H = 1.0 if self.rng.rand() < p else 0.0
        self.t += 1
        return self._obs(), mp, H, False, {}
    def action_probs(self, agent, S_eval, eps=0.05):
        P = np.full((len(S_eval), K), eps/K)
        for j, s in enumerate(S_eval):
            a, _ = agent.act(np.array([s[0]-70.0, s[1]-70.0, s[2], 0.0]))
            P[j, int(np.clip(np.round(float(a)*(K-1)),0,K-1))] += 1.0-eps
        return P / P.sum(axis=1, keepdims=True)
def make_agent(name, seed, knob):
    if name == "NeuroSafe-Ctx":
        return NeuroSafeCtxV3(ACTS, seed, tau=TAU, alpha=knob, beta_h=1.0, beta_u=0.5, basis="cubic8", context="raw")
    if name == "Ctx-noContext":
        return NeuroSafeCtxV3(ACTS, seed, tau=TAU, alpha=knob, beta_h=1.0, beta_u=0.5, basis="cubic8", context="none")
    if name == "Ctx-noGLM":
        return NeuroSafeCtxV3(ACTS, seed, tau=TAU, alpha=knob, harm="gauss", basis="cubic8", context="raw")
    if name == "SafeGP": return SafeGPUCB(ACTS, seed, tau=TAU, beta=knob)
    if name == "Conservative": return ConservativeUCB(ACTS, seed, eta=knob)
    if name == "Chance":
        return NeuroSafeCtxV3(ACTS, seed, tau=TAU, alpha=knob, harm="gauss", utility="pooled", context="none")
    if name == "RiskNeutral":
        return NeuroSafeCtxV3(ACTS, seed, tau=TAU, alpha=0.99, beta_h=0.0, basis="cubic8", context="raw")
    raise KeyError(name)
KNOBS = {"NeuroSafe-Ctx":[0.02,0.05,0.10,0.20,0.35], "Chance":[0.02,0.05,0.10,0.20,0.35],
         "Ctx-noContext":[0.02,0.05,0.10,0.20,0.35], "Ctx-noGLM":[0.02,0.05,0.10,0.20,0.35],
         "SafeGP":[1.0,2.0,3.0], "Conservative":[0.05,0.1,0.3], "RiskNeutral":[0.99]}
def run_episode(sim, name, seed, knob, horizon):
    ag = make_agent(name, seed, knob); obs = sim.reset(); U, V = [], []
    for _ in range(horizon):
        a, z = ag.act(obs); obs, u, h, _, _ = sim.step(a); ag.observe(a, z, u, h)
        U.append(u); V.append(h > TAU)
    return ag, float(np.mean(U)), float(np.mean(V))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="results/mimic_windows.pkl")
    ap.add_argument("--out", default="results"); ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seeds", default="0:20"); ap.add_argument("--agents", default="")
    a = ap.parse_args(); s0, s1 = (int(x) for x in a.seeds.split(":"))
    df = pd.read_pickle(a.windows)
    hypo = df[df.hypotensive & (df.prev_dbin > 0)].dropna(subset=FEATS+["map_now"]).copy()
    hypo["new_arr"] = hypo.new_arr.astype(float)
    log(f"hypotensive on-drug windows: {len(hypo):,} in {hypo.stay_id.nunique():,} stays")
    stays = np.array(sorted(hypo.stay_id.unique())); rng = np.random.RandomState(0); rng.shuffle(stays)
    cut = int(0.5*len(stays))
    fit_df = hypo[hypo.stay_id.isin(set(stays[:cut]))]; ope_df = hypo[hypo.stay_id.isin(set(stays[cut:]))]
    log(f"simulator-fit windows {len(fit_df):,} | held-out OPE windows {len(ope_df):,}")
    sim = MimicSim(fit_df, seed=0); diag = sim.diagnostics()
    log("simulator fit: " + json.dumps({k: round(v,4) for k,v in diag.items()}))
    S_ope = ope_df[FEATS].to_numpy(dtype=float); a_ope = ope_df.dbin.to_numpy(dtype=int)
    r_ope = ope_df.map_now.to_numpy(dtype=float); h_ope = ope_df.new_arr.to_numpy(dtype=float)
    bp = OPE.BehaviourPolicy().fit(S_ope, a_ope); pi_b = bp.proba(S_ope)
    log("behaviour policy: " + json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in bp.diag.items()}))
    q_r = OPE.QModel(K).fit(S_ope, a_ope, r_ope); q_h = OPE.QModel(K).fit(S_ope, a_ope, h_ope)
    names = a.agents.split(",") if a.agents else list(KNOBS)
    rows, ope_rows = [], []
    for name in names:
        for knob in KNOBS[name]:
            agents = []
            for s in range(s0, s1):
                ag, u, v = run_episode(sim, name, s, knob, a.horizon)
                rows.append(dict(agent=name, knob=knob, seed=s, ret=u, viol=v)); agents.append(ag)
            P = sim.action_probs(agents[0], S_ope)
            res = OPE.evaluate(P, S_ope, a_ope, r_ope, h_ope, pi_b, q_r, q_h)
            res.update(agent=name, knob=knob); ope_rows.append(res)
            log(f"  {name} @{knob}: sim ret {np.mean([r['ret'] for r in rows[-(s1-s0):]]):.2f} "
                f"viol {np.mean([r['viol'] for r in rows[-(s1-s0):]]):.4f} | OPE wis {res['wis_return']:.2f} "
                f"harm {res['wis_harm']:.4f} ESS {res['ess']:.0f} ({res['ess_frac']*100:.1f}%) reliable={res['reliable']}")
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out,"mimic_sim_episodes.csv"), index=False)
    pd.DataFrame(ope_rows).to_csv(os.path.join(a.out,"mimic_ope.csv"), index=False)
    json.dump({"n_hypo_windows": int(len(hypo)), "n_hypo_stays": int(hypo.stay_id.nunique()),
        "n_fit": int(len(fit_df)), "n_ope": int(len(ope_df)), "simulator": diag,
        "behaviour_policy": bp.diag, "behaviour_return": float(r_ope.mean()),
        "behaviour_harm": float(h_ope.mean())}, open(os.path.join(a.out,"mimic_domain_report.json"),"w"), indent=2)
    print("\n=== SIM (mean over seeds) ===")
    print(pd.DataFrame(rows).groupby(["agent","knob"])[["ret","viol"]].mean().round(4).to_string())
    print("\n=== OPE vs the clinicians logged actions ===")
    o = pd.DataFrame(ope_rows)
    print(o[["agent","knob","wis_return","wis_harm","dr_return","dr_harm","ess","ess_frac","reliable"]].round(4).to_string(index=False))
    print(f"\nclinicians: return {r_ope.mean():.2f}  harm {h_ope.mean():.4f}")
if __name__ == "__main__": main()
