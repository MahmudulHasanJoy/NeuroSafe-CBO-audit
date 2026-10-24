import numpy as np, pandas as pd
d = "results/v3/ihdp/"
a2 = pd.read_csv(d + "ablation2_episodes.csv")
fr = pd.read_csv(d + "frontier_episodes.csv")
old = pd.read_csv(d + "ablation_episodes.csv")
for name, ref in [("NeuroSafe-Ctx", fr), ("Ctx-noGLM", old), ("Ctx-noBucket", old), ("Ctx-noKalman", old)]:
    m = a2[a2.agent == name].merge(ref[ref.agent == name], on=["knob", "seed"], suffixes=("", "_old"))
    print(f"REPRODUCTION {name:13s}: {len(m)} rows | max|ret diff| {np.abs(m.ret - m.ret_old).max():.2e} "
          f"| max|viol diff| {np.abs(m.viol - m.viol_old).max():.2e}")
base = a2[a2.agent == "NeuroSafe-Ctx"].set_index(["knob", "seed"])
for name in ["Ctx-noContext", "Ctx-BetaCell"]:
    other = a2[a2.agent == name].set_index(["knob", "seed"])
    print(f"\n{name} vs NeuroSafe-Ctx (paired by seed; + means NeuroSafe-Ctx higher)")
    wins = 0
    for k in sorted(a2.knob.unique()):
        dr = base.loc[k].ret - other.loc[k].ret
        dv = base.loc[k].viol - other.loc[k].viol
        ser = dr.std(ddof=1) / np.sqrt(len(dr)); sev = dv.std(ddof=1) / np.sqrt(len(dv))
        counts = k in (0.10, 0.15, 0.20)
        wins += int(counts and dr.mean() > 2 * ser and dv.mean() <= 2 * sev)
        ef = other.loc[k].empty_frac.mean() if "empty_frac" in other.columns else float("nan")
        print(f"  a={k:<6} ret {dr.mean():+.3f} (se {ser:.3f}) viol {dv.mean():+.4f} (se {sev:.4f}) "
              f"| {name} viol {other.loc[k].viol.mean():.4f} empty {ef:.2f}{'  <- counts' if counts else ''}")
    print(f"  VERDICT ({wins}/3): " + ("NeuroSafe-Ctx WINS - component matters" if wins >= 2
                                       else "no reliable win - component not shown to matter"))
