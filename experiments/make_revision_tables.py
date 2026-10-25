"""Supplementary tables for the revision: the IHDP informativeness sweep and the small-alpha check.
Reads results/v3/ihdp_info_sweep.csv and results/v3/small_alpha_check.csv; writes paper/v3/tables/."""
import os
import numpy as np, pandas as pd
T = "paper/v3/tables"; os.makedirs(T, exist_ok=True)
def boot(x, B=2000, seed=0):
    rng = np.random.RandomState(seed); m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def tab(path, cols, header, rows):
    open(os.path.join(T, path), "w").write("\\begin{tabular}{" + cols + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
                                           + "\n".join(r + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
fmt = lambda m, l, h, d=3: (f"\\textbf{{{m:+.{d}f}}}" if (l > 0 or h < 0) else f"{m:+.{d}f}") + f" [{l:+.{d}f}, {h:+.{d}f}]"

df = pd.read_csv("results/v3/ihdp_info_sweep.csv"); rows = []
for r in sorted(df.rho.unique(), reverse=True):
    d = df[df.rho == r]
    a = d[(d.agent == "NeuroSafe-Ctx") & np.isclose(d.knob, 0.15)].set_index("seed")
    b = d[(d.agent == "Chance") & np.isclose(d.knob, 0.02)].set_index("seed")
    f = d[d.agent == "NeuroSafe-Ctx"].groupby("knob")[["ret", "viol"]].mean().sort_values("viol")
    c = d[d.agent == "Chance"].groupby("knob")[["ret", "viol"]].mean()
    gaps = [float(np.interp(v, f.viol, f.ret) - rr) for rr, v in zip(c.ret, c.viol) if f.viol.min() <= v <= f.viol.max()]
    rows.append(f"{r:.2f}{' (main text)' if r > 0.99 else ''} & {fmt(*boot((a.ret - b.ret).to_numpy()))} & "
                f"{fmt(*boot((a.viol - b.viol).to_numpy()), d=4)} & {sum(g > 0 for g in gaps)}/{len(gaps)} ({np.mean(gaps):+.2f}) & {f.viol.min():.3f}")
tab("ihdp_info_sweep.tex", "lcccc",
    "$\\rho$ & $\\Delta$R, ours $\\alpha{=}0.15$ vs Chance $\\alpha{=}0.02$ & $\\Delta$V, same pair & frontier: ours above (mean gap) & our lowest $V$", rows)

g = pd.read_csv("results/v3/small_alpha_check.csv")
g = g.groupby(["domain", "alpha", "kind"]).agg(ret=("ret", "mean"), viol=("viol", "mean"), wu=("viol_warmup", "mean")).reset_index()
NAME = {"ours": "NeuroSafe-Ctx", "ours_safe_warmup": "safe warm-up variant", "always_safe": "always $a=0$", "oracle": "context oracle"}
rows = [f"{r.domain.upper() if r.domain == 'scm' else 'IHDP'} & {r.alpha:g} & {NAME[r.kind]} & {r.ret:.2f} & {r.viol:.4f} & {r.wu:.4f}"
        for r in g.sort_values(["domain", "alpha"]).itertuples()]
tab("small_alpha.tex", "llcccc", "Domain & $\\alpha$ & Policy & return & violation & of which warm-up", rows)
print("wrote", T + "/ihdp_info_sweep.tex and", T + "/small_alpha.tex")

# ---- Table S10: Safe UCB-GLM against NeuroSafe-Ctx at equal alpha (IHDP and SCM)
rows = []
for dom in ["ihdp", "scm"]:
    p = f"results/v3/{dom}/safeglm_episodes.csv"
    if not os.path.exists(p):
        continue
    gl = pd.read_csv(p); o = pd.read_csv(f"results/v3/{dom}/frontier_episodes.csv"); o = o[o.agent == "NeuroSafe-Ctx"]
    for a in sorted(gl.knob.unique()):
        G = gl[np.isclose(gl.knob, a)].set_index("seed"); O = o[np.isclose(o.knob, a)].set_index("seed")
        rows.append(f"{'IHDP' if dom == 'ihdp' else 'SCM'} & {a:g} & {G.ret.mean():.3f} / {G.viol.mean():.4f} & {O.ret.mean():.3f} / {O.viol.mean():.4f} & "
                    f"{fmt(*boot((O.ret - G.ret).to_numpy()))} & {fmt(*boot((O.viol - G.viol).to_numpy()), d=4)}")
tab("safe_glm.tex", "llcccc", "Domain & $\\alpha$ & Safe UCB-GLM: R / V & ours: R / V & $\\Delta$R (ours $-$ Safe UCB-GLM) & $\\Delta$V", rows)
print("wrote", T + "/safe_glm.tex")
