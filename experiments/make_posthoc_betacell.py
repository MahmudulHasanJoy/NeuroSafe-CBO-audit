import numpy as np, pandas as pd
from scipy import stats
d = "results/v3/ihdp/"
fr = pd.read_csv(d + "frontier_episodes.csv"); fr = fr[fr.agent == "NeuroSafe-Ctx"]
bc = pd.read_csv(d + "posthoc_betacell_episodes.csv")
def tci(x):
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    return x.mean(), lo, hi
lines = []
print("A) PRE-REGISTERED, same alpha (full - BetaCell):")
for a in (0.10, 0.15, 0.20):
    f = fr[fr.knob == a].set_index("seed"); b = bc[bc.knob == a].set_index("seed")
    r, v = tci(f.ret - b.ret), tci(f.viol - b.viol)
    print(f"  a={a:.2f} dR {r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]  dV {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]")
    lines.append(f"same $\\alpha={a:g}$ & {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}] & {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] \\\\")
lines.append("\\midrule")
print("B) POST HOC, matched violation:")
F = fr.pivot(index="seed", columns="knob", values=["ret", "viol"])
B = bc.pivot(index="seed", columns="knob", values=["ret", "viol"])
seeds = F.index.values; rng = np.random.RandomState(0)
def curve(Fs):
    v = Fs["viol"].mean().values; r = Fs["ret"].mean().values; o = np.argsort(v, kind="stable")
    return v[o], r[o]
v0, r0 = curve(F)
for a in sorted(bc.knob.unique()):
    bv, br = B["viol"][a].mean(), B["ret"][a].mean()
    if not (v0.min() <= bv <= v0.max()):
        print(f"  BetaCell a={a:g}: viol {bv:.4f} outside full range [{v0.min():.4f},{v0.max():.4f}] - no matched point")
        continue
    gap = np.interp(bv, v0, r0) - br
    boots = []
    for _ in range(2000):
        s = rng.choice(seeds, len(seeds), replace=True)
        vv, rr = curve(F.loc[s])
        boots.append(np.interp(B["viol"][a].loc[s].mean(), vv, rr) - B["ret"][a].loc[s].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  BetaCell a={a:g}: at viol {bv:.4f} full {np.interp(bv, v0, r0):.3f} vs BetaCell {br:.3f} -> gap {gap:+.3f} [{lo:+.3f},{hi:+.3f}]")
    lines.append(f"matched $V\\approx{bv:.3f}$ (BetaCell $\\alpha={a:g}$) & {gap:+.3f} [{lo:+.3f}, {hi:+.3f}] & -- \\\\")
hdr = ("\\begin{tabular}{lll}\n\\toprule\nComparison (full $-$ BetaCell) & $\\Delta$R [95\\% CI] & "
       "$\\Delta$V [95\\% CI] \\\\\n\\midrule\n")
open("paper/v3/tables/posthoc_betacell.tex", "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
print("wrote paper/v3/tables/posthoc_betacell.tex")
