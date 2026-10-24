import sys, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
OUT = "paper/v3/figures"; plt.rcParams.update({"font.size": 8})
STY = {"NeuroSafe-Ctx": ("#d95f02", "o", "NeuroSafe-Ctx (ours)"), "Chance": ("#1b73e8", "s", "Chance (context-blind)"),
       "SafeGP": ("#1b9e77", "^", "Safe GP-UCB (reimpl.)"), "Conservative": ("#777777", "D", "Conservative UCB"),
       "Ctx-noContext": ("#d95f02", "x", "ours, context removed"),
       "SafeOpt-ref": ("#7570b3", "v", "SafeOpt (reference pkg.)")}
interp = {}
for dom in [d for d in ["scm", "lending", "ihdp"] if os.path.exists(f"results/v3/{d}/summary.json")]:
    S = json.load(open(f"results/v3/{dom}/summary.json"))
    F = pd.DataFrame(S["frontier"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for ag, (col, mk, lab) in STY.items():
        f = F[F.agent == ag].sort_values("viol")
        if f.empty: continue
        ls = ":" if ag == "Ctx-noContext" else "-"
        ax.errorbar(f.viol, f.ret, xerr=1.96 * f.viol_se, yerr=1.96 * f.ret_se, color=col, marker=mk, ms=3.5, lw=1, ls=ls, capsize=1.5, label=lab)
    if dom == "lending":
        c = json.load(open("results/v3/lending/channel_ceiling.json"))
        cv = sorted((v["viol"], v["ret"]) for v in c.values())
        ax.plot(*zip(*cv), "k--", lw=0.8, label="channel ceiling (any agent)")
        o = S["defaults"]["Oracle"]; ax.plot(o["viol"], o["ret"], "k*", ms=8, label="full-information oracle")
    ax.set_xlabel("violation rate (lower is better)"); ax.set_ylabel("return / step")
    if dom == "lending": ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.28)
    if dom == "scm": ax.set_xlim(0, 0.26)
    ax.legend(fontsize=5.5, loc="lower right" if dom == "scm" else "center right", frameon=False)
    ax.set_title({"scm": "Nonstationary SCM (50 held-out seeds)", "lending": "German Credit, real defaults (50 held-out seeds)", "ihdp": "IHDP, binding harm channel (50 held-out seeds)"}[dom], fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_frontier_{dom}.pdf"); plt.close(fig)
    # frontier interpolation: at each competitor point inside our violation range, our interpolated return minus theirs
    ours = F[F.agent == "NeuroSafe-Ctx"].sort_values("viol")
    interp[dom] = {}
    for ag in ["Chance", "SafeGP", "SafeOpt-ref", "Conservative", "Ctx-noGLM", "Ctx-noContext", "Ctx-noBucket", "Ctx-noKalman"]:
        pts = []
        for _, r in F[F.agent == ag].iterrows():
            if ours.viol.min() <= r.viol <= ours.viol.max():
                pts.append(float(np.interp(r.viol, ours.viol, ours.ret) - r.ret))
        interp[dom][ag] = dict(n_comparable=len(pts), n_ours_above=int(sum(p > 0 for p in pts)),
                               deltas=[round(p, 4) for p in pts],
                               n_total=int((F.agent == ag).sum()), their_min_viol=float(F[F.agent == ag].viol.min()) if (F.agent == ag).any() else None)
json.dump(interp, open("results/v3/frontier_interp.json", "w"), indent=2)
for d in interp:
    for a, v in interp[d].items(): print(d, a, v)
# floor figure
R = json.load(open("results/v3/floor_check.json"))
rh = np.linspace(0.4, 0.99, 100)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(rh, np.arccos(rh) / np.pi, "k-", lw=1, label=r"Theorem 1 floor $\arccos(\rho)/\pi$")
ax.plot([r["rho"] for r in R], [r["bayes"]["combo"] for r in R], "ko", ms=4, label="Bayes rule (simulated)")
ax.plot([r["rho"] for r in R], [r["cbayes"]["combo"] for r in R], "s", color="#1b73e8", ms=4, label=r"$\alpha$-constrained Bayes rule")
ax.plot([r["rho"] for r in R], [r["ours"]["combo"] for r in R], "o", color="#d95f02", ms=4, label="NeuroSafe-Ctx ($n=600$)")
ax.set_xlabel(r"filter correlation $\rho=\mathrm{corr}(c,\hat c)$"); ax.set_ylabel(r"$(R_n/\Delta+2V_n)/n$")
ax.legend(fontsize=5.5, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/fig_floor.pdf"); plt.close(fig)
