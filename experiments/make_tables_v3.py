"""Generate every LaTeX table in paper/v3 directly from result files."""
import json, os
T = "paper/v3/tables"; os.makedirs(T, exist_ok=True)
f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"
def p(x): return "--" if x != x else (f"{x:.1e}" if x < 1e-3 else f"{x:.3f}")
def ci(a, fmt): return f"[{fmt(a[0])}, {fmt(a[1])}]"
def sig(c): return c[0] > 0 or c[1] < 0
NAME = {"Chance": "Chance", "SafeGP": "Safe GP-UCB", "SafeOpt-ref": "SafeOpt (reference)"}
DOMS = [(d, l) for d, l in [("scm", "SCM"), ("lending", "Lending"), ("ihdp", "IHDP")]
        if os.path.exists(f"results/v3/{d}/summary.json")]
HDR = ("\\begin{tabular}{lcllll}\n\\toprule\nBaseline @knob & our $\\alpha$ & Return (base$\\to$ours) & "
       "$\\Delta$R [95\\% CI] & Violation (base$\\to$ours) & $\\Delta$V [95\\% CI] \\\\\n\\midrule\n")
for dom, lab in DOMS:
    S = json.load(open(f"results/v3/{dom}/summary.json"))
    rows = []
    for r in S["matched"]:
        if r["base"] == "Conservative":
            continue
        dR = ("\\textbf{%s}" if sig(r["dR_ci"]) else "%s") % f"{r['dR']:+.3f}"
        dV = ("\\textbf{%s}" if sig(r["dV_ci"]) else "%s") % f"{r['dV']:+.4f}"
        rows.append(f"{NAME[r['base']]} @{r['base_knob']:g} & {r['our_alpha']:g} & "
                    f"{f3(r['base_ret'])}$\\to${f3(r['new_ret'])} & {dR} {ci(r['dR_ci'], lambda x: f'{x:+.3f}')} & "
                    f"{f4(r['base_viol'])}$\\to${f4(r['new_viol'])} & {dV} {ci(r['dV_ci'], lambda x: f'{x:+.4f}')} \\\\")
    open(f"{T}/matched_{dom}.tex", "w").write(HDR + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# ablations
I = json.load(open("results/v3/frontier_interp.json"))
ab = []
for v, lab in [("Ctx-noGLM", "no GLM (Gaussian-CDF cells)"), ("Ctx-noContext", "no context ($z\\equiv0$)"),
               ("Ctx-noBucket", "no bucketed utility"), ("Ctx-noKalman", "no filter (raw channel)")]:
    cells = [lab]
    for dom in ["scm", "lending", "ihdp"]:
        S = json.load(open(f"results/v3/{dom}/summary.json"))
        r = [x for x in S["ablation"] if x["variant"] == v][0]
        it = I[dom][v]
        d = it["deltas"]
        rng = f"{min(d):+.3f}..{max(d):+.3f}" if d else "--"
        dR = ("\\textbf{%s}" if sig(r["dR_ci"]) else "%s") % f"{-r['dR']:+.3f}"
        dV = ("\\textbf{%s}" if sig(r["dV_ci"]) else "%s") % f"{-r['dV']:+.4f}"
        cells += [(f"{it['n_ours_above']}/{it['n_comparable']} ({rng})" if it['n_comparable'] else f"none (min $V$ {it['their_min_viol']:.3f})"), f"{dR} / {dV}"]
    ab.append(" & ".join(cells) + " \\\\")
open(f"{T}/ablation.tex", "w").write(
"\\begin{tabular}{lllllll}\n\\toprule\n & \\multicolumn{2}{c}{SCM} & \\multicolumn{2}{c}{Lending} & \\multicolumn{2}{c}{IHDP} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
"Variant & frontier: full above & $\\alpha{=}0.15$: $\\Delta$R / $\\Delta$V & frontier: full above & $\\alpha{=}0.15$: $\\Delta$R / $\\Delta$V & frontier: full above & $\\alpha{=}0.15$: $\\Delta$R / $\\Delta$V \\\\\n\\midrule\n"
+ "\n".join(ab) + "\n\\bottomrule\n\\end{tabular}\n")
# floor check
R = json.load(open("results/v3/floor_check.json"))
fl = [f"{r['rho']:.3f} & {r['floor']:.4f} & {r['bayes']['combo']:.4f} & {r['cbayes']['combo']:.3f} ({r['cbayes']['V']:.3f}) & {r['ours']['combo']:.3f} ({r['ours']['V']:.3f}) & {r['always_safe']['combo']:.3f} \\\\" for r in R]
open(f"{T}/floor.tex", "w").write(
"\\begin{tabular}{cccccc}\n\\toprule\n$\\rho$ & $\\arccos(\\rho)/\\pi$ & Bayes rule & $\\alpha$-constr.\\ Bayes ($V_n/n$) & NeuroSafe-Ctx ($V_n/n$) & always safe \\\\\n\\midrule\n"
+ "\n".join(fl) + "\n\\bottomrule\n\\end{tabular}\n")
print(open(f"{T}/ablation.tex").read()); print(open(f"{T}/floor.tex").read())

# ---- IHDP auto-summary (only if IHDP was run) ------------------------------
if os.path.exists("results/v3/ihdp/summary.json"):
    S = json.load(open("results/v3/ihdp/summary.json"))
    lines = ["IHDP results, generated automatically from \\texttt{results/v3/ihdp}; the authors must"
             " review and rewrite this paragraph. Paired tests at matched operating points:"]
    for r in S["matched"]:
        if r["base"] == "Conservative":
            continue
        lines.append(f"versus {NAME[r['base']]} @{r['base_knob']:g}, return {r['base_ret']:.3f}$\\to${r['new_ret']:.3f} "
                     f"($\\Delta$ {r['dR']:+.3f}, CI {ci(r['dR_ci'], lambda x: f'{x:+.3f}')}{', significant' if sig(r['dR_ci']) else ''}), "
                     f"violation {r['base_viol']:.4f}$\\to${r['new_viol']:.4f} "
                     f"($\\Delta$ {r['dV']:+.4f}, CI {ci(r['dV_ci'], lambda x: f'{x:+.4f}')}{', significant' if sig(r['dV_ci']) else ''});")
    open(f"{T}/ihdp_summary.tex", "w").write("\n".join(lines) + "\n")

# ---- flag for the limitations paragraph when the reference SafeOpt was run ---
has_ref = any(r["base"] == "SafeOpt-ref" for d, _ in DOMS
              for r in json.load(open(f"results/v3/{d}/summary.json"))["matched"])
flag = f"{T}/safeopt_ref_flag.tex"
if has_ref:
    open(flag, "w").write("SafeOpt is evaluated with the reference \\texttt{safeopt} package, whose "
                          "\\texttt{gp\\_opt.py}, \\texttt{utilities.py} and \\texttt{swarm.py} required "
                          "Python 3.11 compatibility edits (\\texttt{collections.abc} imports); the edits are "
                          "released with this paper and do not touch the algorithm, but we have not diffed them "
                          "against the released package line by line. Its GP hyperparameters were chosen on tuning "
                          "seeds from a 6-configuration grid, smaller than the 18-configuration grid given to our "
                          "method, which favours our method; the additional safe GP-UCB is our own, more "
                          "exploratory reimplementation and is not equivalent to SafeOpt.")
elif os.path.exists(flag):
    os.remove(flag)
