"""Figure 2 of the Scientific Reports manuscript: safety-utility frontiers on all three domains.

Reads the held-out episode files written by run_v3.py / run_safeopt_ref.py / regen_*.py
(columns: agent, knob, seed, ret, viol) and writes paper/v3/figures/fig_domains.{pdf,png}.
Only held-out seeds (>= 100) are used. Error bars are 95% intervals (1.96 x s.e.m., n = 50 seeds).
Run from the repository root:  python experiments/make_fig_domains.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOMAINS = [("scm", "Nonstationary SCM"), ("lending", "Lending (German Credit)"), ("ihdp", "IHDP")]
FILES = ["frontier_episodes.csv", "safeopt_ref_episodes.csv", "defaults_episodes.csv"]
STYLE = {  # agent -> (label, colour, marker, linestyle); colour-blind-safe, also separable by marker/line
    "NeuroSafe-Ctx": ("NeuroSafe-Ctx (ours)", "#0072B2", "o", "-"),
    "Chance": ("Chance-constrained", "#E69F00", "s", "--"),
    "SafeGP": ("Safe GP-UCB", "#009E73", "^", ":"),
    "SafeOpt-ref": ("SafeOpt (reference)", "#CC79A7", "D", "-."),
}
POINTS = {"NeuroSafeCBO": ("Published NeuroSafe-CBO", "#555555", "X"), "Oracle": ("Context oracle", "#000000", "*")}
# Lending channel ceiling: output of experiments/channel_ceiling.py in the regenerated environment
# (alpha: return, violation). Update if that script is rerun.
CEILING = [(0.0725, 0.0011), (0.0900, 0.0038), (0.1100, 0.0116), (0.1325, 0.0268), (0.1650, 0.0617), (0.1900, 0.0901)]


def summarise(dom, root="results/v3"):
    parts = [pd.read_csv(os.path.join(root, dom, f)) for f in FILES if os.path.exists(os.path.join(root, dom, f))]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.seed >= 100]
    g = df.groupby(["agent", "knob"])
    out = g.agg(ret=("ret", "mean"), viol=("viol", "mean"), n=("ret", "size")).reset_index()
    out["ret_se"] = g.ret.sem(ddof=1).values
    out["viol_se"] = g.viol.sem(ddof=1).values
    return out


def plot_domains(summ, ceiling=CEILING, out="paper/v3/figures/fig_domains"):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
    for ax, (dom, title), letter in zip(axes, DOMAINS, "abc"):
        s = summ[dom]
        for ag, (lab, col, mk, ls) in STYLE.items():
            d = s[s.agent == ag].sort_values("viol")
            if d.empty:
                continue
            ax.errorbar(d.viol, d.ret, xerr=1.96 * d.viol_se, yerr=1.96 * d.ret_se, color=col, marker=mk,
                        ls=ls, lw=1.1, ms=3.5, elinewidth=0.6, capsize=1.5, label=lab, zorder=3)
        for ag, (lab, col, mk) in POINTS.items():
            d = s[s.agent == ag]
            if d.empty:
                continue
            ax.errorbar(d.viol, d.ret, xerr=1.96 * d.viol_se, yerr=1.96 * d.ret_se, color=col, marker=mk,
                        ls="none", ms=7 if mk == "*" else 5, elinewidth=0.6, capsize=1.5, label=lab, zorder=4)
        if dom == "lending" and ceiling:
            r, v = zip(*ceiling)
            ax.plot(v, r, color="#7F7F7F", ls=(0, (4, 2)), lw=1.0, label="Channel ceiling (best possible)", zorder=2)
        ax.set_xlabel("Violation rate")
        ax.set_ylabel("Return per step" if letter == "a" else "")
        ax.set_title(title, fontsize=8)
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlim(left=0)
    handles, labels = {}, {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print("wrote", out + ".pdf", "and", out + ".png")


if __name__ == "__main__":
    summ = {dom: summarise(dom) for dom, _ in DOMAINS}
    for dom, s in summ.items():
        print(dom, "agents:", ", ".join(f"{a}({k})" for a, k in s.groupby("agent").size().items()))
    plot_domains(summ)
