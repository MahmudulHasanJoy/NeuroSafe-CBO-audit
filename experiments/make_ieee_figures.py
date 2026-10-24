"""IEEE-column figures for the 4-page conference paper."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PUB = os.path.join(ROOT, "results", "day7_publish")
OUT = os.path.join(ROOT, "paper", "ieee", "figures")

NAVY = "#1B4F72"
TEAL = "#1D6A5C"
RUST = "#A03A2A"
SLATE = "#4A5560"
SAND = "#C4A35A"
INK = "#1A1A1A"


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def plot_results():
    abl = pd.read_csv(os.path.join(PUB, "ablation_table.csv"))
    seeds = pd.read_csv(os.path.join(PUB, "summary_by_seed_primary10.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.28))

    # --- (a) return vs violation ---
    ax = axes[0]
    agents = [
        ("NeuroSafe-CBO", "NeuroSafeCBO", TEAL, "D", 70),
        ("Chance", "Chance", NAVY, "o", 55),
        ("Random", "Random", SLATE, "s", 40),
        ("RiskNeutral / CVaR", "RiskNeutral", SAND, "^", 45),
        ("Greedy", "Greedy", RUST, "X", 50),
    ]
    for label, key, color, marker, size in agents:
        row = abl[abl["label"] == key].iloc[0]
        ax.scatter(
            row["mean_viol"],
            row["mean_return"],
            s=size,
            c=color,
            marker=marker,
            zorder=3,
            label=label,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean violation (log scale, ↓)")
    ax.set_ylabel("Mean return (↑)")
    ax.set_title("(a) Return–violation tradeoff")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1.2e-3, 0.85)
    ax.annotate(
        "weak dual\n(tied viol.)",
        xy=(0.00194, 28.455),
        xytext=(0.0045, 38),
        fontsize=6.5,
        color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.7),
    )

    # --- (b) per-seed delta ---
    ax = axes[1]
    dR = seeds["dR"].to_numpy()
    idx = np.arange(len(dR))
    colors = [TEAL if v >= 0 else RUST for v in dR]
    ax.bar(idx, dR, color=colors, width=0.72, edgecolor="none")
    ax.axhline(0.0, color=INK, lw=0.6)
    ax.axhline(float(np.mean(dR)), color=NAVY, lw=0.8, ls="--", label=r"mean $\Delta$=+0.180")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in seeds["seed"]])
    ax.set_xlabel("Seed")
    ax.set_ylabel(r"$\Delta$return (NeuroSafe $-$ Chance)")
    ax.set_title(r"(b) Seed-level $\Delta$return (viol. matched)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_results.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ihdp():
    ihdp = pd.read_csv(os.path.join(ROOT, "results", "day7", "summary_ihdp.csv"))
    order = ["none", "safe_optimal", "top_quartile", "max_outcome"]
    ihdp = ihdp.set_index("policy").loc[order].reset_index()
    labels = ["Never treat", "Safe-optimal", "Top quartile", "Max outcome"]

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    x = np.arange(len(ihdp))
    ax.bar(x - 0.18, ihdp["mean_util"], 0.36, color=NAVY, label="Utility")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, ihdp["mean_viol"], 0.36, color=RUST, alpha=0.85, label="Violation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Mean utility")
    ax2.set_ylabel("Mean violation")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_ihdp.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_fallback():
    """Bitmap backup of the method pipeline if TikZ is unavailable."""
    fig, ax = plt.subplots(figsize=(7.16, 1.55))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    boxes = [
        (0.15, 0.7, "Observe o_t\nPC context z_t"),
        (2.55, 0.7, "Calibrated\nharm / util"),
        (4.95, 0.7, "Chance UCB\npick a_c"),
        (7.35, 0.7, "Safe pool\np >= p(a_c)"),
        (9.75, 0.7, "Util floor then\nplay a_t or a_c"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.15,
                1.15,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=0.9,
                edgecolor=NAVY,
                facecolor="#F4F7F8",
            )
        )
        ax.text(x + 1.075, y + 0.58, text, ha="center", va="center", fontsize=7.5)
    for x in (2.30, 4.70, 7.10, 9.50):
        ax.annotate("", xy=(x, 1.25), xytext=(x - 0.15, 1.25), arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_pipeline.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    _style()
    os.makedirs(OUT, exist_ok=True)
    plot_results()
    plot_ihdp()
    plot_pipeline_fallback()
    print(f"Wrote IEEE figures to {OUT}")


if __name__ == "__main__":
    main()
