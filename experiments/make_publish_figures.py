"""Generate publication figures from frozen day7_publish artifacts."""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "results", "day7_publish", "figures")
PUB = os.path.join(ROOT, "results", "day7_publish")


def plot_method_flow(out_dir: str):
    """Decision-flow schematic for NeuroSafe-CBO (Chance-anchored path)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 2.8, "Observe\n$o_t$"),
        (2.0, 2.8, "PC context\n$z_t$"),
        (3.7, 2.8, "Calibrated\nharm $H$"),
        (5.4, 2.8, "Chance UCB\npick $a_c$"),
        (7.1, 2.8, "Safe pool\np_safe >= p_c"),
        (8.6, 2.8, "Util floor\nmu_U >= mu_U(a_c)+eps"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 1.3, 1.0, fill=False, lw=1.5))
        ax.text(x + 0.65, y + 0.5, text, ha="center", va="center", fontsize=8)

    ax.annotate("", xy=(2.0, 3.3), xytext=(1.6, 3.3), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(3.7, 3.3), xytext=(3.3, 3.3), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(5.4, 3.3), xytext=(5.0, 3.3), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(7.1, 3.3), xytext=(6.7, 3.3), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(8.6, 3.3), xytext=(8.2, 3.3), arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.add_patch(plt.Rectangle((7.8, 0.8), 2.0, 1.0, fill=False, lw=1.5, linestyle="--"))
    ax.text(8.8, 1.3, "Play $a_c$\n(empty pool)", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(8.8, 1.8), xytext=(9.2, 2.8), arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="dashed"))

    ax.add_patch(plt.Rectangle((7.8, 4.0), 2.0, 0.7, fill=False, lw=1.5))
    ax.text(8.8, 4.35, "Play argmax util\n(+ PC mix)", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(8.8, 4.0), xytext=(9.2, 3.8), arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.set_title("NeuroSafe-CBO: Chance-anchored optimal-among-safe rule")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "method_flow.png"), dpi=180)
    plt.close(fig)


def plot_dual_bars(out_dir: str):
    abl = pd.read_csv(os.path.join(PUB, "ablation_table.csv"))
    focus = abl[abl["label"].isin(["Chance", "NeuroSafeCBO", "Greedy", "Random"])].copy()
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    x = np.arange(len(focus))
    ax[0].bar(x, focus["mean_return"], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(focus["label"], rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("Mean return")
    ax[0].set_title("Return (↑)")
    ax[1].bar(x, focus["mean_viol"], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(focus["label"], rotation=20, ha="right", fontsize=8)
    ax[1].set_ylabel("Mean violation")
    ax[1].set_title("Violation (↓)")
    fig.suptitle("Primary dual grid (10 seeds × 20 eps × 4 cells)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dual_primary10.png"), dpi=180)
    plt.close(fig)


def plot_ablation(out_dir: str):
    abl = pd.read_csv(os.path.join(PUB, "ablation_table.csv"))
    ns = abl[abl["label"].str.startswith("NS") | (abl["label"] == "NeuroSafeCBO")].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ns))
    width = 0.35
    ax.bar(x - width / 2, ns["mean_return"], width, label="Return", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, ns["mean_viol"] * 1000, width, label="Viol (×10³)", color="#C44E52", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ns["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean return")
    ax2.set_ylabel("Mean violation (×10³)")
    ax.set_title("NeuroSafe ablations vs Chance (dual_weak flag in CSV)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablations.png"), dpi=180)
    plt.close(fig)


def plot_seed_delta(out_dir: str):
    path = os.path.join(PUB, "summary_by_seed_primary10.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "dR" not in df.columns:
        ch = df[df["agent"] == "Chance"].set_index("seed")["return"]
        ns = df[df["agent"] == "NeuroSafeCBO"].set_index("seed")["return"]
        dR = (ns - ch).dropna()
    else:
        dR = df.groupby("seed")["dR"].first() if "seed" in df.columns else df["dR"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(dR)), dR.values, color="#55A868")
    ax.axhline(0, color="k", lw=0.8)
    stats = json.loads(open(os.path.join(PUB, "seed_stats_primary10.json")).read())
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Δreturn (NeuroSafe − Chance)")
    ax.set_title(f"Per-seed Δreturn (mean={stats['mean_dR']:.3f}, Wilcoxon p={stats['wilcoxon_dR_greater']['pvalue']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "seed_delta_primary10.png"), dpi=180)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    plot_method_flow(OUT)
    plot_dual_bars(OUT)
    plot_ablation(OUT)
    plot_seed_delta(OUT)
    print(f"Figures saved to {OUT}")


if __name__ == "__main__":
    main()
