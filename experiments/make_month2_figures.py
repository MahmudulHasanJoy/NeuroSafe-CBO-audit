"""Publication figures for Month-2 main-track paper."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MT = os.path.join(ROOT, "results", "main_track")
OUT = os.path.join(MT, "figures", "month2")

NAVY, TEAL, RUST, SLATE, SAND = "#1B4F72", "#1D6A5C", "#A03A2A", "#4A5560", "#C4A35A"


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_pareto_scm50():
    s = pd.read_csv(os.path.join(MT, "scm50", "summary.csv"))
    colors = {
        "NeuroSafeCBO": TEAL,
        "Chance": NAVY,
        "Random": SLATE,
        "RiskNeutral": SAND,
        "CVaR": SAND,
        "Greedy": RUST,
    }
    markers = {
        "NeuroSafeCBO": "D",
        "Chance": "o",
        "Random": "s",
        "RiskNeutral": "^",
        "CVaR": "v",
        "Greedy": "X",
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for _, r in s.iterrows():
        name = r["agent"]
        ax.scatter(
            r["mean_viol"],
            r["mean_return"],
            c=colors.get(name, SLATE),
            marker=markers.get(name, "o"),
            s=70,
            label=name,
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean violation (log, ↓)")
    ax.set_ylabel("Mean return (↑)")
    ax.set_title("SCM-50: safety–utility Pareto")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pareto_scm50.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_pareto_scm50.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_regret_week1():
    path = os.path.join(MT, "week1", "regret", "regret_curve.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, g in df.groupby("agent"):
        g = g.sort_values("step")
        ax.plot(g["step"], g["mean_cum_regret"], label=name, lw=1.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean cumulative constrained regret")
    ax.set_title("Week-1 constrained regret vs horizon")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_regret_week1.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_regret_week1.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_kstudy():
    path = os.path.join(MT, "kstudy", "kstudy", "kstudy_summary.csv")
    if not os.path.exists(path):
        path = os.path.join(MT, "kstudy", "kstudy_summary.csv")
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for name, g in df.groupby("agent"):
        g = g.sort_values("k")
        axes[0].plot(g["k"], g["mean_return"], marker="o", label=name)
        axes[1].plot(g["k"], g["mean_creg"], marker="o", label=name)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Mean return")
    axes[0].set_title("Return vs discretization")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Mean constrained regret")
    axes[1].set_title("Regret vs discretization")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_kstudy.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_kstudy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_ihdp():
    path = os.path.join(MT, "ihdp", "ihdp", "summary.csv")
    if not os.path.exists(path):
        path = os.path.join(MT, "ihdp", "summary.csv")
    df = pd.read_csv(path)
    order = ["Chance", "NeuroSafeCBO", "Random", "RiskNeutral", "Greedy"]
    df = df.set_index("agent").reindex([a for a in order if a in df["agent"].values]).reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["mean_return"], 0.36, color=NAVY, label="Return")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, df["mean_viol"], 0.36, color=RUST, alpha=0.85, label="Violation")
    ax.set_xticks(x)
    ax.set_xticklabels(df["agent"], rotation=20, ha="right")
    ax.set_ylabel("Mean return")
    ax2.set_ylabel("Mean violation")
    ax.set_title("IHDP sequential bandit")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ihdp.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_ihdp.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    _style()
    os.makedirs(OUT, exist_ok=True)
    fig_pareto_scm50()
    fig_regret_week1()
    fig_kstudy()
    fig_ihdp()
    print("Month-2 figures ->", OUT)


if __name__ == "__main__":
    main()
