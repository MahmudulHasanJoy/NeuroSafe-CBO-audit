"""Figures for A* main-track Month 1 results."""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def plot_regret_curve(regret_dir: str, out_dir: str):
    path = os.path.join(regret_dir, "regret_curve.csv")
    if not os.path.exists(path):
        print("Skip regret curve:", path, "missing")
        return
    df = pd.read_csv(path)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for name, g in df.groupby("agent"):
        g = g.sort_values("step")
        plt.plot(g["step"], g["mean_cum_regret"], label=name)
    plt.xlabel("Step")
    plt.ylabel("Mean cumulative constrained regret")
    plt.title("Constrained regret learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regret_curve_maintrack.png"), dpi=180)
    plt.close()


def plot_kstudy(k_dir: str, out_dir: str):
    path = os.path.join(k_dir, "kstudy_summary.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for name, g in df.groupby("agent"):
        g = g.sort_values("k")
        axes[0].plot(g["k"], g["mean_return"], marker="o", label=name)
        axes[1].plot(g["k"], g["mean_creg"], marker="o", label=name)
    axes[0].set_xlabel("K (discretization)")
    axes[0].set_ylabel("Mean return")
    axes[0].set_title("Return vs K")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Mean constrained regret")
    axes[1].set_title("Regret vs K")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "kstudy_maintrack.png"), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(ROOT, "results", "main_track"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "main_track", "figures"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for regret_dir in glob.glob(os.path.join(args.input, "**", "regret"), recursive=True):
        plot_regret_curve(regret_dir, args.out)
    for k_dir in glob.glob(os.path.join(args.input, "**", "kstudy"), recursive=True):
        plot_kstudy(k_dir, args.out)
    print("Figures ->", args.out)


if __name__ == "__main__":
    main()
