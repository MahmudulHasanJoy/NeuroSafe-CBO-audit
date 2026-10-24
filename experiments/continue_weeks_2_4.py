"""Wait for Month-1 experiment PIDs / markers, then run Weeks 2-4 postprocess.

Usage:
  python experiments/continue_weeks_2_4.py
  python experiments/continue_weeks_2_4.py --wait-only-markers   # no PID wait
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "main_track")


def _alive(pid: int) -> bool:
    try:
        # Windows: tasklist; cross-platform: os.kill(pid, 0) may fail on Windows
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return str(pid) in out and "python" in out.lower()
    except Exception:
        return False


def wait_for_python_jobs(patterns: list[str], poll_s: int = 60) -> None:
    """Poll until no python process cmdline matches any pattern."""
    print(f"[{datetime.now():%H:%M:%S}] Waiting for jobs matching {patterns} ...")
    while True:
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            # fallback: PowerShell
            ps = (
                "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
            )
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True)
        active = []
        low = out.lower()
        for pat in patterns:
            if pat.lower() in low:
                active.append(pat)
        if not active:
            print(f"[{datetime.now():%H:%M:%S}] All watched jobs finished.")
            return
        print(f"[{datetime.now():%H:%M:%S}] Still running: {active}")
        time.sleep(poll_s)


def wait_for_markers(markers: list[str], poll_s: int = 60) -> None:
    print(f"[{datetime.now():%H:%M:%S}] Waiting for markers: {markers}")
    while True:
        missing = [m for m in markers if not os.path.exists(os.path.join(ROOT, m))]
        if not missing:
            print(f"[{datetime.now():%H:%M:%S}] All markers present.")
            return
        print(f"[{datetime.now():%H:%M:%S}] Missing: {missing}")
        time.sleep(poll_s)


def run(cmd: list[str]) -> None:
    print(f"[{datetime.now():%H:%M:%S}] RUN:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def write_month1_report() -> None:
    """Aggregate week1 + scm50 + ihdp + kstudy into MONTH1_REPORT.md."""
    lines = [
        "# Month 1 Aggregate Report (Weeks 1–4)",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Headline (reframed)",
        "Constrained regret for safe-optimal causal bandits; weak dual vs Chance is a secondary SCM sanity check.",
        "",
    ]

    def load_json(rel):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def load_csv_head(rel, n=8):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return "".join(f.readlines()[: n + 1])

    w1 = load_json("results/main_track/week1/seed_stats.json")
    lines.append("## Week 1 (done)")
    if w1:
        lines.append(
            f"- Δreturn={w1.get('mean_dR'):.4f}, CI={w1.get('dR_ci95')}, "
            f"Wilcoxon p={w1.get('wilcoxon_dR_p')}, viol tied"
        )
        lines.append(
            f"- Chance return={w1.get('chance_return'):.3f}, NS={w1.get('ns_return'):.3f}, "
            f"creg Chance={w1.get('chance_creg'):.3f}, NS={w1.get('ns_creg'):.3f}"
        )
    lines.append("")

    s50 = load_json("results/main_track/scm50/seed_stats.json")
    lines.append("## Week 2 — SCM-50")
    if s50:
        lines.append(json.dumps(s50, indent=2))
        summ = load_csv_head("results/main_track/scm50/summary.csv")
        if summ:
            lines.append("```")
            lines.append(summ.rstrip())
            lines.append("```")
    else:
        lines.append("- *scm50/seed_stats.json not found yet*")
    lines.append("")

    lines.append("## Week 3 — IHDP + K-study")
    ih = load_csv_head("results/main_track/ihdp/ihdp/summary.csv") or load_csv_head(
        "results/main_track/ihdp/summary.csv"
    )
    if ih:
        lines.append("### IHDP")
        lines.append("```")
        lines.append(ih.rstrip())
        lines.append("```")
    else:
        lines.append("- IHDP summary missing")
    ks = load_csv_head("results/main_track/kstudy/kstudy/kstudy_summary.csv") or load_csv_head(
        "results/main_track/kstudy/kstudy_summary.csv"
    )
    if ks:
        lines.append("### K-study")
        lines.append("```")
        lines.append(ks.rstrip())
        lines.append("```")
    else:
        lines.append("- K-study summary missing")
    lines.append("")

    lines.append("## Week 4 — Deliverables")
    lines.append("- Proofs: `paper/supplement/proofs.tex` / `proofs.pdf`")
    lines.append("- Main-track LaTeX: `paper/main_track/neurosafe_cbo_main.tex`")
    lines.append("- Figures: `results/main_track/figures/`")
    lines.append("")
    lines.append("## Honest claims")
    lines.append("- Lead with constrained regret + Pareto safety, not dual dominance.")
    lines.append("- Report weak dual only if it holds; always report CI / p-values.")
    lines.append("- No knob fishing after frozen config.")

    path = os.path.join(OUT, "MONTH1_REPORT.md")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait-only-markers",
        action="store_true",
        help="Wait for result files instead of process names",
    )
    parser.add_argument("--no-wait", action="store_true", help="Only postprocess existing results")
    parser.add_argument("--poll", type=int, default=120)
    args = parser.parse_args()

    markers = [
        "results/main_track/scm50/seed_stats.json",
        "results/main_track/ihdp/ihdp/summary.csv",
        "results/main_track/kstudy/kstudy/kstudy_summary.csv",
    ]
    # also accept alternate layouts
    alt_markers = [
        "results/main_track/scm50/seed_stats.json",
        "results/main_track/ihdp/summary.csv",
        "results/main_track/kstudy/kstudy_summary.csv",
    ]

    if not args.no_wait:
        if args.wait_only_markers:
            # succeed if either layout appears
            print("Waiting for experiment markers ...")
            while True:
                ok_a = all(os.path.exists(os.path.join(ROOT, m)) for m in markers)
                ok_b = all(os.path.exists(os.path.join(ROOT, m)) for m in alt_markers)
                # also accept scm50 + any ihdp/kstudy path
                scm = os.path.exists(os.path.join(ROOT, "results/main_track/scm50/seed_stats.json"))
                ihdp = any(
                    os.path.exists(os.path.join(ROOT, p))
                    for p in [
                        "results/main_track/ihdp/ihdp/summary.csv",
                        "results/main_track/ihdp/summary.csv",
                    ]
                )
                kstudy = any(
                    os.path.exists(os.path.join(ROOT, p))
                    for p in [
                        "results/main_track/kstudy/kstudy/kstudy_summary.csv",
                        "results/main_track/kstudy/kstudy_summary.csv",
                    ]
                )
                if scm and ihdp and kstudy:
                    break
                if ok_a or ok_b:
                    break
                print(f"[{datetime.now():%H:%M:%S}] waiting scm={scm} ihdp={ihdp} kstudy={kstudy}")
                time.sleep(args.poll)
        else:
            wait_for_python_jobs(
                [
                    "run_main_track.py --profile scm50",
                    "run_main_track.py --profile ihdp",
                    "run_main_track.py --profile kstudy",
                ],
                poll_s=args.poll,
            )

    # Figures + report
    try:
        run([sys.executable, "experiments/make_main_track_figures.py"])
    except subprocess.CalledProcessError as e:
        print("Figure script warning:", e)

    write_month1_report()

    # Touch done marker for Month 1 gate
    done = os.path.join(OUT, "WEEKS_2_4_DONE.txt")
    with open(done, "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat() + "\n")
    print("Weeks 2-4 postprocess complete:", done)


if __name__ == "__main__":
    main()
