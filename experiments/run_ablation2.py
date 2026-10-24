import os, sys, time
import pandas as pd
sys.path.insert(0, os.getcwd())
import experiments.run_v3 as rv
AGENTS = ["NeuroSafe-Ctx", "Ctx-noContext", "Ctx-BetaCell", "Ctx-noGLM", "Ctx-noBucket", "Ctx-noKalman"]
KNOBS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35]
OUT = "results/v3/ihdp/ablation2_episodes.csv"
rows, t0 = [], time.time()
for name in AGENTS:
    for k in KNOBS:
        for s in range(100, 150):
            r = rv.episode("ihdp", name, s, k)
            row = dict(domain="ihdp", agent=name, knob=k, seed=s)
            row.update(r if isinstance(r, dict) else dict(ret=r[0], viol=r[1]))
            rows.append(row)
            if len(rows) == 1:
                print("first row:", row, flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{name:14s} a={k:<6} done ({time.time() - t0:.0f}s)", flush=True)
print("wrote", OUT)
