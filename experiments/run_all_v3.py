"""Run the complete v3 protocol for one domain.
  tuning seeds 0-9   : frontier (used ONLY to choose matched alphas)
  held-out 100-149   : frontier + defaults + ablations + oracle
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_v3 import run_many, FRONTIER_KNOBS
dom = sys.argv[1]
t = time.time()
FRONTIER_KNOBS = {k: v for k, v in FRONTIER_KNOBS.items() if k != "SafeOpt-ref"}
run_many(dom, list(FRONTIER_KNOBS), range(0, 10), knobs=FRONTIER_KNOBS, tag="tuneseeds")
run_many(dom, list(FRONTIER_KNOBS), range(100, 150), knobs=FRONTIER_KNOBS, tag="frontier")
extra = ["NeuroSafeCBO", "RiskNeutral"] + (["Oracle"] if dom in ("lending", "ihdp") else [])
run_many(dom, extra, range(100, 150), tag="defaults")
abl = {k: FRONTIER_KNOBS["NeuroSafe-Ctx"] for k in ["Ctx-noGLM", "Ctx-noBucket", "Ctx-noKalman"]}
run_many(dom, list(abl), range(100, 150), knobs=abl, tag="ablation")
print("total", time.time() - t)
