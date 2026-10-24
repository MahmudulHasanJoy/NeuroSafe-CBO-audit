import os, sys, time
sys.path.insert(0, os.getcwd())
import experiments.run_v3 as rv
dom, t = "lending", time.time()
K = {k: v for k, v in rv.FRONTIER_KNOBS.items() if k != "SafeOpt-ref"}
rv.run_many(dom, list(K), range(0, 10), knobs=K, tag="tuneseeds")
print(f"tuneseeds done {time.time() - t:.0f}s", flush=True)
rv.run_many(dom, ["NeuroSafeCBO", "RiskNeutral", "Oracle"], range(100, 150), tag="defaults")
print(f"defaults done {time.time() - t:.0f}s", flush=True)
abl = {k: rv.FRONTIER_KNOBS["NeuroSafe-Ctx"] for k in ["Ctx-noGLM", "Ctx-noBucket", "Ctx-noKalman", "Ctx-noContext"]}
rv.run_many(dom, list(abl), range(100, 150), knobs=abl, tag="ablation")
print(f"ablation done {time.time() - t:.0f}s", flush=True)
