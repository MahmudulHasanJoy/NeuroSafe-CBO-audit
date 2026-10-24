import os, sys, time
sys.path.insert(0, os.getcwd())
import experiments.run_v3 as rv
dom = sys.argv[1]; t = time.time()
K = {k: v for k, v in rv.FRONTIER_KNOBS.items() if k != "SafeOpt-ref"}
rv.run_many(dom, list(K), range(0, 10), knobs=K, tag="tuneseeds")
print(f"{dom} tuneseeds done {time.time() - t:.0f}s", flush=True)
extra = ["NeuroSafeCBO", "RiskNeutral"] + (["Oracle"] if dom in ("lending", "ihdp") else [])
rv.run_many(dom, extra, range(100, 150), tag="defaults")
print(f"{dom} defaults done {time.time() - t:.0f}s", flush=True)
if dom == "scm":
    abl = {k: rv.FRONTIER_KNOBS["NeuroSafe-Ctx"] for k in ["Ctx-noGLM", "Ctx-noBucket", "Ctx-noKalman", "Ctx-noContext"]}
    rv.run_many(dom, list(abl), range(100, 150), knobs=abl, tag="ablation")
    print(f"{dom} ablation done {time.time() - t:.0f}s", flush=True)
