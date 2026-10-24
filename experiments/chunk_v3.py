import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_v3 import run_many, FRONTIER_KNOBS, DEFAULT_KNOB
dom, agent, tag, s0, s1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
key = "NeuroSafe-Ctx" if agent.startswith("Ctx-") else agent
knobs = {agent: FRONTIER_KNOBS[key]} if key in FRONTIER_KNOBS and tag != "defaults" else None
run_many(dom, [agent], range(s0, s1), knobs=knobs, tag=tag)
