"""Single declared protocol for the context-recovery diagnostic:
decoupled SCM, shift 0.35, obs noise 0.15, 30 seeds x 400 steps, uniform-random
actions, |Pearson corr| with the latent confounder C pooled over all steps."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurosafe_cbo.env import NonstationarySCMEnv
from neurosafe_cbo.models import PredictiveCoder
from neurosafe_cbo.config import CFG
from neurosafe_cbo.v3 import AdaptiveKalman
C, R, P, K = [], [], [], []
for s in range(30):
    env = NonstationarySCMEnv(200, 0.35, 0.15, seed=5000 + s, harm_mode="decoupled"); rng = np.random.RandomState(s)
    obs = env.reset(); pc = PredictiveCoder(dict(CFG), s); kf = AdaptiveKalman(); h = 0.0
    for t in range(400):
        R.append(obs[0]); P.append(pc.infer(obs[0], obs[3], h)); K.append(kf.infer(obs[0]))
        obs, u, hh, _, _ = env.step(float(rng.choice(np.linspace(0, 1, 11)))); h = 0.95 * h + 0.05 * hh
        C.append(env.C)   # latent after the step is what generated this obs? use pre-step latent instead:
C = np.array(C); R, P, K = map(np.array, (R, P, K))
# obs at step t is generated from C BEFORE step t; align by shifting C
Cpre = np.concatenate([[np.nan], C[:-1]])
m = ~np.isnan(Cpre)
# reset boundaries: first obs of each seed comes from reset C; drop those rows
m[::400] = False
out = {k: float(abs(np.corrcoef(v[m], Cpre[m])[0, 1])) for k, v in [("raw", R), ("pc_published", P), ("kalman_adaptive", K)]}
print(out); json.dump(out, open("results/v3/context_recovery.json", "w"), indent=2)
