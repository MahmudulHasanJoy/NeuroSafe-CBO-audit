"""Best achievable policy for ANY agent that sees only the lending context
channel c = mean(noisy risk_score, noisy duration) -- an information ceiling."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurosafe_cbo.env_lending_bandit import _load_and_fit
L = _load_and_fit(); m = L["model"]; X = L["covariates"]
A = np.linspace(0, 1, 11); tau = 0.25; rng = np.random.RandomState(0)
P = np.stack([m.predict_proba(np.column_stack([X[:, 0], X[:, 1], np.full(len(X), a)]))[:, 1] for a in A], 1)
P[:, 0] = 0.0
N = 200000
idx = rng.randint(0, len(X), N)
c = (X[idx] + rng.normal(0, 0.05, (N, 2))).mean(1)
viol = (P[idx] + rng.normal(0, 0.01, (N, 11))) > tau          # N x 11
edges = np.quantile(c, np.linspace(0, 1, 41)[1:-1]); b = np.searchsorted(edges, c)
out = {}
for alpha in [0.02, 0.05, 0.10, 0.15, 0.25, 0.35]:
    R = V = 0.0
    for k in range(40):
        s = b == k
        pv = viol[s].mean(0); ok = pv <= alpha
        j = int(np.where(ok)[0].max())       # a=0 always feasible; utility = a
        R += s.sum() * A[j]; V += viol[s, j].sum()
    out[alpha] = dict(ret=R / N, viol=V / N)
    print(f"channel-ceiling alpha={alpha}: return {R/N:.4f}  violation {V/N:.4f}")
# how much feasibility information does c carry? corr of c with max-feasible action under full info
full = np.array([np.where(r <= tau - 0.0104)[0].max() for r in P])[idx] / 10.0
print("corr(c, full-info max feasible a):", np.corrcoef(c, full)[0, 1])
json.dump(out, open("results/v3/lending/channel_ceiling.json", "w"), indent=2)
