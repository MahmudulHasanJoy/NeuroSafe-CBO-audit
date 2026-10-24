"""Machine checks of Theorem 1 / Corollary 1. These catch arithmetic and
modelling errors; they are NOT a substitute for a human checking the proof.

(a) the pathwise identity  R_n + 2*Delta*V_n = Delta*M_n  on all four cases;
(b) Sheppard's formula  P(sign c != sign o) = arccos(rho)/pi  by Monte Carlo;
(c) the corollary's bound against 7 policy families, incl. randomised and
    feedback-adaptive ones -- none may fall below the floor;
(d) Theorem 1 itself (not the corollary) with AUTOCORRELATED contexts, where
    the floor is sum_t E[beta_t] computed from each round's Kalman posterior.
"""
import json, os, sys
import numpy as np
from scipy.stats import norm
rng = np.random.RandomState(0); D = 1.0; out = {}

# (a)
ok = True
for a in (0, 1):
    for c in (-1.0, 1.0):
        astar = 1 if c <= 0 else 0
        R = D * (astar - a); V = 1 if (a == 1 and c > 0) else 0; M = int(a != astar)
        ok &= abs(R + 2 * D * V - D * M) < 1e-12
out["a_identity_all_cases"] = bool(ok); print("(a) identity holds in all 4 cases:", ok)

# (b)
worst = 0.0
for rho in np.linspace(0.05, 0.99, 20):
    s = np.sqrt(1 / rho**2 - 1); n = 400000
    c = rng.normal(size=n); o = c + rng.normal(0, s, n)
    worst = max(worst, abs(np.mean(np.sign(c) != np.sign(o)) - np.arccos(rho) / np.pi))
out["b_sheppard_max_abs_err"] = worst; print(f"(b) Sheppard max |MC - formula| over 20 rho: {worst:.5f}")

# (c)
def run(policy, s, n=4000, reps=30):
    vals = []
    for r in range(reps):
        g = np.random.RandomState(1000 + r); th = 0.0; hist = []
        tot = 0.0
        for t in range(n):
            c = g.normal(); o = c + g.normal(0, s)
            a = policy(o, g, th, hist)
            H = 1 if (a == 1 and c > 0) else 0
            astar = 1 if c <= 0 else 0
            tot += (astar - a) + 2 * H
            hist.append((o, a, H))
            if policy.__name__ == "adaptive":           # lowers its threshold after each violation
                th = th - 0.05 if H else min(th + 0.002, 0.5)
        vals.append(tot / n)
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(reps))
def bayes(o, g, th, h): return 1 if o <= 0 else 0
def thr_plus(o, g, th, h): return 1 if o <= 0.3 else 0
def thr_minus(o, g, th, h): return 1 if o <= -0.5 else 0
def randomised(o, g, th, h): return int(g.rand() < norm.cdf(-2 * o))
def coin(o, g, th, h): return int(g.rand() < 0.5)
def adaptive(o, g, th, h): return 1 if o <= th else 0
def always_risky(o, g, th, h): return 1
viol = []
for s in (0.4, 1.0):
    rho = 1 / np.sqrt(1 + s * s); fl = np.arccos(rho) / np.pi
    print(f"(c) rho={rho:.3f} floor={fl:.4f}")
    for p in (bayes, thr_plus, thr_minus, randomised, coin, adaptive, always_risky):
        m, se = run(p, s, reps=30 if p is not adaptive else 10)
        below = m < fl - 3 * se
        viol.append(below)
        print(f"    {p.__name__:13s} {m:.4f} +- {se:.4f} {'BELOW FLOOR' if below else ''}")
out["c_any_policy_below_floor"] = bool(any(viol))

# (d) AR(1) contexts, observations only; floor = mean_t Phi(-|m_t|/sqrt(P_t))
phi, q, r = 0.75, 1 - 0.75**2, 0.5
res = {}
for name in ("kalman_bayes", "raw_obs", "stale"):
    tot = beta = 0.0; n = 0
    for rep in range(20):
        g = np.random.RandomState(50 + rep); c = g.normal(); m, P = 0.0, 1.0; prev = 0.0
        for t in range(3000):
            c = phi * c + np.sqrt(q) * g.normal(); o = c + np.sqrt(r) * g.normal()
            mp, Pp = phi * m, phi**2 * P + q; K = Pp / (Pp + r); m, P = mp + K * (o - mp), (1 - K) * Pp
            beta += norm.cdf(-abs(m) / np.sqrt(P))
            a = {"kalman_bayes": int(m <= 0), "raw_obs": int(o <= 0), "stale": int(prev <= 0)}[name]
            prev = o
            tot += int(a != (1 if c <= 0 else 0)); n += 1
    res[name] = dict(mistakes=tot / n, floor=beta / n)
    print(f"(d) AR(1) {name:12s} mistakes/round {tot/n:.4f}  vs  Theorem-1 floor {beta/n:.4f}")
out["d_ar1"] = res
out["d_ok"] = all(v["mistakes"] >= v["floor"] - 0.005 for v in res.values())
os.makedirs("results/v3", exist_ok=True); json.dump(out, open("results/v3/theorem_checks.json", "w"), indent=2)
print("ALL CHECKS PASSED" if (out["a_identity_all_cases"] and not out["c_any_policy_below_floor"] and out["d_ok"] and worst < 0.005) else "A CHECK FAILED")
