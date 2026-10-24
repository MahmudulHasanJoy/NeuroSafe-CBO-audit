"""Ctx-BetaCell ablation: a calibrated, non-GLM harm model.

Identical cell structure to GaussCellHarm (same context buckets, same
shrinkage toward the per-action pooled estimate with strength 4, same
default prior mean 0.08). Only the likelihood changes: the indicator
1{H > tau} is modelled as Bernoulli, and the feasibility statistic is the
upper posterior quantile of a Beta distribution at Phi(beta_h), matching
the GLM's one-sided bound with beta_h = 1.0 (quantile 0.8413).
Fixed before any run; not tuned.
"""
import numpy as np
from scipy.stats import beta as _beta, norm as _norm


class BetaCellHarm:
    def __init__(self, K, tau, beta_h=1.0, m=4.0, p_default=0.08):
        self.K, self.tau, self.m, self.p0 = K, float(tau), float(m), float(p_default)
        self.q = float(_norm.cdf(beta_h))
        self.rec = []

    def update(self, idx, zs, H):
        self.rec.append((idx, zs, float(H > self.tau)))

    def p_viol(self, zs, edges):
        if not self.rec:
            return np.full(self.K, 0.5)
        R = np.asarray(self.rec, float)
        idx = R[:, 0].astype(int)
        b_all = np.searchsorted(edges, R[:, 1])
        b = int(np.searchsorted(edges, zs))
        out = np.empty(self.K)
        for k in range(self.K):
            pool = R[idx == k, 2]
            cell = R[(idx == k) & (b_all == b), 2]
            p_pool = (pool.sum() + 2.0 * self.p0) / (len(pool) + 2.0)
            a = self.m * p_pool + 0.5 + cell.sum()
            bb = self.m * (1.0 - p_pool) + 0.5 + len(cell) - cell.sum()
            out[k] = float(_beta.ppf(self.q, a, bb))
        return out
