"""Reference SafeOpt baseline (Sui et al., 2015) via the `safeopt` package.

Requires:  pip install safeopt GPy        (not available offline; run on your machine)

Contextual SafeOpt with two GPs over (action, context):
  * utility GP  -- objective to maximise
  * safety GP   -- models s = -H; the constraint is s >= -tau  (i.e. H <= tau)
The safe seed is a = 0, which is the safe fallback in every domain here
(lowest-risk dose in the SCM, "decline" in lending, "no treatment" in IHDP).
SafeOpt needs a safe seed rather than a random warm-up; that is its design.

Context enters the GP inputs scaled by a per-domain constant (`ctx_scale`)
estimated by the harness from contexts of a separate environment seed, so
the GP lengthscale on context is comparable across domains.

This file was written without the package installed (no network in the
authoring environment). Run `python experiments/run_safeopt_ref.py --smoke`
first; it exercises every API call on 1 seed x 30 steps.
"""
from __future__ import annotations

import numpy as np


class SafeOptRef:
    def __init__(self, actions, seed=0, tau=0.2, beta=2.0, ctx_scale=1.0,
                 ls_a=0.2, ls_z=1.0, u_var=1.0, u_noise=0.25, h_var=0.1, h_noise=0.01):
        import GPy
        import safeopt
        self._GPy, self._safeopt = GPy, safeopt
        np.random.seed(seed)
        self.actions = np.asarray(actions, dtype=float)
        self.tau, self.beta, self.cs = float(tau), float(beta), float(ctx_scale)
        self.hp = dict(ls_a=ls_a, ls_z=ls_z, u_var=u_var, u_noise=u_noise,
                       h_var=h_var, h_noise=h_noise)
        self.opt = None
        self.fallbacks = 0
        self.t = 0

    def _build(self, z, U, H):
        GPy, safeopt, hp = self._GPy, self._safeopt, self.hp
        X0 = np.array([[0.0, z]])
        ku = GPy.kern.RBF(input_dim=2, variance=hp["u_var"], lengthscale=[hp["ls_a"], hp["ls_z"]], ARD=True)
        ks = GPy.kern.RBF(input_dim=2, variance=hp["h_var"], lengthscale=[hp["ls_a"], hp["ls_z"]], ARD=True)
        gp_u = GPy.models.GPRegression(X0, np.array([[U]]), ku, noise_var=hp["u_noise"])
        gp_s = GPy.models.GPRegression(X0, np.array([[-H]]), ks, noise_var=hp["h_noise"])
        self.opt = safeopt.SafeOpt([gp_u, gp_s], self.actions.reshape(-1, 1),
                                   fmin=[-np.inf, -self.tau], beta=self.beta, num_contexts=1)

    def act(self, obs):
        self.t += 1
        z = float(obs[0]) / self.cs
        self._z = z
        if self.opt is None:  # first round: play the safe seed; GP built in observe()
            return 0.0, z
        try:
            x = self.opt.optimize(context=np.array([z]))
            return float(np.asarray(x).ravel()[0]), z
        except Exception:  # SafeOpt raises when the safe set is empty in this context
            self.fallbacks += 1
            return 0.0, z

    def observe(self, a, z, U, H):
        if self.opt is None:
            self._build(z, float(U), float(H))
            return
        self.opt.add_new_data_point(np.array([float(a)]), np.array([float(U), -float(H)]),
                                    context=np.array([z]))
