# Revision log and draft response to the referee

Status key: **DONE** = changed in this revision · **RUN** = needs an experiment on the authors' machine · **AUTHORS** = needs information only the authors have.

| # | Referee point | What we did | Status |
|---|---|---|---|
| 1a | Theorem near-trivial; 2Δ reverse-engineered | Text now says the proof is elementary and that the contribution is the framing and the attained constant. Added the explanation that 2Δ is the unique weight under which both mistakes cost Δ. | DONE |
| 1b | α plays no role | New Proposition 1: for any price λ≥Δ on violations the floor holds, and the α-chance-constrained rule is exactly optimal at λ=Δ/α (Theorem 1 is α=½). Gaussian closed form given; verified by simulation; Table S6 numbers agree with it to rounding. | DONE |
| 1c | Learner barely beats always-safe | Now stated plainly: at its own price it beats always-declining only for ρ≥0.812; at ρ≤0.707 even the optimal rule is within 15% of always-declining. | DONE |
| 1d | No upper bound | Stated as a limitation (unchanged). | DONE |
| 2 | Method novelty minimal | Positioned as a Bayesian variant of safe GLM bandits (Amani et al. 2020) with forgetting (Russac et al. 2021); no algorithmic novelty claimed; differences listed. | DONE |
| 3a | NeuroSafe-CBO undisclosed | Now "our own earlier method", cited. Citation needs the year or the earlier paper/preprint. | DONE / AUTHORS |
| 3b | "Four benchmarks" | Now "four defects in three environments, built or adapted by us". | DONE |
| 3c | Draft-history phrases | All removed. | DONE |
| 4 | IHDP result built in | Reframed as a mechanism test. | DONE |
| 4+ | — | Dose-response sweep over how much the context reveals about harm. | RUN |
| 5a | Headline pairs α=0.15 with α=0.02 | Clarified: pairs are matched on realised violation (Chance at α=0.02 already violates at 0.063 > our 0.043). | DONE |
| 5b | Conservative bandit ignores harm | Relabelled as a reference for a different safety notion, not a competitor. | DONE |
| 5c | Contextual SafeOpt unexplained | Placeholder in Baselines; needs `safeopt_ref.py`. | AUTHORS / RUN |
| 5d | Missing baselines; 18 vs 6 tuning | Safe UCB-GLM baseline and equal-budget re-tuning. | RUN |
| 6 | Violates own constraint at small α | Calibrated range now stated: violation ≤ α for every α≥0.075 in all three domains; below that the method is a cautious default. | DONE |
| 6+ | — | Oracle-feasibility check at α≤0.05. | RUN |
| 7 | Scale | Scoped in Limitations. | DONE |
| 7+ | — | Optional higher-dimensional SCM. | RUN (optional) |
| M1 | "Pre-registered" | Now "specified in advance, recorded in the analysis code, not publicly registered". | DONE |
| M2 | Gate on fit half | Defence added: the in-sample choice favours passing, so failing there is conservative. | DONE |
| M2+ | — | Held-out-half result from `experiments/mimic_gate_holdout.py`. | RUN |
| M3 | Weak harm model | Conclusion scoped to this formulation and these covariates. | DONE |
| M4 | Confounded arrhythmia finding | Removed from the main text; moved to Supplementary Note 3 and labelled likely confounded. | DONE |
| M5 | AI and MIMIC data use | Explicit statement added, citing PhysioNet's LLM guidance. Authors must confirm it is true after checking chat histories. | AUTHORS |
| M6 | MIMIC section fit | Shortened (gate table and exploratory analysis moved to the SI). | DONE |
| P | Placeholders, URL bug, IRB wording | URL bug fixed; DOI inserted; IRB sentence uses PhysioNet's wording; acknowledgements template removed. Still needed: MIMIC-IV version, IHDP realisation, author roles, 3 equations checked against `v3.py`. | DONE / AUTHORS |

## Draft response paragraphs (for the response letter)

**Theory.** We agree the proof is elementary and now say so. The weight 2Δ is not arbitrary: it is the unique weight under which declining when acting is safe and acting when it is unsafe both cost Δ. We have added Proposition 1, which prices violations at any λ≥Δ and shows that the α-chance-constrained rule is exactly the Bayes-optimal policy at λ=Δ/α, so the floor now depends explicitly on α; Theorem 1 is the case α=½. We know of no prior lower bound of this Bayes-error form for safe bandits with partially observed contexts; Kim et al. (AISTATS 2023) leave the problem-specific lower bound for noisy contexts open.

**Headline comparison.** The matched pairs are matched on realised violation measured on tuning seeds, not on nominal α. Chance's realised violation at α=0.02 is already 0.063, above ours at α=0.15 (0.043); the nominal values differ because Chance over-violates its own budget. We have made this explicit in the Methods.

**MIMIC gate on the fit half.** Evaluating the gate in-sample biases it toward passing, because the simulator is fitted to those windows; its failure there therefore strengthens the negative conclusion. We now also report the gate on the held-out half [result], and the held-out off-policy estimate (+0.046 mmHg, 95% CI −0.182 to +0.285) agrees.
