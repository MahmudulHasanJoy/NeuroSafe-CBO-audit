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
| 4+ | — | Sweep over how much the context reveals about harm (ρ 0.99→0.3): frontier gain falls +2.09 → +0.13, the safety advantage reverses below ρ=0.7, and Chance is unchanged. Results subsection plus Table S8. | DONE |
| 5a | Headline pairs α=0.15 with α=0.02 | Clarified: pairs are matched on realised violation (Chance at α=0.02 already violates at 0.063 > our 0.043). | DONE |
| 5b | Conservative bandit ignores harm | Relabelled as a reference for a different safety notion, not a competitor. | DONE |
| 5c | Contextual SafeOpt unexplained | Described from the code: the package's own contextual mode, a kernel over action and context, the context held fixed, and safe seed a=0. It was already contextual; only the description was missing. | DONE |
| 5d | Missing baselines | Safe UCB-GLM (Amani et al. design) added. SCM: ours earns more and violates less at every α≥0.05, frontier 5/5 above. IHDP: frontier 2/2 above; at equal α the two trade off. Table S10. | DONE |
| 5e | Tuning budgets 18 vs 6 | `experiments/safeopt_equal_budget.py` re-tunes SafeOpt over 18 configurations. | RUN (authors' machine) |
| 6 | Violates own constraint at small α | Calibrated range now stated: violation ≤ α for every α≥0.075 in all three domains; below that the method is a cautious default. | DONE |
| 6+ | — | Decomposition (Table S9). SCM: 91% of the excess is the 10 random warm-up rounds, and a safe-warm-up variant meets every budget with higher return. IHDP: the safest action itself violates at 0.021, so α≤0.02 leaves no room to learn. The frozen method is kept for all comparisons. | DONE |
| 7 | Scale | Scoped in Limitations. | DONE |
| 7+ | — | Optional higher-dimensional SCM. | RUN (optional) |
| M1 | "Pre-registered" | Now "specified in advance, recorded in the analysis code, not publicly registered". | DONE |
| M2 | Gate on fit half | Defence added: the in-sample choice favours passing, so failing there is conservative. | DONE |
| M2+ | — | Held-out half: fails at every α, matching the fit half (0.21 mmHg at α=0.02; at most 0.34). | DONE |
| M3 | Weak harm model | Conclusion scoped to this formulation and these covariates. | DONE |
| M4 | Confounded arrhythmia finding | Removed from the main text; moved to Supplementary Note 3 and labelled likely confounded. | DONE |
| M5 | AI and MIMIC data use | Explicit statement added, citing PhysioNet's LLM guidance. Authors must confirm it is true after checking chat histories. | AUTHORS |
| M6 | MIMIC section fit | Shortened (gate table and exploratory analysis moved to the SI). | DONE |
| P | Placeholders, URL bug, IRB wording | URL bug fixed; DOI inserted; IRB sentence uses PhysioNet's wording; acknowledgements template removed. IHDP realisation (realisation 1 of IHDP-100; 75-infant test split) and SafeOpt details filled in from the code. The 3 equations were checked against `v3.py`: the filter and posterior covariance were correct, the UCB bonus was wrong and is now corrected, and the 10-round random warm-up was missing and is now stated. Still needed: MIMIC-IV version, author roles, NeuroSafe-CBO citation year, AI statement confirmation. | DONE / AUTHORS |

## Draft response paragraphs (for the response letter)

**Theory.** We agree the proof is elementary and now say so. The weight 2Δ is not arbitrary: it is the unique weight under which declining when acting is safe and acting when it is unsafe both cost Δ. We have added Proposition 1, which prices violations at any λ≥Δ and shows that the α-chance-constrained rule is exactly the Bayes-optimal policy at λ=Δ/α, so the floor now depends explicitly on α; Theorem 1 is the case α=½. We know of no prior lower bound of this Bayes-error form for safe bandits with partially observed contexts; Kim et al. (AISTATS 2023) leave the problem-specific lower bound for noisy contexts open.

**Headline comparison.** The matched pairs are matched on realised violation measured on tuning seeds, not on nominal α. Chance's realised violation at α=0.02 is already 0.063, above ours at α=0.15 (0.043); the nominal values differ because Chance over-violates its own budget. We have made this explicit in the Methods.

**MIMIC gate on the fit half.** Evaluating the gate in-sample biases it toward passing, because the simulator is fitted to those windows; its failure there therefore strengthens the negative conclusion. We now also report the gate on the held-out half, where it fails in the same way at every α (best constrained improvement 0.21 mmHg at α=0.02, at most 0.34 mmHg), and the held-out off-policy estimate (+0.046 mmHg, 95% CI −0.182 to +0.285) agrees.

**IHDP construction.** We agree the harm channel is constructed and now treat IHDP as a test of the mechanism. To show the result is not circular, we varied how much the observed context reveals about harm (ρ from 0.99 to 0.3) with everything else fixed. The gain over the context-blind baseline falls from +2.09 to +0.13 and the safety advantage reverses, as our central claim predicts. This result could have come out otherwise.

**Small-α violations.** We located the cause. On the SCM, the excess comes almost entirely from ten random warm-up rounds; a variant that warms up with the safest action meets every budget and earns more. On IHDP the safest action itself violates at 2.1%, so a 2% budget leaves no room to learn anything. We keep the method frozen for all comparisons, report the variant as a diagnostic, and claim calibration only for α≥0.075.

**Closest prior method.** We added Safe UCB-GLM following Amani et al. (2020). On the SCM our method earns more and violates less at every α≥0.05, and on IHDP its frontier is higher at every comparable point. The design choices we describe as engineering therefore matter empirically, although we claim no algorithmic novelty.
