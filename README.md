# NeuroSafe-CBO: auditing safe-optimal decision-making

Code, result files and scripts for the manuscript

> **Context-conditional feasibility improves safe sequential decisions only as far as the observed context reveals harm.**
> M. M. Hasan, A. Subah and S. M. Hussain. Submitted to *Scientific Reports*.

The repository contains the audited published method (NeuroSafe-CBO), the context-conditional rule
NeuroSafe-Ctx (`neurosafe_cbo/v3.py`), all baselines, the three evaluation environments (nonstationary
SCM, German Credit lending, IHDP), the MIMIC-IV audit code, and every result file behind the paper's
tables and figures.

## Layout

| Path | Contents |
|---|---|
| `neurosafe_cbo/` | Method, baselines and environments |
| `experiments/` | One script per step of the pipeline (see below) |
| `results/v3/` | Episode-level results (`*_episodes.csv`) and summaries for each domain |
| `results/v3/mimic/audit.json` | MIMIC-IV audit: aggregate statistics only |
| `paper/v3/` | Manuscript source, Supplementary Information, tables and figures |
| `patched_safeopt/` | 15-line compatibility patch for `safeopt==0.16` (see its README) |
| `PROJECT_MAP.md` | Plain-language map of the project |

## Setup

Python 3.11.9. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the exact versions used for every reported result. Then apply the SafeOpt
compatibility patch as described in `patched_safeopt/README.md` (copy the three files over the installed
`safeopt` package). The patch changes only imports and NumPy type aliases;
`python experiments/diff_safeopt.py` verifies this.

## Reproduce the paper

Run from the repository root, in order (each command prints what it wrote):

```bash
python experiments/run_v3.py frontier --domain scm --seeds 100:150 --agents "NeuroSafe-Ctx,Chance,SafeGP,Conservative"  # also lending, ihdp
python experiments/regen_domain.py scm       # tuning seeds, defaults, ablations; also ihdp
python experiments/regen_lending.py          # the same for lending
python experiments/run_safeopt_ref.py --domain scm           # also lending, ihdp
python experiments/run_ablation2.py
python experiments/swap_ihdp_ablation.py
python experiments/analyze_v3.py scm         # each domain
python experiments/channel_ceiling.py
python experiments/verify_theorem.py
python experiments/verify_floor.py
python experiments/figures_v3.py
python experiments/make_fig_domains.py
python experiments/make_tables_v3.py
python experiments/make_posthoc_betacell.py
python experiments/mimic_audit.py            # requires credentialed MIMIC-IV access
python experiments/make_mimic_tables.py
```

Tuning uses seeds 0-9; every reported number uses held-out seeds 100-149.

## Data

* **German Credit** (Statlog), UCI Machine Learning Repository, CC BY 4.0, https://doi.org/10.24432/C5NC77 — included under `data/german_credit/`.
* **IHDP**: public semi-synthetic benchmark (Hill, 2011).
* **MIMIC-IV**: available only to credentialed users through PhysioNet. It is **not** included, and no
  patient-level or derived data are distributed. To rerun the audit, obtain access, place the files
  locally, and run `experiments/build_mimic_cohort.py` followed by `experiments/mimic_audit.py`.

## Use of AI tools

AI assistants were used to write and debug code, run experiments, generate tables and figures, and
draft text; see the Methods section of the paper. The authors are responsible for every result.

## License

See `LICENSE`.
