# patched_safeopt

Compatibility patch for `safeopt==0.16` (PyPI). Three files replace their counterparts in
`site-packages/safeopt/`. Changes, verified by `experiments/diff_safeopt.py`
(15 compatibility lines, 0 other changes):

- `collections` -> `collections.abc` imports (2 lines; required for Python >= 3.10)
- removed NumPy aliases `np.float` / `np.bool` -> `float` / `bool` (13 lines; required for NumPy >= 1.24)

No algorithmic change. SafeOpt-ref results reproduce exactly across environments
(SCM and IHDP `safeopt_ref_episodes.csv`: 240/240 rows identical).

Install:
    pip install safeopt==0.16
    copy patched_safeopt/*.py over <venv>/Lib/site-packages/safeopt/
