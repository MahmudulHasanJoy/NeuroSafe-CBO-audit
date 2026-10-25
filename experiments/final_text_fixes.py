import re
def fix(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        p = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
        n = len(p.findall(s)); s = p.sub(lambda m: new, s)
        print(f"{'fixed  ' if n else 'already'} {path}: {old[:55]}")
    open(path, "w", encoding="utf-8").write(s)
fix("paper/v3/neurosafe_scirep.tex", [
    (r"\newcommand{\coderepo}{\url{https://github.com/MahmudulHasanJoy/NeuroSafe-CBO}}",
     r"\newcommand{\coderepo}{\url{https://github.com/MahmudulHasanJoy/NeuroSafe-CBO-audit}}"),
    ("This replaces the published path,", "This replaces the path of our earlier method,"),
    ("the published predictive-coding latent", "our earlier predictive-coding latent"),
    ("the published update fitted", "our earlier update fitted")])
fix("paper/v3/figures/fig_overview.tex", [
    ("A pre-registered gate requires", "A gate specified in advance requires"),
    ("compare with Chance, SafeOpt, safe GP-UCB and an oracle.", "compare with Chance, SafeOpt, safe GP-UCB, Safe UCB-GLM and an oracle.")])
fix("experiments/make_fig_domains.py", [('"Published NeuroSafe-CBO"', '"NeuroSafe-CBO (earlier method)"')])
