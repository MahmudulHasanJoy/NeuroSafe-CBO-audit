import sys, pandas as pd
dom = sys.argv[1]
for tag in sys.argv[2:]:
    new = pd.read_csv(f"results/v3/{dom}/{tag}_episodes.csv")
    old = pd.read_csv(f"results/v3/{dom}_backup_20261024/{tag}_episodes.csv")
    for ag in sorted(new.agent.unique()):
        m = new[new.agent == ag].merge(old[old.agent == ag], on=["knob", "seed"], suffixes=("", "_old"))
        dr, dv = m.ret - m.ret_old, m.viol - m.viol_old
        print(f"{dom} {tag:9s} {ag:13s}: {len(m)} rows | ret differs {(dr.abs() > 1e-9).sum()} | "
              f"viol differs {(dv.abs() > 1e-9).sum()} | max per-knob mean shift {m.assign(d=dr).groupby('knob').d.mean().abs().max():.4f}")
