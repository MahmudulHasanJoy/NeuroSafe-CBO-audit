import json
old = json.load(open("results/v3/lending_backup_20261024/summary.json"))
new = json.load(open("results/v3/lending/summary.json"))
cat = lambda x, c: "+" if c[0] > 0 else ("-" if c[1] < 0 else "0")
changed = total = 0
print("MATCHED rows (ours - baseline): category old -> new  [+ sig pos, - sig neg, 0 not sig]")
O = {(r["base"], r["base_knob"]): r for r in old["matched"]}
N = {(r["base"], r["base_knob"]): r for r in new["matched"]}
for k in sorted(set(O) | set(N), key=str):
    if k not in O or k not in N:
        print(f"  {k}: only in {'NEW' if k in N else 'OLD'}"); continue
    o, n = O[k], N[k]
    flag = "" if o["our_alpha"] == n["our_alpha"] else f"  (our alpha {o['our_alpha']:g} -> {n['our_alpha']:g})"
    for m in ["dR", "dV"]:
        co, cn = cat(o[m], o[m + "_ci"]), cat(n[m], n[m + "_ci"])
        total += 1; changed += co != cn
        print(f"  {k[0]:12s}@{k[1]:<5g} {m}: {o[m]:+.4f} [{co}] -> {n[m]:+.4f} [{cn}]{'  <-- CHANGED' if co != cn else ''}{flag}")
print("\nABLATION rows (variant - full, alpha=0.15)")
OA = {r["variant"]: r for r in old["ablation"]}; NA = {r["variant"]: r for r in new["ablation"]}
for v in sorted(set(OA) | set(NA)):
    if v not in OA or v not in NA:
        print(f"  {v}: only in {'NEW' if v in NA else 'OLD'}"); continue
    for m in ["dR", "dV"]:
        co, cn = cat(OA[v][m], OA[v][m + "_ci"]), cat(NA[v][m], NA[v][m + "_ci"])
        total += 1; changed += co != cn
        print(f"  {v:14s} {m}: {OA[v][m]:+.4f} [{co}] -> {NA[v][m]:+.4f} [{cn}]{'  <-- CHANGED' if co != cn else ''}")
print("\nFRONTIER comparisons (full method above / comparable points)")
IO = json.load(open("results/v3/ihdp_backup_20261024/frontier_interp.json"))["lending"]
IN = json.load(open("results/v3/frontier_interp.json"))["lending"]
for ag in sorted(set(IO) | set(IN)):
    if ag in IO and ag in IN:
        print(f"  {ag:14s} {IO[ag]['n_ours_above']}/{IO[ag]['n_comparable']} -> {IN[ag]['n_ours_above']}/{IN[ag]['n_comparable']}")
print(f"\nVERDICT: {changed} of {total} conclusions changed category across environments")
print("SafeOpt config  old:", open("results/v3/lending/safeopt_ref_config_sandbox.json").read().strip(),
      "| new:", open("results/v3/lending/safeopt_ref_config.json").read().strip())
