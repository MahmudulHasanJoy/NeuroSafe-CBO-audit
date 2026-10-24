import json
A = json.load(open("results/v3/mimic/audit.json")); T = "paper/v3/tables"
NAME = {"hgb_current": "gradient boosting (original)", "hgb_regularised": "gradient boosting (regularised)",
        "logistic": "logistic regression", "ridge": "ridge regression"}
def tab(path, cols, header, rows):
    open(f"{T}/{path}", "w").write("\\begin{tabular}{" + cols + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
                                   + "\n".join(r + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
rows = [f"harm: {NAME[n]}{' (chosen)' if n == A['chosen_harm'] else ''} & {v['logloss']:.5f} & {v['auc']:.3f} & --"
        for n, v in A["cv"]["harm"].items()]
rows += [f"MAP: {NAME[n]}{' (chosen)' if n == A['chosen_map'] else ''} & -- & -- & {v['r2']:.3f}"
         for n, v in A["cv"]["map"].items()]
rows += [f"original models, in-sample & -- & {A['insample_original']['auc']:.3f} & {A['insample_original']['r2']:.3f}",
         f"chosen models, held-out stays & -- & {A['heldout_chosen']['auc']:.3f} & {A['heldout_chosen']['r2']:.3f}"]
tab("mimic_validation.tex", "lccc", "Model (grouped 5-fold CV on the fit half unless stated) & log-loss & AUC & $R^2$", rows)
tab("mimic_gate.tex", "cccccc",
    "$\\alpha$ & $\\geq$2 safe doses & best dose unsafe & cost when binding (mmHg) & headroom (mmHg) & gate",
    [f"{g['alpha']:g} & {g['choice']:.3f} & {g['binds']:.3f} & {g['cost']:.2f} & {g['headroom']:+.2f} & "
     f"{'pass' if g['passed'] else 'fail'}" for g in A["gate"]])
o = A["ope"]; md, hd = o["map_diff"], o["harm_diff"]
tab("mimic_ope.tex", "lcccc",
    "Outcome & clinicians & constrained policy (WIS) & difference [95\\% CI, stays resampled] & difference (DR)",
    [f"MAP (mmHg) & {o['clin_map']:.2f} & {o['wis_map']:.2f} & {md['d']:+.3f} [{md['lo']:+.3f}, {md['hi']:+.3f}] & {o['dr_map'] - o['clin_map']:+.3f}",
     f"arrhythmia rate & {o['clin_harm']:.4f} & {o['wis_harm']:.4f} & {hd['d']:+.4f} [{hd['lo']:+.4f}, {hd['hi']:+.4f}] & {o['dr_harm'] - o['clin_harm']:+.4f}"])
grid = {}
for s in A["sensitivity"]:
    grid.setdefault(s["eps"], {})[s["clip"]] = f"{s['d']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]"
tab("mimic_sensitivity.tex", "lccc", "$\\varepsilon$ & cap 10 & cap 20 & cap 50",
    [f"{e:g} & " + " & ".join(grid[e][c] for c in (10.0, 20.0, 50.0)) for e in sorted(grid)])
print(open(f"{T}/mimic_ope.tex").read())
print(f"ESS {o['ess']:.0f} ({100 * o['ess_frac']:.1f}%), {o['n_clipped']} weights clipped, reliable={o['reliable']}")
