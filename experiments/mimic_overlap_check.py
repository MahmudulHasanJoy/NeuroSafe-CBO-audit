"""Does MIMIC-IV support off-policy evaluation of a safe dosing policy?

Three questions, each stricter than the last:

  A. COARSE     Do clinicians give different norepinephrine doses to patients
                in different severity terciles?            (always runs)
  B. CONDITIONAL Given the SAME previous dose and fluid state, does the next
                dose still vary?                            (always runs)
  C. MAP-CONDITIONED  Given the same previous dose AND the same previous mean
                arterial pressure, does the dose still vary? If this collapses,
                the apparent variation was clinicians titrating to blood
                pressure, and off-policy evaluation is not supported.
                                            (runs only if chartevents is present)

  D. TENSION    At fixed previous MAP and dose, does the next-window MAP rise
                with dose (utility) AND new tachyarrhythmia also rise (harm)?
                Both positive means safety and utility genuinely oppose.
                                            (runs only if chartevents is present)

Only aggregates are printed or written. No patient-level rows leave memory.

Usage:
    python experiments/mimic_overlap_check.py --data C:\\Users\\abdus\\data
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

NOREPI_LABELS = ("norepinephrine", "levophed")
OTHER_PRESSORS = ("epinephrine", "vasopressin", "phenylephrine", "dopamine", "dobutamine")
MAP_ITEMS = {220052: "art", 220181: "nibp", 225312: "art2"}   # mean arterial pressure
HR_ITEM = 220045
RHYTHM_ITEM = 220048
ARRHYTHMIA_WORDS = ("af ", "a fib", "atrial fib", "flutter", "svt", "v tach",
                    "vt ", "ventricular tach", "junctional")
WINDOW_H = 4
MAX_RATE = 5.0          # mcg/kg/min; above this is a charting error
SEPSIS_ICD9 = ("99591", "99592", "78552", "0380", "0381", "0382", "0383",
               "0384", "0388", "0389")
SEPSIS_ICD10 = ("A40", "A41", "R652", "R6520", "R6521")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def need(path):
    if not os.path.exists(path):
        log(f"MISSING: {path}")
        return False
    return True


# ---------------------------------------------------------------- cohort
def build_cohort(D):
    icu = pd.read_csv(os.path.join(D, "icu", "icustays.csv.gz"),
                      usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
                      parse_dates=["intime", "outtime"])
    log(f"icustays: {len(icu):,} rows")
    icu = icu[icu.los >= 1.0]                       # >= 24h
    pat = pd.read_csv(os.path.join(D, "hosp", "patients.csv.gz"),
                      usecols=["subject_id", "anchor_age"])
    icu = icu.merge(pat, on="subject_id", how="left")
    icu = icu[icu.anchor_age >= 18]

    dx = pd.read_csv(os.path.join(D, "hosp", "diagnoses_icd.csv.gz"),
                     usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    code = dx.icd_code.str.strip().str.upper()
    sep = ((dx.icd_version == 9) & code.str.startswith(SEPSIS_ICD9)) | \
          ((dx.icd_version == 10) & code.str.startswith(SEPSIS_ICD10))
    sep_hadm = set(dx.loc[sep, "hadm_id"].unique())
    icu = icu[icu.hadm_id.isin(sep_hadm)]
    # comorbidity count = distinct ICD codes for that admission
    com = dx.groupby("hadm_id").icd_code.nunique().rename("n_dx")
    icu = icu.merge(com, on="hadm_id", how="left")
    log(f"cohort: {len(icu):,} adult sepsis ICU stays >= 24h")
    return icu.reset_index(drop=True)


# ---------------------------------------------------------------- doses
def norepi_itemids(D):
    it = pd.read_csv(os.path.join(D, "icu", "d_items.csv.gz"))
    lab = it.label.fillna("").str.lower()
    ne = it[lab.str.contains("|".join(NOREPI_LABELS))]
    other = it[lab.str.contains("|".join(OTHER_PRESSORS))]
    log(f"norepinephrine itemids: {sorted(ne.itemid.tolist())}")
    return set(ne.itemid), set(other.itemid)


def load_inputevents(D, stays, ne_ids, other_ids):
    cols = ["stay_id", "starttime", "endtime", "itemid", "amount", "amountuom",
            "rate", "rateuom", "patientweight", "ordercategorydescription"]
    chunks, units = [], {}
    for ch in pd.read_csv(os.path.join(D, "icu", "inputevents.csv.gz"),
                          usecols=lambda c: c in cols, parse_dates=["starttime", "endtime"],
                          chunksize=2_000_000, low_memory=False):
        ch = ch[ch.stay_id.isin(stays)]
        keep = ch[ch.itemid.isin(ne_ids | other_ids)]
        if len(keep):
            chunks.append(keep)
        for u, n in ch[ch.itemid.isin(ne_ids)].rateuom.value_counts().items():
            units[u] = units.get(u, 0) + int(n)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    log(f"inputevents rows for cohort (pressors): {len(df):,}")
    return df, units


def to_mcg_kg_min(df, ne_ids):
    """Convert every norepinephrine rate to mcg/kg/min. Report what was done."""
    ne = df[df.itemid.isin(ne_ids)].copy()
    ne["uom"] = ne.rateuom.fillna("").str.lower().str.strip()
    w = ne.patientweight.where(ne.patientweight.between(20, 400))
    r = ne.rate.astype(float)
    out = pd.Series(np.nan, index=ne.index, dtype=float)
    rep = {}
    m = ne.uom == "mcg/kg/min";  out[m] = r[m];                     rep["mcg/kg/min (as-is)"] = int(m.sum())
    m = ne.uom == "mg/kg/min";   out[m] = r[m] * 1000;              rep["mg/kg/min (x1000)"] = int(m.sum())
    m = ne.uom == "mcg/min";     out[m] = r[m] / w[m];              rep["mcg/min (/weight)"] = int(m.sum())
    m = ne.uom == "mg/hr";       out[m] = r[m] * 1000 / 60 / w[m];  rep["mg/hr (x1000/60/weight)"] = int(m.sum())
    m = ne.uom == "mcg/hr";      out[m] = r[m] / 60 / w[m];         rep["mcg/hr (/60/weight)"] = int(m.sum())
    ne["dose"] = out
    bad_unit = int(ne.dose.isna().sum() - ne.rate.isna().sum())
    ne = ne[ne.dose.notna()]
    hi = ne.dose > MAX_RATE
    rep["dropped: unconvertible unit"] = max(bad_unit, 0)
    rep["dropped: rate > %.1f mcg/kg/min" % MAX_RATE] = int(hi.sum())
    ne = ne[~hi]
    return ne, rep


def window_index(t, intime):
    return ((t - intime).dt.total_seconds() // (WINDOW_H * 3600)).astype("int64")


def build_windows(cohort, ne, others, fluids):
    """One row per (stay_id, window): max norepinephrine dose, other-pressor flag, fluid mL."""
    base = cohort.set_index("stay_id")[["intime", "outtime", "anchor_age", "n_dx"]]
    n_win = ((base.outtime - base.intime).dt.total_seconds() // (WINDOW_H * 3600)).astype(int) + 1
    idx = [(s, w) for s, n in n_win.items() for w in range(int(n))]
    W = pd.DataFrame(idx, columns=["stay_id", "win"])
    W = W.merge(base.reset_index()[["stay_id", "anchor_age", "n_dx"]], on="stay_id", how="left")

    def spread(df, value_col, agg):
        """Attribute an infusion to every window it overlaps."""
        if not len(df):
            return pd.Series(dtype=float)
        d = df.join(base[["intime"]], on="stay_id")
        d["w0"] = window_index(d.starttime, d.intime)
        d["w1"] = window_index(d.endtime.fillna(d.starttime), d.intime)
        d["w1"] = d[["w0", "w1"]].max(axis=1).clip(upper=d.w0 + 24)   # cap runaway rows
        d = d.loc[d.index.repeat((d.w1 - d.w0 + 1).clip(lower=1))]
        d["win"] = d.groupby(level=0).cumcount() + d.w0
        return d.groupby(["stay_id", "win"])[value_col].agg(agg)

    dose = spread(ne, "dose", "max").rename("dose")
    oth = spread(others, "itemid", "nunique").rename("n_other")
    flu = spread(fluids, "amount", "sum").rename("fluid_ml") if len(fluids) else pd.Series(dtype=float)
    W = W.merge(dose, on=["stay_id", "win"], how="left") \
         .merge(oth, on=["stay_id", "win"], how="left")
    W = W.merge(flu, on=["stay_id", "win"], how="left") if len(flu) else W.assign(fluid_ml=0.0)
    W["dose"] = W.dose.fillna(0.0)
    W["n_other"] = W.n_other.fillna(0).astype(int)
    W["fluid_ml"] = W.fluid_ml.fillna(0.0)
    return W


# ---------------------------------------------------------------- tables
def dose_bins(W):
    nz = W.loc[W.dose > 0, "dose"]
    edges = np.unique(np.quantile(nz, [0.25, 0.5, 0.75])) if len(nz) else np.array([0.05, 0.1, 0.2])
    lab = ["zero"] + [f"Q{i+1}" for i in range(len(edges) + 1)]
    b = np.zeros(len(W), dtype=int)
    b[W.dose.values > 0] = np.searchsorted(edges, W.dose.values[W.dose.values > 0], side="right") + 1
    return pd.Categorical.from_codes(b, lab), edges, lab


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def pct_table(df, by, col, labels):
    t = pd.crosstab(df[by] if isinstance(by, str) else [df[c] for c in by], df[col])
    for l in labels:
        if l not in t.columns:
            t[l] = 0
    t = t[labels]
    n = t.sum(axis=1)
    pct = (t.T / n).T * 100
    pct["n"] = n
    pct["H_bits"] = [(entropy(r / r.sum()) if r.sum() else 0.0) for _, r in t.iterrows()]
    pct["H_norm"] = pct.H_bits / np.log2(len(labels))
    return pct.round(2)


# ---------------------------------------------------------------- chartevents
def load_chart(D, stays, out_parquet):
    """Extract MAP, heart rate and rhythm for the cohort. Cached to parquet."""
    if os.path.exists(out_parquet):
        log(f"using cached {out_parquet}")
        return pd.read_pickle(out_parquet)
    path = os.path.join(D, "icu", "chartevents.csv.gz")
    if not os.path.exists(path):
        return None
    want = set(MAP_ITEMS) | {HR_ITEM, RHYTHM_ITEM}
    keep, seen = [], 0
    t0 = time.time()
    for ch in pd.read_csv(path, usecols=["stay_id", "charttime", "itemid", "valuenum", "value"],
                          parse_dates=["charttime"], chunksize=5_000_000, low_memory=False):
        seen += len(ch)
        k = ch[ch.itemid.isin(want) & ch.stay_id.isin(stays)]
        if len(k):
            keep.append(k)
        if seen % 50_000_000 < 5_000_000:
            log(f"  chartevents: {seen:,} rows scanned ({time.time()-t0:.0f}s)")
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    log(f"chartevents: {seen:,} scanned, {len(df):,} kept")
    try:
        df.to_pickle(out_parquet)
    except Exception as e:
        log(f"  (parquet cache failed: {e})")
    return df


def chart_to_windows(chart, cohort):
    base = cohort.set_index("stay_id")[["intime"]]
    c = chart.join(base, on="stay_id")
    c["win"] = window_index(c.charttime, c.intime)
    maps = c[c.itemid.isin(MAP_ITEMS) & c.valuenum.between(20, 200)]
    hr = c[(c.itemid == HR_ITEM) & c.valuenum.between(20, 300)]
    rh = c[c.itemid == RHYTHM_ITEM].copy()
    rh["arr"] = rh.value.fillna("").str.lower().str.contains("|".join(ARRHYTHMIA_WORDS))
    out = pd.DataFrame(index=maps.groupby(["stay_id", "win"]).size().index)
    out["map_mean"] = maps.groupby(["stay_id", "win"]).valuenum.mean()
    out["map_min"] = maps.groupby(["stay_id", "win"]).valuenum.min()
    out["hr_max"] = hr.groupby(["stay_id", "win"]).valuenum.max()
    out["arrhythmia"] = rh.groupby(["stay_id", "win"]).arr.max()
    return out.reset_index()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\abdus\data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-chart", action="store_true")
    a = ap.parse_args()
    D, OUT = a.data, a.out
    os.makedirs(OUT, exist_ok=True)
    report = {}

    for f in [("icu", "icustays.csv.gz"), ("icu", "inputevents.csv.gz"), ("icu", "d_items.csv.gz"),
              ("hosp", "patients.csv.gz"), ("hosp", "diagnoses_icd.csv.gz")]:
        if not need(os.path.join(D, *f)):
            sys.exit("required file missing")

    cohort = build_cohort(D)
    stays = set(cohort.stay_id)
    ne_ids, other_ids = norepi_itemids(D)
    raw, units = load_inputevents(D, stays, ne_ids, other_ids)
    ne, conv = to_mcg_kg_min(raw, ne_ids)
    others = raw[raw.itemid.isin(other_ids)]
    fluids = raw[raw.ordercategorydescription.fillna("").str.contains("Bolus|Infusion", case=False)
                 & raw.amountuom.fillna("").str.lower().eq("ml")] if "ordercategorydescription" in raw else raw.iloc[:0]
    log("unit conversion: " + json.dumps(conv))
    report["units_raw"] = {str(k): v for k, v in units.items()}
    report["unit_conversion"] = conv

    W = build_windows(cohort, ne, others, fluids)
    W["dbin"], edges, labels = dose_bins(W)
    log(f"windows: {len(W):,}  stays: {W.stay_id.nunique():,}  nonzero dose: {(W.dose>0).mean()*100:.1f}%")
    report["n_windows"] = int(len(W))
    report["n_stays"] = int(W.stay_id.nunique())
    report["dose_edges_mcg_kg_min"] = [round(float(e), 4) for e in edges]

    # ---- A. coarse severity terciles (excludes norepinephrine itself)
    sev = (W.n_other.rank(pct=True) + W.fluid_ml.rank(pct=True)
           + W.anchor_age.rank(pct=True) + W.n_dx.rank(pct=True)) / 4
    W["sev"] = pd.qcut(sev, 3, labels=["T1 low", "T2 mid", "T3 high"])
    A = pct_table(W, "sev", "dbin", labels)
    print("\n=== A. COARSE: dose distribution by severity tercile (row %) ===")
    print(A.to_string())
    A.to_csv(os.path.join(OUT, "mimic_overlap.csv"))
    report["A_min_cell"] = int(pd.crosstab(W.sev, W.dbin).values.min())

    # ---- B. conditional on previous dose + fluid state
    W = W.sort_values(["stay_id", "win"])
    W["prev_dbin"] = W.groupby("stay_id").dbin.shift(1)
    W["prev_fluid"] = W.groupby("stay_id").fluid_ml.shift(1)
    W["prev_other"] = W.groupby("stay_id").n_other.shift(1)
    B = W.dropna(subset=["prev_dbin"]).copy()
    nzf = B.loc[B.prev_fluid > 0, "prev_fluid"]
    med = float(nzf.median()) if len(nzf) else float("nan")
    if not np.isfinite(med) or med <= 0:
        log(f"fluid volumes unusable ({len(nzf)} nonzero windows) - using a none/some split")
        B["fluid"] = np.where(B.prev_fluid > 0, "some", "none")
    else:
        B["fluid"] = pd.cut(B.prev_fluid, [-1, 0, med, 1e12], labels=["none", "low", "high"])
    B["pressor"] = np.where(B.prev_other > 0, "other_on", "NE_only")
    Bt = pct_table(B, ["prev_dbin", "fluid", "pressor"], "dbin", labels)
    print("\n=== B. CONDITIONAL on previous dose bin + fluid state (row %) ===")
    print(Bt.to_string())
    Bt.to_csv(os.path.join(OUT, "mimic_overlap_conditional.csv"))
    on = Bt[~Bt.index.get_level_values(0).isin(["zero"])]
    report["B_median_H_norm_on_drug"] = float(on.H_norm.median()) if len(on) else None

    # ---- C/D. chartevents
    chart = None if a.skip_chart else load_chart(D, stays, os.path.join(OUT, "mimic_chart_cache.pkl"))
    if chart is None or not len(chart):
        log("chartevents not available - skipping C and D (the decisive tests)")
        report["C_ran"] = False
    else:
        cw = chart_to_windows(chart, cohort)
        W2 = W.merge(cw, on=["stay_id", "win"], how="left")
        W2["prev_map"] = W2.groupby("stay_id").map_mean.shift(1)
        C = W2.dropna(subset=["prev_dbin", "prev_map"]).copy()
        C["map_bin"] = pd.cut(C.prev_map, [0, 60, 65, 70, 75, 80, 300],
                              labels=["<60", "60-65", "65-70", "70-75", "75-80", ">80"])
        Ct = pct_table(C, ["prev_dbin", "map_bin"], "dbin", labels)
        print("\n=== C. MAP-CONDITIONED: previous dose bin x previous MAP (row %) ===")
        print("If the spread survives here, off-policy evaluation is supported.")
        print(Ct.to_string())
        Ct.to_csv(os.path.join(OUT, "mimic_overlap_map.csv"))
        on = Ct[~Ct.index.get_level_values(0).isin(["zero"])]
        report["C_ran"] = True
        report["C_median_H_norm_on_drug"] = float(on.H_norm.median()) if len(on) else None
        report["C_verdict"] = ("usable" if report["C_median_H_norm_on_drug"] and
                               report["C_median_H_norm_on_drug"] > 0.35 else "collapsed")

        # ---- D. do utility and harm oppose?
        C["step"] = C.dbin.cat.codes - C.prev_dbin.cat.codes
        C["next_map"] = C.map_mean
        C["new_arr"] = (C.arrhythmia.fillna(False)
                        & ~C.groupby("stay_id").arrhythmia.shift(1).fillna(False))
        d = C[C.prev_dbin != "zero"].groupby(["map_bin", "step"], observed=True).agg(
            n=("next_map", "size"), next_map=("next_map", "mean"), arr=("new_arr", "mean"))
        d = d[d.n >= 30]
        print("\n=== D. TENSION: next-window MAP and new arrhythmia by dose step ===")
        print(d.round(3).to_string())
        d.to_csv(os.path.join(OUT, "mimic_tension.csv"))
        def _slopes(g):
            if g.step.nunique() < 3:
                return pd.Series({"map_slope": np.nan, "arr_slope": np.nan, "n_steps": g.step.nunique()})
            return pd.Series({"map_slope": np.polyfit(g.step, g.next_map, 1)[0],
                              "arr_slope": np.polyfit(g.step, g.arr, 1)[0],
                              "n_steps": g.step.nunique()})
        sl = d.reset_index().groupby("map_bin", observed=True).apply(_slopes, include_groups=False)
        print("\nSlopes vs dose step (want map_slope > 0 AND arr_slope > 0):")
        print(sl.round(4).to_string())
        if sl.map_slope.notna().sum() == 0:
            # fall back: pool all MAP bins so at least one estimate exists
            g = d.reset_index()
            if g.step.nunique() >= 3:
                pooled = {"map_slope": float(np.polyfit(g.step, g.next_map, 1)[0]),
                          "arr_slope": float(np.polyfit(g.step, g.arr, 1)[0])}
                print(f"Per-MAP-bin slopes unavailable (too few distinct dose steps); pooled: {pooled}")
                report["D_pooled"] = pooled
            else:
                print("Too few distinct dose steps for a slope. Dose moves are almost all 0 "
                      "-- that itself means the data cannot separate dose effects.")
                report["D_pooled"] = None
        report["D_slopes"] = json.loads(sl.to_json())

    json.dump(report, open(os.path.join(OUT, "mimic_report.json"), "w"), indent=2)
    print("\n=== VERDICT ===")
    print(f"A coarse overlap: min cell {report['A_min_cell']} windows")
    print(f"B within-patient normalised entropy (on drug): {report.get('B_median_H_norm_on_drug')}")
    if report.get("C_ran"):
        print(f"C MAP-conditioned entropy: {report.get('C_median_H_norm_on_drug')} -> {report.get('C_verdict')}")
        print("   > 0.35 means real variation survives conditioning on blood pressure.")
    else:
        print("C not run (chartevents missing) - this is the decisive test.")
    log("done. Aggregates in " + OUT)


if __name__ == "__main__":
    main()
