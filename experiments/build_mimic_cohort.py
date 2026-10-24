"""Build the canonical MIMIC-IV sepsis vasopressor trajectory table."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
WINDOW_H = 4; MAX_RATE = 5.0
MAP_ITEMS = (220052, 220181, 225312); HR_ITEM = 220045; RHYTHM_ITEM = 220048
ARR_WORDS = ("af ", "a fib", "atrial fib", "flutter", "svt", "v tach", "vt ", "ventricular tach", "junctional")
S9 = ("99591","99592","78552","0380","0381","0382","0383","0384","0388","0389")
S10 = ("A40","A41","R652","R6520","R6521")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def build_cohort(D):
    icu = pd.read_csv(os.path.join(D,"icu","icustays.csv.gz"),
        usecols=["subject_id","hadm_id","stay_id","intime","outtime","los"], parse_dates=["intime","outtime"])
    icu = icu[icu.los >= 1.0]
    pat = pd.read_csv(os.path.join(D,"hosp","patients.csv.gz"), usecols=["subject_id","anchor_age"])
    icu = icu.merge(pat, on="subject_id", how="left"); icu = icu[icu.anchor_age >= 18]
    dx = pd.read_csv(os.path.join(D,"hosp","diagnoses_icd.csv.gz"),
        usecols=["hadm_id","icd_code","icd_version"], dtype={"icd_code": str})
    code = dx.icd_code.str.strip().str.upper()
    sep = ((dx.icd_version==9)&code.str.startswith(S9))|((dx.icd_version==10)&code.str.startswith(S10))
    icu = icu[icu.hadm_id.isin(set(dx.loc[sep,"hadm_id"]))]
    icu = icu.merge(dx.groupby("hadm_id").icd_code.nunique().rename("n_dx"), on="hadm_id", how="left")
    log(f"cohort: {len(icu):,} adult sepsis ICU stays >= 24h"); return icu.reset_index(drop=True)
def resolve_items(D):
    it = pd.read_csv(os.path.join(D,"icu","d_items.csv.gz")); lab = it.label.fillna("").str.lower()
    ne = set(it.loc[lab.str.contains("norepinephrine|levophed"),"itemid"])
    other = set(it.loc[lab.str.contains("epinephrine|vasopressin|phenylephrine|dopamine|dobutamine|angiotensin"),"itemid"]) - ne
    cat = it.category.fillna("").str.lower() if "category" in it else pd.Series("", index=it.index)
    fluid = set(it.loc[cat.str.contains("fluid|solution|blood product|crystalloid|colloid")&(it.linksto=="inputevents"),"itemid"])
    log(f"itemids -> NE {sorted(ne)}, other {len(other)}, fluids {len(fluid)}")
    if not fluid: log("  WARNING: no fluid itemids resolved")
    return ne, other, fluid
def load_inputs(D, stays, keep):
    cols = ["stay_id","starttime","endtime","itemid","amount","amountuom","rate","rateuom","patientweight"]
    out = []
    for ch in pd.read_csv(os.path.join(D,"icu","inputevents.csv.gz"), usecols=lambda c: c in cols,
                          parse_dates=["starttime","endtime"], chunksize=2_000_000, low_memory=False):
        ch = ch[ch.stay_id.isin(stays) & ch.itemid.isin(keep)]
        if len(ch): out.append(ch)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
    log(f"inputevents rows kept: {len(df):,}"); return df
def to_rate(ne, rep):
    d = ne.copy(); d["uom"] = d.rateuom.fillna("").str.lower().str.strip()
    w = d.patientweight.where(d.patientweight.between(20,400)); r = d.rate.astype(float)
    out = pd.Series(np.nan, index=d.index, dtype=float)
    for u, conv in [("mcg/kg/min",r),("mg/kg/min",r*1000),("mcg/min",r/w),("mg/hr",r*1000/60/w),("mcg/hr",r/60/w)]:
        m = d.uom == u; out[m] = conv[m]; rep[u] = int(m.sum())
    d["dose"] = out; d = d[d.dose.notna()]; hi = d.dose > MAX_RATE
    rep["dropped_implausible"] = int(hi.sum()); return d[~hi]
def spread(df, base, col, agg):
    if not len(df): return pd.Series(dtype=float)
    d = df.join(base[["intime"]], on="stay_id")
    w0 = ((d.starttime-d.intime).dt.total_seconds()//(WINDOW_H*3600)).astype("int64")
    w1 = ((d.endtime.fillna(d.starttime)-d.intime).dt.total_seconds()//(WINDOW_H*3600)).astype("int64")
    d = d.assign(w0=w0, w1=np.maximum(w0, np.minimum(w1, w0+24)))
    d = d.loc[d.index.repeat((d.w1-d.w0+1).clip(lower=1))]
    d["win"] = d.groupby(level=0).cumcount() + d.w0
    return d.groupby(["stay_id","win"])[col].agg(agg)
def chart_windows(D, stays, base, cache):
    if os.path.exists(cache):
        log(f"using cached {cache}"); chart = pd.read_pickle(cache)
    else:
        want = set(MAP_ITEMS)|{HR_ITEM, RHYTHM_ITEM}; keep=[]; seen=0; t0=time.time()
        for ch in pd.read_csv(os.path.join(D,"icu","chartevents.csv.gz"),
                usecols=["stay_id","charttime","itemid","valuenum","value"],
                parse_dates=["charttime"], chunksize=5_000_000, low_memory=False):
            seen += len(ch); k = ch[ch.itemid.isin(want) & ch.stay_id.isin(stays)]
            if len(k): keep.append(k)
            if seen % 50_000_000 < 5_000_000: log(f"  chartevents: {seen:,} ({time.time()-t0:.0f}s)")
        chart = pd.concat(keep, ignore_index=True); chart.to_pickle(cache)
        log(f"chartevents: {seen:,} scanned, {len(chart):,} kept")
    c = chart.join(base[["intime"]], on="stay_id")
    c["win"] = ((c.charttime-c.intime).dt.total_seconds()//(WINDOW_H*3600)).astype("int64")
    maps = c[c.itemid.isin(MAP_ITEMS) & c.valuenum.between(20,200)]
    hr = c[(c.itemid==HR_ITEM) & c.valuenum.between(20,300)]
    rh = c[c.itemid==RHYTHM_ITEM].copy()
    rh["arr"] = rh.value.fillna("").str.lower().str.contains("|".join(ARR_WORDS))
    g = maps.groupby(["stay_id","win"])
    out = pd.DataFrame({"map_now": g.valuenum.mean(), "map_min": g.valuenum.min()})
    out["hr_max"] = hr.groupby(["stay_id","win"]).valuenum.max()
    out["arr_now"] = rh.groupby(["stay_id","win"]).arr.max()
    return out.reset_index()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=r"C:\Users\abdus\data")
    ap.add_argument("--out", default="results"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rep = {}
    cohort = build_cohort(a.data); stays = set(cohort.stay_id)
    base = cohort.set_index("stay_id")[["intime","outtime","anchor_age","n_dx"]]
    ne_ids, other_ids, fluid_ids = resolve_items(a.data)
    raw = load_inputs(a.data, stays, ne_ids|other_ids|fluid_ids)
    conv = {}; ne = to_rate(raw[raw.itemid.isin(ne_ids)], conv); rep["unit_conversion"] = conv
    others = raw[raw.itemid.isin(other_ids)]
    fl = raw[raw.itemid.isin(fluid_ids) & raw.amountuom.fillna("").str.lower().eq("ml")]
    log(f"fluid rows: {len(fl):,}")
    n_win = ((base.outtime-base.intime).dt.total_seconds()//(WINDOW_H*3600)).astype(int)+1
    W = pd.DataFrame([(s,w) for s,n in n_win.items() for w in range(int(n))], columns=["stay_id","win"])
    W = W.merge(base.reset_index()[["stay_id","anchor_age","n_dx"]], on="stay_id", how="left")
    W = W.merge(spread(ne, base, "dose", "max").rename("dose"), on=["stay_id","win"], how="left")
    W = W.merge(spread(others, base, "itemid", "nunique").rename("n_other"), on=["stay_id","win"], how="left")
    W = W.merge(spread(fl, base, "amount", "sum").rename("fluid_ml"), on=["stay_id","win"], how="left")
    W[["dose","n_other","fluid_ml"]] = W[["dose","n_other","fluid_ml"]].fillna(0.0)
    cw = chart_windows(a.data, stays, base, os.path.join(a.out,"mimic_chart_cache.pkl"))
    W = W.merge(cw, on=["stay_id","win"], how="left")
    nz = W.loc[W.dose>0,"dose"]; edges = np.quantile(nz,[0.25,0.5,0.75])
    W["dbin"] = 0
    W.loc[W.dose>0,"dbin"] = np.searchsorted(edges, W.loc[W.dose>0,"dose"], side="right")+1
    rep["dose_edges"] = [round(float(e),4) for e in edges]
    W = W.sort_values(["stay_id","win"]); g = W.groupby("stay_id")
    for col,new in [("dose","prev_dose"),("dbin","prev_dbin"),("map_now","map_prev"),("map_min","map_min_prev"),
                    ("hr_max","hr_prev"),("arr_now","arr_prev"),("fluid_ml","fluid_prev"),("n_other","n_other_prev")]:
        W[new] = g[col].shift(1)
    W["new_arr"] = W.arr_now.fillna(False).astype(bool) & ~W.arr_prev.fillna(False).astype(bool)
    W["hypotensive"] = W.map_prev < 70
    keep = W.dropna(subset=["prev_dbin","map_prev","map_now"]).copy()
    keep["prev_dbin"] = keep.prev_dbin.astype(int)
    hypo = keep[keep.hypotensive & (keep.prev_dbin>0)]
    rep.update({"n_windows_all": int(len(W)), "n_windows_usable": int(len(keep)),
        "n_stays_usable": int(keep.stay_id.nunique()), "n_hypotensive_ondrug": int(len(hypo)),
        "n_stays_hypotensive": int(hypo.stay_id.nunique()),
        "new_arr_rate_overall": float(keep.new_arr.mean()),
        "new_arr_rate_hypo": float(hypo.new_arr.mean()) if len(hypo) else None,
        "fluid_nonzero_frac": float((keep.fluid_prev>0).mean()),
        "other_pressor_frac": float((keep.n_other_prev>0).mean())})
    keep.to_pickle(os.path.join(a.out,"mimic_windows.pkl"))
    json.dump(rep, open(os.path.join(a.out,"mimic_cohort_report.json"),"w"), indent=2)
    log(f"wrote results/mimic_windows.pkl ({len(keep):,} windows, {keep.stay_id.nunique():,} stays)")
    print(json.dumps(rep, indent=2))
if __name__ == "__main__": main()
