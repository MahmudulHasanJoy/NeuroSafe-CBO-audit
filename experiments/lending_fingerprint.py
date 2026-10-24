import os, sys, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import neurosafe_cbo.env_lending_bandit as L

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else "MISSING"
print("raw     ", L.RAW_PATH, sha(L.RAW_PATH))
print("prepped ", L.PREPPED_PATH, sha(L.PREPPED_PATH))

cached = pd.read_csv(L.PREPPED_PATH) if os.path.exists(L.PREPPED_PATH) else None
orig = L.PREPPED_PATH
os.makedirs("results/tmp", exist_ok=True)
L.PREPPED_PATH = "results/tmp/_never_exists.csv.gz"
fresh = L._prep_table(L.RAW_PATH)
L.PREPPED_PATH = orig
if os.path.exists("results/tmp/_never_exists.csv.gz"):
    os.remove("results/tmp/_never_exists.csv.gz")
if cached is None:
    print("PREP: no cached table - every run rebuilds it with this pandas version")
else:
    cols = [c for c in ["risk_score", "duration_norm", "amount_action", "default"] if c in cached and c in fresh]
    same = len(cached) == len(fresh) and all(np.allclose(cached[c].to_numpy(float), fresh[c].to_numpy(float)) for c in cols)
    print(f"PREP: cached vs freshly rebuilt table ({len(cached)} vs {len(fresh)} rows, cols {cols}) -> {'IDENTICAL' if same else 'DIFFERENT'}")

d = L._load_and_fit(L.RAW_PATH, seed=0)
cov, model = d["covariates"], d["model"]
acts = np.linspace(0, 1, 11)
Q = np.column_stack([np.repeat(cov, len(acts), axis=0), np.tile(acts, len(cov))])
p = model.predict_proba(Q)[:, 1]
print(f"HARM MODEL fingerprint: mean {p.mean():.6f} | quantiles 5/50/95 {np.round(np.quantile(p, [.05, .5, .95]), 6)} "
      f"| hash {hashlib.sha256(np.round(p, 6).tobytes()).hexdigest()[:16]}")
