import os, re
SKIP = {".venv", "_archive_20261024", "__pycache__", ".git"}
files = []
for root, dirs, fs in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP]
    files += [os.path.join(root, f) for f in fs if f.endswith(".py")]
makers = [f for f in files if re.search(r"to_pickle|mimic_windows", open(f, encoding="utf-8", errors="replace").read())
          and re.search(r"chartevents|inputevents|icustays|read_csv", open(f, encoding="utf-8", errors="replace").read())]
print("EXTRACTION SCRIPT(S):", makers or "none found - search manually for the file that writes mimic_windows.pkl")
KEYS = r"hypotens|sepsis|suspicion|sofa|icd|anchor_age|age\b|>=\s*18|\b70\b|\b65\b|hours?|4h|'4H'|window|floor|resample|bin|norepi|221906|map_"
for f in makers:
    print(f"\n===== {f}")
    for i, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
        if re.search(KEYS, line, re.I):
            print(f"{i:4d}: {line.rstrip()[:150]}")
