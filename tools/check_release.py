"""Pre-publication safety check. Run after `git add -A`; exits 1 (blocking the push) if any staged file
looks like patient-level MIMIC data, a pickle/cache, an oversized file, or a credential.
Usage (from the repository root):  python tools/check_release.py"""
import gzip, os, re, subprocess, sys

PATIENT_KEYS = re.compile(rb"\b(stay_id|subject_id|hadm_id|charttime|storetime)\b")
SECRET = re.compile(rb"(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                    rb"(?i:password|passwd|api[_-]?key|secret)\s*[:=]\s*['\"][^'\"]{6,})")
BINARY_DATA = (".pkl", ".pickle", ".parquet", ".feather", ".h5", ".hdf5", ".joblib")
DATA_EXT = (".csv", ".tsv", ".gz", ".json", ".txt", ".log", ".jsonl")
CODE_EXT = (".py", ".ps1", ".sh", ".md", ".tex", ".txt", ".json", ".yml", ".yaml", ".cfg", ".ini", ".toml", ".ipynb")

files = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True, check=True).stdout.split()
problems = []
for f in files:
    if not os.path.isfile(f):
        continue
    low = f.lower().replace("\\", "/")
    size = os.path.getsize(f)
    if size > 50 * 1024 * 1024:
        problems.append(f"{f}: {size / 1e6:.0f} MB (GitHub rejects files over 100 MB; keep large data out)")
    if low.endswith(BINARY_DATA):
        problems.append(f"{f}: binary data/cache file - may contain patient-level data")
    if "mimic" in low and low.startswith(("data/", "results/")) and not low.startswith("results/v3/mimic/"):
        problems.append(f"{f}: MIMIC-related data/result file - only results/v3/mimic/audit.json (aggregates) may be published")
    head = b""
    try:
        if low.endswith(".gz"):
            with gzip.open(f, "rb") as h:
                head = h.read(65536)
        else:
            with open(f, "rb") as h:
                head = h.read(262144)
    except OSError:
        pass
    if low.endswith(DATA_EXT) and low.startswith(("data/", "results/")) and PATIENT_KEYS.search(head):
        problems.append(f"{f}: contains MIMIC patient identifiers/timestamps (stay_id, subject_id, ...)")
    if low.endswith(CODE_EXT) and SECRET.search(head):
        problems.append(f"{f}: looks like it contains a password, token or API key")

print(f"checked {len(files)} staged files")
if problems:
    print("\nBLOCKED - do not publish until these are removed or ignored:")
    for p in problems:
        print("  -", p)
    print("\nTo unstage a file: git rm --cached <file>   (then add it to .gitignore)")
    sys.exit(1)
print("OK - nothing patient-level, no caches, no credentials, no oversized files")
