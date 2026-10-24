import os, re, difflib
from importlib.metadata import version
rel = os.path.join(".venv", "Lib", "site-packages", "safeopt")
print("released: safeopt", version("safeopt"))
norm = lambda s: re.sub(r"\bnp\.(float|bool|int|object)(64|_)?\b", r"\1", s.replace("collections.abc", "collections")).rstrip()
compat = other = 0
for f in sorted(os.listdir("patched_safeopt")):
    if not f.endswith(".py"):
        continue
    A = open(os.path.join(rel, f), encoding="utf-8", errors="replace").read().splitlines()
    B = open(os.path.join("patched_safeopt", f), encoding="utf-8", errors="replace").read().splitlines()
    print(f"\n=== {f}")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        a, b = A[i1:i2], B[j1:j2]
        if tag == "replace" and len(a) == len(b) and all(norm(x) == norm(y) for x, y in zip(a, b)):
            compat += len(a)
            for x, y in zip(a, b):
                print(f"  compat L{i1+1}: {x.strip()}  ->  {y.strip()}")
                i1 += 1
        else:
            real_a = [x for x in a if x.strip()]; real_b = [y for y in b if y.strip()]
            if not real_a and not real_b:
                continue
            other += len(real_a) + len(real_b)
            for x in real_a: print(f"  OTHER - {x}")
            for y in real_b: print(f"  OTHER + {y}")
print(f"\ncompatibility changes: {compat} | OTHER changes: {other}")
print("VERDICT:", "COMPATIBILITY-ONLY - paper claim verified" if other == 0 else "NOT compatibility-only - inspect every OTHER line")
