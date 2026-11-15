"""Freeze the 60/20/20 train/tune/test split (run once, before any modelling).

Run (from ml/): .venv/bin/python make_split.py
Families (variants of one base question) stay in one split so a typo variant in test never has its
clean twin in train. Stratified by label: within each label, families are shuffled with SEED and
assigned greedily to the split furthest below its 60/20/20 share of that label's questions.
Refuses to overwrite an existing data/split.csv; run_all.sh checks data/FROZEN.sha256 instead.
"""
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q, S, H = HERE / "data/questions.csv", HERE / "data/split.csv", HERE / "data/FROZEN.sha256"
SEED, SHARE = 1945, {"train": 0.6, "tune": 0.2, "test": 0.2}

if S.exists():
    sys.exit("data/split.csv exists: the split is frozen. Delete it by hand only if you mean to re-split.")
rows = list(csv.DictReader(Q.open(encoding="utf-8")))
rng = random.Random(SEED)
out = {}
for label in sorted({r["label"] for r in rows}):
    fams = defaultdict(list)
    for r in rows:
        if r["label"] == label:
            fams[r["family"]].append(r["id"])
    keys = sorted(fams)
    rng.shuffle(keys)
    total = sum(len(v) for v in fams.values())
    got = dict.fromkeys(SHARE, 0)
    for k in keys:
        s = max(SHARE, key=lambda s: SHARE[s] * total - got[s])
        got[s] += len(fams[k])
        for i in fams[k]:
            out[i] = s
with S.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "split"])
    for r in rows:
        w.writerow([r["id"], out[r["id"]]])
H.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  data/{p.name}\n" for p in (Q, S)))
cnt = defaultdict(int)
for r in rows:
    cnt[(out[r["id"]], r["label"])] += 1
print("split frozen (seed", SEED, "):", {s: {l: cnt[(s, l)] for l in ("lookup", "calc", "oos")} for s in SHARE})
