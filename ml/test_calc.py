"""15 hand-computed exact-match cases for calc.py. Run (from ml/): .venv/bin/python test_calc.py
Expected values were worked out by hand from the Important Dates weights (Proposal 2, Poster 10,
Video 3, Final Report 7, Registration 0; total 22) and are typed here as literals; nothing is
imported from the weights table. Demo scores unless stated: proposal 85, poster 90, video 80, report 80
-> before = 2*85 + 10*90 + 3*80 + 7*80 = 170 + 900 + 240 + 560 = 1870 hundredths = 18.7.
Exit 0 = all pass.
"""
import json
import os
import sys
from pathlib import Path

from calc import calc

D = {"registration": 100, "proposal": 85, "poster": 90, "video": 80, "report": 80}
S2 = {"registration": 100, "proposal": 73, "poster": 88, "video": 91, "report": 64}
CASES = [  # (question, scores, expected subset)
    ("If I get 0 on the poster, what is my project mark?", D, {"stamp": "CALCULATED", "before": "18.7", "after": "9.7", "delta": "-9"}),
    ("what if I score 50 on the poster", D, {"after": "14.7", "delta": "-4"}),              # 1870 - 10*40
    ("what if I fail the video", D, {"after": "16.3", "delta": "-2.4"}),                     # 1870 - 3*80
    ("video trailer gets 75?", D, {"after": "18.55", "delta": "-0.15"}),                     # 1870 - 3*5
    ("what if the trailer gets full marks", D, {"after": "19.3", "delta": "0.6"}),          # 1870 + 3*20
    ("If the proposal got 90, what's my project mark?", D, {"after": "18.8"}),             # 1870 + 2*5
    ("proposal攞到100分project total幾多?", D, {"after": "19"}),                              # 1870 + 2*15
    ("If the final report gets 55 what is my project score?", D, {"after": "16.95"}),      # 1870 - 7*25
    ("what if I score 0 on the report", D, {"after": "13.1", "total": "22"}),              # 1870 - 560
    ("If I got 0 for group registration, what's my project mark?", D, {"after": "18.7", "delta": "0"}),
    ("what if poster is 100", S2, {"before": "17.47", "after": "18.67"}),  # 146+880+273+448=1747; +10*12
    ("If I get 0 in Quiz 2, what is my final percentage?", D, {"stamp": "ESCALATED", "reason": "weight not in table"}),
    ("if I fail the midterm what is my grade", D, {"stamp": "ESCALATED", "reason": "weight not in table"}),
    ("what if I get 120 on the poster", D, {"stamp": "ESCALATED", "reason": "score out of range 0-100"}),
    ("If I score 40 on the poster and 60 on the video?", D, {"stamp": "ESCALATED", "reason": "more than one item; ask about one item at a time"}),
]
fails = 0
for q, sc, want in CASES:
    got = calc(q, sc)
    bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    fails += bool(bad)
    print(("PASS " if not bad else "FAIL ") + q + ("" if not bad else f"  want/got {bad}"))
print(f"{len(CASES) - fails}/{len(CASES)} exact match")
out = Path(__file__).resolve().parent.parent / "results" / "calc.json"
key = "mutant_pass" if os.environ.get("CALC_MUTANT") else "pass"
d = json.loads(out.read_text()) if out.exists() else {}
d.update({key: len(CASES) - fails, "n": len(CASES)})
out.write_text(json.dumps(d))
sys.exit(1 if fails else 0)
