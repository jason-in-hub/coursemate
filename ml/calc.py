"""Deterministic what-if calculator over the project weights table. No model does arithmetic.

Weights come only from data/weights.json (the Important Dates table). Arithmetic is in integer
hundredths of a course-mark point (weight% x score/100), so 18.7 is exactly 1870; no binary floats.
index.html ports this logic (calc() in the core script); parity.py checks both agree on every question.

calc(question, scores) -> dict with stamp "CALCULATED" or "ESCALATED":
  CALCULATED: item, score, before, after, delta  (strings, exact, trailing zeros dropped)
  ESCALATED:  reason (e.g. "weight not in table")
"""
import json
import re
from pathlib import Path

from textnorm import normalize, tokens

TABLE = json.loads((Path(__file__).resolve().parent / "data" / "weights.json").read_text(encoding="utf-8"))
ITEMS = {i["key"]: i for i in TABLE["items"]}
NO_WEIGHT = set(TABLE["no_weight_terms"])


def fmt(h):
    """Integer hundredths -> exact decimal string ("1870" -> "18.7", "-60" -> "-0.6")."""
    sign, h = ("-" if h < 0 else ""), abs(h)
    s = f"{h // 100}.{h % 100:02d}".rstrip("0").rstrip(".")
    return sign + s


def calc(q, scores=None):
    scores = dict(TABLE["demo_scores"] if scores is None else scores)
    t = tokens(q)
    esc = lambda reason: {"stamp": "ESCALATED", "reason": reason}
    if NO_WEIGHT & set(t):
        return esc("weight not in table")
    found = {k for k, i in ITEMS.items() if set(i["aliases"]) & set(t)}
    if "proposal" in found:
        found.discard("report")  # "proposal report" is the proposal
    if not found:
        return esc("no project item named")
    if len(found) > 1:
        return esc("more than one item; ask about one item at a time")
    item = found.pop()
    if re.search(r"[0-9]+[.][0-9]+", normalize(q)):
        return esc("whole-number scores only")
    nums = [int(w) for i, w in enumerate(t) if w.isdigit() and not (i and t[i - 1] in ("team", "l"))]
    if nums:
        val = nums[0]
    elif {"zero", "fail", "failed", "fails"} & set(t):
        val = 0
    elif "full" in t:
        val = 100
    else:
        return esc("no score given")
    if not 0 <= val <= 100:
        return esc("score out of range 0-100")
    w = ITEMS[item]["weight"]
    before = sum(ITEMS[k]["weight"] * scores[k] for k in ITEMS)
    after = before + w * (val - scores[item])
    return {"stamp": "CALCULATED", "item": item, "score": val, "before": fmt(before), "after": fmt(after),
            "delta": fmt(after - before), "total": fmt(100 * sum(i["weight"] for i in ITEMS.values()))}


if __name__ == "__main__":
    import sys
    print(json.dumps(calc(" ".join(sys.argv[1:])), ensure_ascii=False))
