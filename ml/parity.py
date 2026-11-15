"""Parity: the JS inference embedded in index.html must reproduce the fitted Python pipeline.

Run (from ml/): .venv/bin/python parity.py   (needs node, artifacts/model.pkl and an exported index.html)
Python side = the pickled sklearn / rank_bm25 objects (not a re-read of model.json).
Must match on every question: router label, top chunk, final stamp, escalation reason, calc result.
Also reports the largest probability / retrieval-score difference and the smallest router top1-top2 margin.
Optional argument: path of an HTML file to test instead of ../index.html (used by mutate_parity.sh).
Exit 0 = full parity.
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

from common import ART, load_chunks, load_questions
from pipeline import CourseMate

KEYS = ["label", "top_chunk", "stamp", "reason", "item", "score", "before", "after", "delta"]


def main():
    m = pickle.load((ART / "model.pkl").open("rb"))
    cm = CourseMate(m["router_vec"], m["lr"], m["retriever"], load_chunks(), m["K"], m["threshold"])
    qs = load_questions()
    extra = ["What if I get 0 on the poster?", "proposal幾時due", "ＱＵＩＺ　２ due?", "", "!!!", "report 55.5",
             "team 9 poster 70", "What is the meaning of life?"]  # edge cases beyond the labelled set
    texts = [q["q"] for q in qs] + extra
    js = json.loads(subprocess.run(["node", str(Path(__file__).with_name("parity_dump.mjs"))] + sys.argv[1:2], input=json.dumps(texts),
                                   capture_output=True, text=True, check=True).stdout)
    mism, dp, dc, ds, mm = [], 0.0, 0.0, 0.0, 1.0
    for t, j in zip(texts, js):
        p = cm.answer(t)
        bad = [k for k in KEYS if p.get(k) != j.get(k)]
        if bad:
            mism.append((t, {k: (p.get(k), j.get(k)) for k in bad}))
        dp = max(dp, max(abs(a - b) for a, b in zip(p["p"], j["p"])))
        dc = max(dc, abs(p["conf"] - j["conf"]))
        ds = max(ds, abs(p["top_score"] - j["top_score"]))
        mm = min(mm, p["margin"])
    print(f"parity: {len(texts) - len(mism)}/{len(texts)} questions match on {', '.join(KEYS)} "
          f"({len(qs)} labelled + {len(extra)} edge cases)")
    print(f"max |p_py - p_js| = {dp:.2e}; max |conf diff| = {dc:.2e}; max |top score diff| = {ds:.2e}; "
          f"smallest router top1-top2 margin = {mm:.4f}")
    for t, d in mism:
        print("MISMATCH", repr(t), d)
    out = {"n": len(texts), "match": len(texts) - len(mism), "max_p_diff": dp, "max_conf_diff": dc,
           "max_score_diff": ds, "min_margin": mm, "mismatches": [[t, str(d)] for t, d in mism]}
    if len(sys.argv) < 2:
        (Path(__file__).resolve().parent.parent / "results" / "parity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
