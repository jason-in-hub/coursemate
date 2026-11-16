"""Score the frozen blind holdout once per model. Nothing is refitted.

Run (from ml/): .venv/bin/python eval_holdout.py
data/holdout.csv: 150 questions written from data/chunks.jsonl only, by an author (an LLM agent in a
separate context) that saw neither questions.csv, train_aug.csv, the results nor the model. Its SHA-256
is in data/FROZEN.sha256. Columns as questions.csv (label, gold_chunk_ids, kind, calc_item, calc_score,
expected_stamp).
Models: artifacts/model_v2.pkl (the v2 pipeline, trained on the 100 train questions) and
artifacts/model.pkl (deployed v3). Both are scored on the same questions, so the comparison is paired.
Metrics: router macro-F1; lookup Top-1/Top-3/MRR; coverage, risk, end-to-end accuracy at each model's
own threshold; 95% bootstrap CIs (2000 resamples over questions); paired end-to-end wins/losses.
Writes results/holdout.json and results/holdout_predictions.csv.
"""
import csv
import json
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from common import ART, DATA, RESULTS, load_chunks, rank
from pipeline import CourseMate, is_correct_answer

LABELS, B, SEED = ["lookup", "calc", "oos"], 2000, 0


def boot(fn, n):
    rng = np.random.default_rng(SEED)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    v = [x for x in v if x is not None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score(m, rows, chunks):
    ids = [c["id"] for c in chunks]
    cm = CourseMate.from_pickle(m, chunks)
    outs = [cm.answer(r["q"]) for r in rows]
    n = len(rows)
    y, yp = np.array([r["label"] for r in rows]), np.array([o["label"] for o in outs])
    ans = np.array([o["stamp"] != "ESCALATED" for o in outs])
    right = np.array([bool(a) and is_correct_answer(r, o) for r, o, a in zip(rows, outs, ans)])
    e2e = np.array([bool(c) or (not a and r["expected_stamp"] == "ESCALATED") for r, a, c in zip(rows, ans, right)])
    lk = [i for i, r in enumerate(rows) if r["label"] == "lookup"]
    pos = {i: min(rank(m["retriever"].scores(rows[i]["q"])).index(ids.index(g)) + 1 for g in rows[i]["gold"]) for i in lk}
    f1 = lambda idx: float(f1_score(y[idx], yp[idx], labels=LABELS, average="macro", zero_division=0))
    risk = lambda idx: (float((ans[idx] & ~right[idx]).sum() / ans[idx].sum()) if ans[idx].sum() else None)
    lki = np.array(lk)
    top = lambda k: (lambda idx: float(np.mean([pos[lki[j]] <= k for j in idx])))
    al = np.arange(n)
    exp_esc = np.array([r["expected_stamp"] == "ESCALATED" for r in rows])
    calc_ok = [i for i, r in enumerate(rows) if r["expected_stamp"] == "CALCULATED"]
    by_kind = {}
    for k in sorted({r["kind"] for r in rows}):
        idx = np.array([i for i, r in enumerate(rows) if r["kind"] == k])
        by_kind[k] = {"n": len(idx), "coverage": float(ans[idx].mean()), "risk": risk(idx), "end_to_end": float(e2e[idx].mean())}
    res = {"n": n, "by_label": {l: int((y == l).sum()) for l in LABELS}, "threshold": m["threshold"], "conf_formula": m["conf_formula"],
           "router_macro_f1": f1(al), "router_macro_f1_ci": boot(f1, n), "router_accuracy": float((y == yp).mean()),
           "router_per_class_f1": dict(zip(LABELS, map(float, f1_score(y, yp, labels=LABELS, average=None, zero_division=0)))),
           "confusion": confusion_matrix(y, yp, labels=LABELS).tolist(),
           "lookup_n": len(lk), "top1": top(1)(np.arange(len(lk))), "top1_ci": boot(top(1), len(lk)),
           "top3": top(3)(np.arange(len(lk))), "top3_ci": boot(top(3), len(lk)), "mrr": float(np.mean([1 / pos[i] for i in lk])),
           "answered": int(ans.sum()), "wrong": int((ans & ~right).sum()), "coverage": float(ans.mean()), "coverage_ci": boot(lambda i: float(ans[i].mean()), n),
           "risk": risk(al), "risk_ci": boot(risk, n),
           "end_to_end_correct": int(e2e.sum()), "end_to_end_acc": float(e2e.mean()), "end_to_end_ci": boot(lambda i: float(e2e[i].mean()), n),
           "should_escalate": int(exp_esc.sum()), "escalated_of_those": int((exp_esc & ~ans).sum()),
           "calc_expected": len(calc_ok), "calc_exact": int(sum(right[i] for i in calc_ok)),
           "calc_answered": int(sum(o["stamp"] == "CALCULATED" for o in outs)),
           "calc_answered_exact": int(sum(right[i] for i, o in enumerate(outs) if o["stamp"] == "CALCULATED")),
           "wrong_by_type": {"lookup_wrong_chunk": int(sum(a and not c and r["label"] == "lookup" and o["stamp"] == "SOURCED" for r, o, a, c in zip(rows, outs, ans, right))),
                             "answered_should_escalate": int(sum(a and r["expected_stamp"] == "ESCALATED" for r, a in zip(rows, ans))),
                             "other": 0},
           "by_kind": by_kind}
    res["wrong_by_type"]["other"] = res["wrong"] - res["wrong_by_type"]["lookup_wrong_chunk"] - res["wrong_by_type"]["answered_should_escalate"]
    return res, outs, e2e


def main():
    path = DATA / "holdout.csv"
    if not path.exists():
        print("data/holdout.csv not found: nothing scored.")
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    chunks = load_chunks()
    ids = {c["id"] for c in chunks}
    for r in rows:
        r["gold"] = [g for g in r["gold_chunk_ids"].split(";") if g]
        assert r["label"] in LABELS and set(r["gold"]) <= ids and (r["label"] != "lookup" or r["gold"]), r
    out, e2e, preds = {}, {}, {}
    for name, f in (("v2", "model_v2.pkl"), ("v3", "model.pkl")):
        if (ART / f).exists():
            out[name], preds[name], e2e[name] = score(pickle.load((ART / f).open("rb")), rows, chunks)
    if len(e2e) == 2:
        a, b = e2e["v2"], e2e["v3"]
        rng = np.random.default_rng(SEED)
        d = b.astype(float) - a.astype(float)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        out["paired"] = {"v3_right_v2_wrong": int((b & ~a).sum()), "v2_right_v3_wrong": int((a & ~b).sum()),
                         "end_to_end_diff": float(d.mean()), "end_to_end_diff_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    (RESULTS / "holdout.json").write_text(json.dumps(out, indent=1))
    with (RESULTS / "holdout_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "q", "label", "kind", "expected_stamp", "gold", "model", "pred", "conf", "stamp", "reason", "top_chunk", "item", "score", "correct"])
        for name in preds:
            for r, o, c in zip(rows, preds[name], e2e[name]):
                w.writerow([r["id"], r["q"], r["label"], r["kind"], r["expected_stamp"], ";".join(r["gold"]), name, o["label"], f"{o['conf']:.4f}",
                            o["stamp"], o.get("reason", ""), o["top_chunk"], o.get("item", ""), o.get("score", ""), int(c)])
    for name in preds:
        r = out[name]
        print(f"{name} (t={r['threshold']:.4f}) holdout n={r['n']} {r['by_label']}")
        print(f"  router macro-F1 {r['router_macro_f1']:.3f} [{r['router_macro_f1_ci'][0]:.3f}, {r['router_macro_f1_ci'][1]:.3f}]; "
              f"lookup Top-1 {r['top1']:.3f}, Top-3 {r['top3']:.3f} [{r['top3_ci'][0]:.3f}, {r['top3_ci'][1]:.3f}], MRR {r['mrr']:.3f} (n={r['lookup_n']})")
        print(f"  coverage {r['coverage']:.3f}, risk {r['risk']:.3f} [{r['risk_ci'][0]:.3f}, {r['risk_ci'][1]:.3f}] ({r['wrong']}/{r['answered']}), "
              f"end-to-end {r['end_to_end_correct']}/{r['n']}, escalated {r['escalated_of_those']}/{r['should_escalate']} that should be, "
              f"calc exact {r['calc_answered_exact']}/{r['calc_answered']} answered ({r['calc_exact']}/{r['calc_expected']} of expected)")
    if "paired" in out:
        p = out["paired"]
        print(f"paired end-to-end: v3 right & v2 wrong {p['v3_right_v2_wrong']}, v2 right & v3 wrong {p['v2_right_v3_wrong']}, "
              f"diff {p['end_to_end_diff']:+.3f} [{p['end_to_end_diff_ci'][0]:+.3f}, {p['end_to_end_diff_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
