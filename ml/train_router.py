"""Train and evaluate the 3-class router (lookup / calc / oos) and choose the answer threshold.

Run (from ml/): .venv/bin/python train_router.py   (needs results/retrieval.json from eval_retrieval.py)
- Model: TF-IDF (analyzer chosen on tune from word1 / word12 / char35) + LogisticRegression
  (C=10, class_weight="balanced", lbfgs multinomial). Fitted on TRAIN only.
- Report: test macro-F1 + confusion matrix vs a majority-class baseline; 5-fold StratifiedGroupKFold
  (groups = question family) on train+tune as a stability check.
- Risk-coverage over the full pipeline (pipeline.py). Coverage = answered / all questions;
  risk = wrong answers / answered. Threshold rule (fixed before running): on TUNE, the highest
  coverage with risk <= RISK_CAP; ties -> the higher threshold. Then report test at that threshold.
Writes results/router.json, results/predictions.csv, results/risk_coverage.png, artifacts/model.pkl.
"""
import csv
import json
import pickle

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import ART, RESULTS, load_chunks, load_questions, make_retriever, rank
from pipeline import CourseMate, is_correct_answer
from textnorm import ANALYZERS

LABELS, ROUTER_ANALYZERS, RISK_CAP, SEED = ["lookup", "calc", "oos"], ["word1", "word12", "char35"], 0.10, 0


def fit(an, X, y):
    v = TfidfVectorizer(analyzer=ANALYZERS[an])
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=5000).fit(v.fit_transform(X), y)
    return v, lr


def f1(y, yp):
    return float(f1_score(y, yp, labels=LABELS, average="macro", zero_division=0))


def rc_points(qs, outs):
    """Risk/coverage for every candidate threshold (each distinct conf among non-oos, plus 0)."""
    pts = []
    for t in sorted({0.0} | {o["conf"] for o in outs if o["label"] != "oos"}):
        ans = [(q, o) for q, o in zip(qs, outs) if o["label"] != "oos" and o["conf"] >= t
               and o.get("stamp_raw", o["stamp"]) != "ESCALATED"]
        wrong = sum(not is_correct_answer(q, o) for q, o in ans)
        pts.append({"t": t, "coverage": len(ans) / len(qs), "risk": wrong / len(ans) if ans else 0.0,
                    "answered": len(ans), "wrong": wrong, "n": len(qs)})
    return pts


def main():
    qs, chunks = load_questions(), load_chunks()
    S = {s: [q for q in qs if q["split"] == s] for s in ("train", "tune", "test")}
    X = {s: [q["q"] for q in S[s]] for s in S}
    y = {s: [q["label"] for q in S[s]] for s in S}

    # 1. analyzer selection on tune
    sel = {}
    for a in ROUTER_ANALYZERS:
        va, la = fit(a, X["train"], y["train"])
        sel[a] = f1(y["tune"], la.predict(va.transform(X["tune"])))
    an = sorted(ROUTER_ANALYZERS, key=lambda a: (-sel[a], ROUTER_ANALYZERS.index(a)))[0]
    v, lr = fit(an, X["train"], y["train"])
    yp = list(lr.predict(v.transform(X["test"])))
    maj = DummyClassifier(strategy="most_frequent").fit(X["train"], y["train"])
    test = {"macro_f1": f1(y["test"], yp), "majority_macro_f1": f1(y["test"], maj.predict(X["test"])),
            "n": len(yp), "n_by_class": {l: y["test"].count(l) for l in LABELS},
            "per_class_f1": dict(zip(LABELS, map(float, f1_score(y["test"], yp, labels=LABELS, average=None, zero_division=0)))),
            "confusion": confusion_matrix(y["test"], yp, labels=LABELS).tolist(),
            "accuracy": float(np.mean(np.array(yp) == np.array(y["test"])))}

    # 2. 5-fold CV on train+tune, family-grouped
    tt = S["train"] + S["tune"]
    Xt, yt, gt = [q["q"] for q in tt], [q["label"] for q in tt], [q["family"] for q in tt]
    cv, cvm = [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xt, yt, gt):
        vv, ll = fit(an, [Xt[i] for i in tr], [yt[i] for i in tr])
        cv.append(f1([yt[i] for i in te], ll.predict(vv.transform([Xt[i] for i in te]))))
        cvm.append(f1([yt[i] for i in te], [max(set(yt[i] for i in tr), key=[yt[i] for i in tr].count)] * len(te)))

    # 3. pipeline, threshold on tune
    ret = json.loads((RESULTS / "retrieval.json").read_text())["deployed"]
    r = make_retriever(ret).fit([c["text"] for c in chunks])
    train_lookup_top = [float(max(r.scores(q["q"]))) for q in S["train"] if q["label"] == "lookup"]
    K = float(np.median(train_lookup_top)) if ret == "bm25" else 1.0
    cm = CourseMate(v, lr, r, chunks, K, threshold=0.0)
    outs = {s: [cm.answer(q["q"]) for q in S[s]] for s in S}
    pts = {s: rc_points(S[s], outs[s]) for s in ("tune", "test")}
    ok = [p for p in pts["tune"] if p["risk"] <= RISK_CAP and p["answered"] > 0]
    chosen = sorted(ok, key=lambda p: (-p["coverage"], -p["t"]))[0] if ok else max(pts["tune"], key=lambda p: p["t"])
    t = chosen["t"]
    cm.t = t
    final = {s: [cm.answer(q["q"]) for q in S[s]] for s in S}

    def at(s):
        ans = [(q, o) for q, o in zip(S[s], final[s]) if o["stamp"] != "ESCALATED"]
        wrong = sum(not is_correct_answer(q, o) for q, o in ans)
        exp_esc = [o for q, o in zip(S[s], final[s]) if q["expected_stamp"] == "ESCALATED"]
        e2e = sum((o["stamp"] != "ESCALATED" and is_correct_answer(q, o)) or
                  (o["stamp"] == "ESCALATED" and q["expected_stamp"] == "ESCALATED") for q, o in zip(S[s], final[s]))
        return {"n": len(S[s]), "answered": len(ans), "wrong": wrong, "coverage": len(ans) / len(S[s]),
                "risk": wrong / len(ans) if ans else 0.0,
                "should_escalate": len(exp_esc), "escalated_of_those": sum(o["stamp"] == "ESCALATED" for o in exp_esc),
                "end_to_end_correct": e2e, "end_to_end_acc": e2e / len(S[s])}

    p0 = lambda s: next(p for p in pts[s] if p["t"] == 0.0)
    res = {"router_analyzer_selection_tune_macro_f1": sel, "router_analyzer": an, "C": 10,
           "test": test, "cv5_train_tune": {"folds": cv, "mean": float(np.mean(cv)), "sd": float(np.std(cv, ddof=1)),
                                            "majority_mean": float(np.mean(cvm)), "n": len(tt)},
           "retriever": ret, "K": K, "risk_cap": RISK_CAP, "threshold": t,
           "tune_at_threshold": at("tune"), "test_at_threshold": at("test"),
           "test_no_threshold": p0("test"), "tune_no_threshold": p0("tune"), "curves": pts}
    (RESULTS / "router.json").write_text(json.dumps(res, indent=1))
    with (RESULTS / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "q", "label", "pred", "conf", "stamp", "reason", "top_chunk", "gold", "correct"])
        for s in S:
            for q, o in zip(S[s], final[s]):
                ok_ = (o["stamp"] != "ESCALATED" and is_correct_answer(q, o)) or (o["stamp"] == "ESCALATED" and q["expected_stamp"] == "ESCALATED")
                w.writerow([q["id"], s, q["q"], q["label"], o["label"], f"{o['conf']:.4f}", o["stamp"], o.get("reason", ""),
                            o["top_chunk"], ";".join(q["gold"]), int(ok_)])
    pickle.dump({"router_vec": v, "lr": lr, "retriever": r, "K": K, "threshold": t, "router_analyzer": an,
                 "retriever_name": ret}, (ART / "model.pkl").open("wb"))

    SAMPLE = ["When is the proposal due?", "If I get 0 on the poster, what is my project mark?",
              "Can you write my assignment?", "Is late submission allowed?"]  # the page's sample buttons
    (RESULTS / "sample_buttons.json").write_text(json.dumps(
        [{"q": s, **{k: cm.answer(s).get(k) for k in ("stamp", "reason", "conf", "top_chunk")}} for s in SAMPLE], indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for s, c in (("tune", "#0568A4"), ("test", "#b26a00")):
        cs = sorted(pts[s], key=lambda p: p["coverage"])
        ax.step([p["coverage"] for p in cs], [p["risk"] for p in cs], where="post", color=c,
                label=f"{s} (n={len(S[s])})")
        ax.plot([at(s)["coverage"]], [at(s)["risk"]], "o", color=c)
    ax.axhline(RISK_CAP, color="#646A6E", lw=0.8, ls="--", label=f"risk cap {RISK_CAP:.2f} (tune)")
    ax.set_xlabel("coverage = answered / all questions")
    ax.set_ylabel("risk = wrong / answered")
    ax.set_title(f"CourseMate risk-coverage (dot = threshold {t:.3f} chosen on tune)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, max(p["risk"] for s in pts for p in pts[s]) + 0.05))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "risk_coverage.png", dpi=150)

    T = res["test"]
    print(f"router analyzer (tune macro-F1): {sel} -> {an}")
    print(f"TEST macro-F1 {T['macro_f1']:.3f} (majority baseline {T['majority_macro_f1']:.3f}), acc {T['accuracy']:.3f}, n={T['n']} {T['n_by_class']}")
    print("confusion rows=true cols=pred", LABELS, T["confusion"])
    print(f"5-fold CV train+tune macro-F1 {np.mean(cv):.3f} +- {np.std(cv, ddof=1):.3f} (majority {np.mean(cvm):.3f})")
    print(f"retriever {ret}, K={K:.3f}, threshold {t:.4f}")
    for s in ("tune", "test"):
        a = at(s)
        print(f"{s}: coverage {a['coverage']:.3f} risk {a['risk']:.3f} ({a['wrong']}/{a['answered']} wrong) | "
              f"escalated {a['escalated_of_those']}/{a['should_escalate']} that should be | end-to-end {a['end_to_end_correct']}/{a['n']}")
    print(f"test without threshold: coverage {p0('test')['coverage']:.3f} risk {p0('test')['risk']:.3f}")


if __name__ == "__main__":
    main()
