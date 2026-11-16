"""v3: more training data, a retriever that learns from questions, and a fitted confidence model.

Run (from ml/): .venv/bin/python train_v3.py   (after train_router.py; overwrites artifacts/model.pkl)
Data: POOL = the frozen train split + data/train_aug.csv (written from the chunks only, by an author who
did not see questions.csv). The old tune split (36) is the other-author selection set; the old test
split (31) is a DEV set here (v2 was designed after seeing it). The headline test is data/holdout.csv,
which this script never reads (load_questions only uses it to drop identical training questions).
What is learnt:
  router     TF-IDF + logistic regression; analyzer in ANALYZERS_R and C in CS.
  retriever  BM25 over each chunk's text, optionally expanded with the POOL lookup questions whose gold
             includes that chunk ("q"), so the index learns student wording (deadline, worth, hand in).
  confidence "ideal" (v2 form) or "meta": logistic regression P(top chunk is gold) on
             [router p, ideal factor, margin factor], fitted on out-of-fold predictions.
Selection (rules fixed before running; the holdout is never used):
  router config and retriever expansion: mean of (5-fold family-grouped out-of-fold score on POOL) and
  (score on tune). Router score = macro-F1; retriever score = MRR on lookups. Ties -> list order.
  threshold: on out-of-fold POOL + tune, the highest coverage whose one-sided 95% Wilson upper bound on
  risk is <= RISK_CAP (stricter than v2's point-estimate rule; ties -> higher threshold).
  confidence form: the one with the higher coverage at its own threshold (tie -> "ideal").
Writes results/v3.json, results/predictions_v3.csv, results/risk_coverage_v3.png, artifacts/model.pkl.
"""
import csv
import json
import math
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import ART, RESULTS, BM25Retriever, load_chunks, load_questions, rank
from pipeline import CourseMate, is_correct_answer
from textnorm import ANALYZERS

LABELS = ["lookup", "calc", "oos"]
ANALYZERS_R, CS, EXPAND = ["word1", "word12", "char35"], [10, 1, 100], ["none", "q"]
RISK_CAP, Z, SEED, FOLDS = 0.10, 1.645, 0, 5
CONF_FORMULA = {"ideal": "router p × min(1, top1 / Σ idf(query terms))",
                "meta": "sigmoid(b + w · [router p, ideal factor, margin factor])"}


def f1(y, yp):
    return float(f1_score(y, yp, labels=LABELS, average="macro", zero_division=0))


def fit_router(an, C, qs):
    v = TfidfVectorizer(analyzer=ANALYZERS[an])
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(v.fit_transform([q["q"] for q in qs]), [q["label"] for q in qs])
    return v, lr


def fit_retriever(expand, qs, chunks):
    extra = None
    if expand == "q":
        by = {c["id"]: [] for c in chunks}
        for q in qs:
            if q["label"] == "lookup":
                for g in q["gold"]:
                    by[g].append(q["q"])
        extra = [by[c["id"]] for c in chunks]
    return BM25Retriever().fit([c["text"] for c in chunks], extra)


def mrr(r, qs, ids):
    lk = [q for q in qs if q["label"] == "lookup"]
    out = {"top1": 0.0, "top3": 0.0, "mrr": 0.0}
    for q in lk:
        order = rank(r.scores(q["q"]))
        p = min(order.index(ids.index(g)) + 1 for g in q["gold"])
        out["top1"] += p <= 1
        out["top3"] += p <= 3
        out["mrr"] += 1 / p
    return {k: v / len(lk) for k, v in out.items()} | {"n": len(lk)}


def wilson_upper(wrong, n):
    if n == 0:
        return 1.0
    p = wrong / n
    return (p + Z * Z / (2 * n) + Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / (1 + Z * Z / n)


def rc_points(qs, outs):
    """outs come from a threshold-0 pipeline. One point per candidate threshold."""
    pts = []
    for t in sorted({0.0} | {o["conf"] for o in outs if o["label"] != "oos"}):
        ans = [(q, o) for q, o in zip(qs, outs) if o["stamp"] != "ESCALATED" and o["conf"] >= t]
        wrong = sum(not is_correct_answer(q, o) for q, o in ans)
        pts.append({"t": t, "coverage": len(ans) / len(qs), "risk": wrong / len(ans) if ans else 0.0,
                    "risk_upper": wilson_upper(wrong, len(ans)), "answered": len(ans), "wrong": wrong, "n": len(qs)})
    return pts


def choose(pts):
    ok = [p for p in pts if p["answered"] > 0 and p["risk_upper"] <= RISK_CAP]
    return sorted(ok, key=lambda p: (-p["coverage"], -p["t"]))[0] if ok else max(pts, key=lambda p: p["t"])


def at(cm, qs):
    outs = [cm.answer(q["q"]) for q in qs]
    ans = [(q, o) for q, o in zip(qs, outs) if o["stamp"] != "ESCALATED"]
    wrong = sum(not is_correct_answer(q, o) for q, o in ans)
    exp_esc = [o for q, o in zip(qs, outs) if q["expected_stamp"] == "ESCALATED"]
    e2e = sum((o["stamp"] != "ESCALATED" and is_correct_answer(q, o)) or
              (o["stamp"] == "ESCALATED" and q["expected_stamp"] == "ESCALATED") for q, o in zip(qs, outs))
    return outs, {"n": len(qs), "answered": len(ans), "wrong": wrong, "coverage": len(ans) / len(qs),
                  "risk": wrong / len(ans) if ans else 0.0, "should_escalate": len(exp_esc),
                  "escalated_of_those": sum(o["stamp"] == "ESCALATED" for o in exp_esc),
                  "end_to_end_correct": e2e, "end_to_end_acc": e2e / len(qs)}


def main():
    qs, chunks = load_questions(aug=True), load_chunks()
    ids = [c["id"] for c in chunks]
    S = {s: [q for q in qs if q["split"] == s] for s in ("train", "aug", "tune", "test")}
    pool = S["train"] + S["aug"]
    y = [q["label"] for q in pool]
    folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(pool, y, [q["family"] for q in pool]))
    print(f"pool {len(pool)} (train {len(S['train'])} + aug {len(S['aug'])}) " + str({l: y.count(l) for l in LABELS}))

    # 1. router selection
    rsel = {}
    for an in ANALYZERS_R:
        for C in CS:
            oof = [None] * len(pool)
            for tr, te in folds:
                v, lr = fit_router(an, C, [pool[i] for i in tr])
                for i, p in zip(te, lr.predict(v.transform([pool[i]["q"] for i in te]))):
                    oof[i] = p
            v, lr = fit_router(an, C, pool)
            tune = f1([q["label"] for q in S["tune"]], lr.predict(v.transform([q["q"] for q in S["tune"]])))
            rsel[f"{an}|C={C}"] = {"oof": f1(y, oof), "tune": tune, "score": (f1(y, oof) + tune) / 2}
    rbest = max(rsel, key=lambda k: rsel[k]["score"])  # max keeps the first on ties = list order
    an, C = rbest.split("|C=")[0], float(rbest.split("|C=")[1])

    # 2. retriever selection
    esel = {}
    for ex in EXPAND:
        acc = []
        for tr, te in folds:
            r = fit_retriever(ex, [pool[i] for i in tr], chunks)
            m = mrr(r, [pool[i] for i in te], ids)
            acc.append((m["mrr"], m["n"]))
        oof = sum(a * n for a, n in acc) / sum(n for _, n in acc)
        tune = mrr(fit_retriever(ex, pool, chunks), S["tune"], ids)["mrr"]
        esel[ex] = {"oof": oof, "tune": tune, "score": (oof + tune) / 2}
    ex = max(EXPAND, key=lambda k: esel[k]["score"])

    # 3. out-of-fold pipeline outputs (threshold 0, "ideal" conf) for POOL, plus tune from the full-POOL model
    def model(train_qs, mode="ideal", t=0.0, calib=None):
        v, lr = fit_router(an, C, train_qs)
        return CourseMate(v, lr, fit_retriever(ex, train_qs, chunks), chunks, 1.0, t, mode, calib)
    cal_q, cal_o = [], []
    for tr, te in folds:
        cm = model([pool[i] for i in tr])
        for i in te:
            cal_q.append(pool[i])
            cal_o.append(cm.answer(pool[i]["q"]))
    full = model(pool)
    cal_q += S["tune"]
    cal_o += [full.answer(q["q"]) for q in S["tune"]]

    # 4. confidence model on rows the router sent to lookup: target = the top chunk is a gold chunk
    lk = [(q, o) for q, o in zip(cal_q, cal_o) if o["label"] == "lookup"]
    Xc = np.array([o["feat"] for _, o in lk])
    yc = np.array([int(q["label"] == "lookup" and o["top_chunk"] in q["gold"]) for q, o in lk])
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xc, yc)
    calib = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "features": ["router_p", "ideal", "margin"],
             "n": int(len(yc)), "base_rate": float(yc.mean())}

    def with_conf(mode):
        outs = []
        for o in cal_o:
            o = dict(o)
            if o["label"] == "lookup" and mode == "meta":
                z = calib["intercept"] + sum(w * x for w, x in zip(calib["coef"], o["feat"]))
                o["conf"] = 1 / (1 + math.exp(-z))
            outs.append(o)
        return outs
    csel = {}
    for mode in ("ideal", "meta"):
        pts = rc_points(cal_q, with_conf(mode))
        csel[mode] = {"pts": pts, "chosen": choose(pts)}
    mode = max(("ideal", "meta"), key=lambda k: csel[k]["chosen"]["coverage"])
    t = csel[mode]["chosen"]["t"]

    # 5. final model = full POOL fit; score tune and the old test (dev). The holdout is scored by eval_holdout.py.
    cm = model(pool, mode, t, calib)
    outs, stats = {}, {}
    for s in ("tune", "test"):
        outs[s], stats[s] = at(cm, S[s])
    yt = [q["label"] for q in S["test"]]
    yp = [o["label"] for o in outs["test"]]
    res = {"pool": {"n": len(pool), "train": len(S["train"]), "aug": len(S["aug"]), "by_label": {l: y.count(l) for l in LABELS}},
           "router_selection": rsel, "router": rbest, "expand_selection": esel, "expand": ex,
           "router_oof_macro_f1": rsel[rbest]["oof"],
           "dev_test_router": {"macro_f1": f1(yt, yp), "confusion": confusion_matrix(yt, yp, labels=LABELS).tolist(), "n": len(yt)},
           "dev_test_retrieval": mrr(cm.r, S["test"], ids), "tune_retrieval": mrr(cm.r, S["tune"], ids),
           "calib": calib, "conf_selection": {k: {x: d["chosen"][x] for x in ("t", "coverage", "risk", "risk_upper", "answered", "wrong", "n")} for k, d in csel.items()},
           "conf_mode": mode, "conf_formula": CONF_FORMULA[mode], "threshold": t, "risk_cap": RISK_CAP,
           "tune_at_threshold": stats["tune"], "dev_test_at_threshold": stats["test"], "curve": csel[mode]["pts"]}
    (RESULTS / "v3.json").write_text(json.dumps(res, indent=1))
    with (RESULTS / "predictions_v3.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "q", "label", "pred", "conf", "stamp", "reason", "top_chunk", "gold", "correct"])
        for s in ("tune", "test"):
            for q, o in zip(S[s], outs[s]):
                ok = (o["stamp"] != "ESCALATED" and is_correct_answer(q, o)) or (o["stamp"] == "ESCALATED" and q["expected_stamp"] == "ESCALATED")
                w.writerow([q["id"], s, q["q"], q["label"], o["label"], f"{o['conf']:.4f}", o["stamp"], o.get("reason", ""), o["top_chunk"], ";".join(q["gold"]), int(ok)])
    pickle.dump({"router_vec": cm.v, "lr": cm.lr, "retriever": cm.r, "K": 1.0, "threshold": t, "router_analyzer": an,
                 "retriever_name": "bm25", "conf_mode": mode, "conf_formula": CONF_FORMULA[mode], "calib": calib, "version": 3},
                (ART / "model.pkl").open("wb"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k, c in (("ideal", "#b26a00"), ("meta", "#0568A4")):
        cs = sorted(csel[k]["pts"], key=lambda p: p["coverage"])
        ax.step([p["coverage"] for p in cs], [p["risk"] for p in cs], where="post", color=c, label=f"{k} conf (out-of-fold + tune, n={len(cal_q)})")
        ax.plot([csel[k]["chosen"]["coverage"]], [csel[k]["chosen"]["risk"]], "o", color=c)
    ax.axhline(RISK_CAP, color="#646A6E", lw=0.8, ls="--", label=f"risk cap {RISK_CAP:.2f} (applied to the 95% upper bound)")
    ax.set_xlabel("coverage = answered / all questions")
    ax.set_ylabel("risk = wrong / answered")
    ax.set_title(f"CourseMate v3 risk-coverage (dot = chosen threshold; deployed: {mode}, t={t:.3f})", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "risk_coverage_v3.png", dpi=150)

    print("router selection (oof, tune):", {k: (round(d["oof"], 3), round(d["tune"], 3)) for k, d in rsel.items()}, "->", rbest)
    print("retriever expansion MRR (oof, tune):", {k: (round(d["oof"], 3), round(d["tune"], 3)) for k, d in esel.items()}, "->", ex)
    print("confidence model:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in calib.items()})
    for k, d in res["conf_selection"].items():
        print(f"  {k}: t={d['t']:.4f} coverage {d['coverage']:.3f} risk {d['risk']:.3f} (upper {d['risk_upper']:.3f}) {d['wrong']}/{d['answered']} n={d['n']}")
    print(f"deployed: {mode}, t={t:.4f}")
    for s, a in (("tune", stats["tune"]), ("dev test", stats["test"])):
        print(f"  {s}: coverage {a['coverage']:.3f} risk {a['risk']:.3f} ({a['wrong']}/{a['answered']}) | escalated {a['escalated_of_those']}/{a['should_escalate']} | end-to-end {a['end_to_end_correct']}/{a['n']}")
    print(f"dev test router macro-F1 {res['dev_test_router']['macro_f1']:.3f}; dev test retrieval {res['dev_test_retrieval']}")


if __name__ == "__main__":
    main()
