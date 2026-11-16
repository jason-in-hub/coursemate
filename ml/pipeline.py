"""End-to-end answer(): router -> (retrieval | calculator | escalate) -> confidence threshold.

conf_mode selects the lookup retrieval factor (v2 picks one on TUNE; see train_router.py):
  "v1"     s1 / (s1 + K)                                  (original; penalises short queries)
  "margin" s1 / (s1 + s2)        top-1 vs top-2 BM25 score
  "ideal"  min(1, s1 / sum(idf of query terms))  ideal doc = each query term once, at average length
  "p"      1                     router max-probability alone
  "meta"   v3: conf = sigmoid(b + w . [router p, ideal factor, margin factor]), see train_v3.py
lookup conf = router max-probability x factor. Original v1 description follows.

Confidence (fixed before tuning):
  router max-probability p, and for a lookup the deployed retriever's top-1 score s:
  lookup: conf = p * s / (s + K)   (BM25 scores are unbounded; K = median top-1 BM25 score over the
                                    TRAIN lookup questions, so conf stays in [0, 1])
          for a TF-IDF / cosine retriever s is already in [0, 1] and conf = p * s
  calc:   conf = p
  oos:    always ESCALATED
Answered = stamp SOURCED or CALCULATED. A calc whose item has no weight escalates regardless of conf.
"""
import math

from calc import calc
from common import rank
from textnorm import tokens


class CourseMate:
    def __init__(self, router_vec, lr, retriever, chunks, K, threshold=0.0, conf_mode="v1", calib=None):
        self.v, self.lr, self.r, self.chunks, self.K, self.t = router_vec, lr, retriever, chunks, K, threshold
        self.mode, self.calib = conf_mode, calib

    @classmethod
    def from_pickle(cls, m, chunks):
        return cls(m["router_vec"], m["lr"], m["retriever"], chunks, m["K"], m["threshold"], m["conf_mode"], m.get("calib"))

    def squash(self, s):
        return s / (s + self.K) if self.r.name == "bm25" else s

    def lookup_factor(self, q, sc, mode=None):
        mode = mode or self.mode
        o = rank(sc)
        s1, s2 = float(sc[o[0]]), float(sc[o[1]])
        if mode == "p":
            return 1.0
        if mode == "margin":
            return s1 / (s1 + s2) if s1 + s2 > 0 else 0.0
        if mode == "ideal":
            ideal = 0.0
            for t in tokens(q):
                ideal += self.r.bm.idf.get(t) or 0
            return min(1.0, s1 / ideal) if ideal > 0 else 0.0
        return self.squash(s1)

    def features(self, pmax, q, sc):
        """Inputs of the v3 confidence model: router max-probability, 'ideal' factor, 'margin' factor."""
        return [pmax, self.lookup_factor(q, sc, "ideal"), self.lookup_factor(q, sc, "margin")]

    def lookup_conf(self, pmax, q, sc):
        if self.mode == "meta":  # v3: logistic model of P(top chunk is right), fitted on out-of-fold predictions
            z = self.calib["intercept"] + sum(w * x for w, x in zip(self.calib["coef"], self.features(pmax, q, sc)))
            return 1.0 / (1.0 + math.exp(-z))
        return pmax * self.lookup_factor(q, sc)

    def answer(self, q, scores=None):
        p = self.lr.predict_proba(self.v.transform([q]))[0]
        classes = list(self.lr.classes_)
        order = sorted(range(len(p)), key=lambda i: (-p[i], i))
        label, pmax = classes[order[0]], float(p[order[0]])
        out = {"label": label, "p": [float(x) for x in p], "margin": float(p[order[0]] - p[order[1]])}
        sc = self.r.scores(q)
        top = rank(sc)[0]
        out["top_chunk"], out["top_score"] = self.chunks[top]["id"], float(sc[top])
        if label == "lookup":
            out["conf"] = self.lookup_conf(pmax, q, sc)
            out["feat"] = self.features(pmax, q, sc)
        elif label == "calc":
            out["conf"] = pmax
        else:
            out["conf"] = pmax
        if label == "oos":
            out.update(stamp="ESCALATED", reason="out of scope")
        elif out["conf"] < self.t:
            out.update(stamp="ESCALATED", reason="low confidence")
        elif label == "lookup":
            out.update(stamp="SOURCED")
        else:
            c = calc(q, scores)
            out.update(c)
        return out


def is_correct_answer(q, out):
    """Correct only if answered with the right content. Called for answered questions."""
    if out["stamp"] == "SOURCED":
        return q["label"] == "lookup" and out["top_chunk"] in q["gold"]
    if out["stamp"] == "CALCULATED":
        return (q["label"] == "calc" and q["expected_stamp"] == "CALCULATED"
                and out["item"] == q["calc_item"] and str(out["score"]) == q["calc_score"])
    return False
