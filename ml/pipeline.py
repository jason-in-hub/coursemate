"""End-to-end answer(): router -> (retrieval | calculator | escalate) -> confidence threshold.

Confidence (fixed before tuning):
  router max-probability p, and for a lookup the deployed retriever's top-1 score s:
  lookup: conf = p * s / (s + K)   (BM25 scores are unbounded; K = median top-1 BM25 score over the
                                    TRAIN lookup questions, so conf stays in [0, 1])
          for a TF-IDF / cosine retriever s is already in [0, 1] and conf = p * s
  calc:   conf = p
  oos:    always ESCALATED
Answered = stamp SOURCED or CALCULATED. A calc whose item has no weight escalates regardless of conf.
"""
from calc import calc
from common import rank


class CourseMate:
    def __init__(self, router_vec, lr, retriever, chunks, K, threshold=0.0):
        self.v, self.lr, self.r, self.chunks, self.K, self.t = router_vec, lr, retriever, chunks, K, threshold

    def squash(self, s):
        return s / (s + self.K) if self.r.name == "bm25" else s

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
            out["conf"] = pmax * self.squash(float(sc[top]))
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
