"""Compare text representations for lookup retrieval (TA feedback: compare, don't pick in advance).

Run (from ml/): .venv/bin/python eval_retrieval.py
Methods: BM25 (word tokens), TF-IDF word unigrams, TF-IDF word 1-2-grams, TF-IDF char_wb 3-5-grams,
all-MiniLM-L6-v2 embeddings. Tokeniser axis: word vs sub-word characters vs MiniLM WordPiece.
Metrics per lookup question (gold = every chunk that states the answer):
  Top-k  = 1 if any gold chunk is in the top k, else 0           (k = 1, 3)
  R@k    = |gold in top k| / |gold|                              (k = 1, 3, 5)
  MRR    = 1 / rank of the first gold chunk
Selection rule (fixed before running): highest tune MRR; tie -> higher tune Top-3 -> METHODS order.
The deployable retriever must run in the browser without a model download: if MiniLM wins, the best
lexical method under the same rule is deployed and that substitution is recorded.
Writes results/retrieval.json and results/retrieval_ranks.csv.
"""
import csv
import json

import numpy as np

from common import METHODS, RESULTS, load_chunks, load_questions, make_retriever, rank

B, SEED = 2000, 0


def metrics(order, gold, ids):
    pos = [order.index(ids.index(g)) + 1 for g in gold]
    top = lambda k: float(min(pos) <= k)
    rec = lambda k: sum(p <= k for p in pos) / len(gold)
    return {"top1": top(1), "top3": top(3), "r1": rec(1), "r3": rec(3), "r5": rec(5), "mrr": 1 / min(pos)}


def summarise(rows, ci):
    keys = ["top1", "top3", "r1", "r3", "r5", "mrr"]
    out = {"n": len(rows)}
    rng = np.random.default_rng(SEED)
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k] = float(v.mean())
        if ci:
            boots = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            out[k + "_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def main():
    chunks, qs = load_chunks(), [q for q in load_questions() if q["label"] == "lookup"]
    ids, texts = [c["id"] for c in chunks], [c["text"] for c in chunks]
    res, status, per_q = {}, {}, []
    for m in METHODS:
        try:
            r = make_retriever(m).fit(texts)
        except Exception as e:  # MiniLM missing or not downloadable: record and continue
            status[m] = f"unavailable: {type(e).__name__}: {e}"[:300]
            continue
        status[m] = "ok"
        rows = []
        for q in qs:
            order = rank(r.scores(q["q"]))
            mt = metrics(order, q["gold"], ids)
            rows.append({**mt, "split": q["split"], "kind": q["kind"]})
            per_q.append({"id": q["id"], "method": m, "split": q["split"], "kind": q["kind"],
                          "top1_chunk": ids[order[0]], "top1_gold": int(mt["top1"]),
                          "first_gold_rank": round(1 / mt["mrr"])})
        res[m] = {s: summarise([x for x in rows if x["split"] == s], ci=(s == "test")) for s in ("tune", "test")}
        res[m]["all_lookup"] = summarise(rows, ci=True)
        res[m]["by_kind_all"] = {k: summarise([x for x in rows if x["kind"] == k], ci=False)
                                 for k in sorted({x["kind"] for x in rows})}
    avail = [m for m in METHODS if m in res]
    key = lambda m: (-res[m]["tune"]["mrr"], -res[m]["tune"]["top3"], METHODS.index(m))
    winner = sorted(avail, key=key)[0]
    deploy = sorted([m for m in avail if m != "minilm"], key=key)[0]
    out = {"status": status, "results": res, "winner_on_tune": winner, "deployed": deploy,
           "deploy_note": "" if winner == deploy else
           f"{winner} won on tune; deployed {deploy} (best lexical) so the page runs offline with no model download",
           "selection_rule": "max tune MRR, tie -> tune Top-3 -> order " + ",".join(METHODS)}
    (RESULTS / "retrieval.json").write_text(json.dumps(out, indent=1))
    with (RESULTS / "retrieval_ranks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)
    print(f"{'method':8} {'tuneMRR':>7} | test n  Top-1  Top-3   R@1   R@3   R@5   MRR | all-lookup n Top-1 Top-3")
    for m in avail:
        t, u, a = res[m]["test"], res[m]["tune"], res[m]["all_lookup"]
        print(f"{m:8} {u['mrr']:7.3f} | {t['n']:6} {t['top1']:.3f} {t['top3']:.3f} {t['r1']:.3f} {t['r3']:.3f} "
              f"{t['r5']:.3f} {t['mrr']:.3f} | {a['n']:5} {a['top1']:.3f} {a['top3']:.3f}")
    for m, s in status.items():
        if s != "ok":
            print(m, s)
    print("winner on tune:", winner, "| deployed:", deploy, out["deploy_note"])


if __name__ == "__main__":
    main()
