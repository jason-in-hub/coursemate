"""Shared loaders and retrievers used by every ML script (run from ml/)."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from textnorm import ANALYZERS, word1

HERE = Path(__file__).resolve().parent
DATA, RESULTS, ART = HERE / "data", HERE.parent / "results", HERE / "artifacts"
RESULTS.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)
DECIMALS = 10  # scores are rounded before ranking so Python and JS break near-ties the same way


def load_chunks():
    return [json.loads(l) for l in (DATA / "chunks.jsonl").open(encoding="utf-8") if l.strip()]


def load_questions(aug=False):
    """The 167 frozen questions. aug=True also returns data/train_aug.csv as split "aug" (v3 training
    data, written without sight of questions.csv). An aug row is dropped if its token string equals
    that of any tune/test question or any holdout question, so no evaluation question is trained on."""
    rows = list(csv.DictReader((DATA / "questions.csv").open(encoding="utf-8")))
    split = {r["id"]: r["split"] for r in csv.DictReader((DATA / "split.csv").open(encoding="utf-8"))}
    for r in rows:
        r["split"] = split[r["id"]]
    if aug and (DATA / "train_aug.csv").exists():
        key = lambda s: " ".join(word1(s))
        banned = {key(r["q"]) for r in rows if r["split"] != "train"}
        if (DATA / "holdout.csv").exists():
            banned |= {key(r["q"]) for r in csv.DictReader((DATA / "holdout.csv").open(encoding="utf-8"))}
        for r in csv.DictReader((DATA / "train_aug.csv").open(encoding="utf-8")):
            if key(r["q"]) not in banned:
                r["split"] = "aug"
                rows.append(r)
    for r in rows:
        r["gold"] = [g for g in r["gold_chunk_ids"].split(";") if g]
    return rows


def rank(scores):
    """Indices by score descending, ties by chunk index ascending."""
    s = np.round(np.asarray(scores, dtype=float), DECIMALS)
    return sorted(range(len(s)), key=lambda i: (-s[i], i))


class TfidfRetriever:
    def __init__(self, analyzer):
        self.name = analyzer
        self.vec = TfidfVectorizer(analyzer=ANALYZERS[analyzer])  # smooth idf, raw tf, l2 norm

    def fit(self, texts):
        self.X = self.vec.fit_transform(texts)
        return self

    def scores(self, q):
        return (self.X @ self.vec.transform([q]).T).toarray().ravel()


class BM25Retriever:
    name = "bm25"

    def fit(self, texts, extra=None):
        """extra (v3): per chunk, a list of training questions whose gold includes that chunk; their
        tokens are appended to the chunk's document, so the index learns student vocabulary."""
        from rank_bm25 import BM25Okapi
        docs = [word1(t) for t in texts]
        if extra:
            docs = [d + [w for x in e for w in word1(x)] for d, e in zip(docs, extra)]
        self.bm = BM25Okapi(docs)  # k1=1.5, b=0.75, same tokeniser as word1
        return self

    def scores(self, q):
        return self.bm.get_scores(word1(q))


class MiniLMRetriever:
    """all-MiniLM-L6-v2 (WordPiece tokeniser, raw text in, cosine of normalised embeddings)."""
    name = "minilm"

    def fit(self, texts):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the local cache; no network if cached
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.E = self.m.encode(texts, normalize_embeddings=True)
        return self

    def scores(self, q):
        return self.E @ self.m.encode([q], normalize_embeddings=True)[0]


METHODS = ["bm25", "word1", "word12", "char35", "minilm"]  # this order is also the tie-break order


def make_retriever(name):
    return {"bm25": BM25Retriever, "minilm": MiniLMRetriever}.get(name, lambda: TfidfRetriever(name))()
