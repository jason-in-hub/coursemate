
"""Learned reranker: logistic regression over similarity features, trained on public course FAQs.
Each FAQ question is a query; candidates = all answers of the same course; label 1 = its own answer.
Leave-one-course-group-out CV. Prints Top-1/Top-3/MRR per method."""
import json, re, sys, os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

P = json.load(open('data/external/faq_pairs.json'))
strip = lambda s: re.sub(r'^(Q\d*[:.)]|A[:.)]|\d+[.)])\s*', '', s.strip()).strip()
def group(u):
    for k in ['cs50','data8','eecs189','ubc-dsci','ubc-stat','dsc10','csw8','cis120','cs230','cs246','cs25/','smithcsc210','neu-pdi','cs161','nipunbatra','introcomp','thomaselove','cs16','qmul','cs242']:
        if k in u: return k
    return u
R, seen = [], set()
for r in P:
    q, a, g = strip(r['q']), strip(r['a']), group(r['src'])
    if (g, q.lower()) in seen: continue
    seen.add((g, q.lower())); R.append(dict(q=q, a=a, g=g))
G = sorted(set(r['g'] for r in R)); G = [g for g in G if sum(r['g'] == g for r in R) >= 5]
R = [r for r in R if r['g'] in G]
STOP = set('a an the is are do does i my can to of in for on and or what how when where will be it this course class you your if there any with we am get should'.split())
tok = lambda s: [w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in STOP]

use_emb = '--no-emb' not in sys.argv
if use_emb:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    EQ = m.encode([r['q'] for r in R], normalize_embeddings=True); EA = m.encode([r['a'] for r in R], normalize_embeddings=True)

def feats(idx):
    """Feature matrix for every (question i, answer j) pair within one course."""
    Q = [R[i]['q'] for i in idx]; A = [R[i]['a'] for i in idx]
    bm = BM25Okapi([tok(a) for a in A])
    w = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit(A + Q)
    cw = cosine_similarity(w.transform(Q), w.transform(A))
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True).fit(A + Q)
    cc = cosine_similarity(c.transform(Q), c.transform(A))
    ce = EQ[idx] @ EA[idx].T if use_emb else None
    X, y, qid = [], [], []
    for a_, i in enumerate(idx):
        qt = tok(R[i]['q']); s = np.array(bm.get_scores(qt)) if qt else np.zeros(len(idx))
        sn = s / (s.max() + 1e-9)
        for b_, j in enumerate(idx):
            at = set(tok(R[j]['a'])); ov = len(set(qt) & at) / (len(set(qt)) or 1)
            row = [sn[b_], s[b_] - np.sort(s)[-2] if len(s) > 1 else 0, cw[a_, b_], cw[a_, b_] - cw[a_].max(), cc[a_, b_], cc[a_, b_] - cc[a_].max(), ov]
            if use_emb: row += [ce[a_, b_], ce[a_, b_] - ce[a_].max()]
            X.append(row); y.append(int(i == j)); qid.append(i)
    return np.array(X), np.array(y), np.array(qid)

F = {g: feats([k for k, r in enumerate(R) if r['g'] == g]) for g in G}
def ranks_from(score, y, qid):
    out = []
    for q in np.unique(qid):
        m_ = qid == q; s = score[m_]; gold = np.where(y[m_] == 1)[0][0]
        out.append(int((s > s[gold]).sum()) + 1)
    return np.array(out)
summ = lambda rk: dict(top1=round(float((rk == 1).mean()), 3), top3=round(float((rk <= 3).mean()), 3), mrr=round(float((1 / rk).mean()), 3), n=len(rk))
cols = {'bm25': 0, 'tfidf_word': 2, 'tfidf_char': 4}
if use_emb: cols['minilm'] = 7
res = {k: [] for k in list(cols) + ['logreg']}
calib = []
for g in G:
    Xt = np.vstack([F[h][0] for h in G if h != g]); yt = np.concatenate([F[h][1] for h in G if h != g])
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(Xt, yt)
    X, y, qid = F[g]
    for k, c in cols.items(): res[k].append(ranks_from(X[:, c], y, qid))
    p = lr.predict_proba(X)[:, 1]; res['logreg'].append(ranks_from(p, y, qid))
    for q in np.unique(qid):
        m_ = qid == q; j = np.argmax(p[m_]); calib.append((float(p[m_][j]), int(y[m_][j] == 1)))
out = {k: summ(np.concatenate(v)) for k, v in res.items()}
calib = np.array(calib)
for t in (0.5, 0.7, 0.9):
    a = calib[:, 0] >= t
    out[f'logreg_p>={t}'] = dict(answered=round(float(a.mean()), 3), precision=round(float(calib[a, 1].mean()), 3) if a.any() else None)
out['n_courses'] = len(G); out['n_questions'] = len(R)
print(json.dumps(out, indent=1))
lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(np.vstack([F[h][0] for h in G]), np.concatenate([F[h][1] for h in G]))
json.dump({'features': ['bm25_norm', 'bm25_gap', 'tfidf_word', 'tfidf_word_gap', 'tfidf_char', 'tfidf_char_gap', 'overlap'] + (['minilm', 'minilm_gap'] if use_emb else []),
           'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]), 'cv': out}, open('../results/reranker' + ('' if use_emb else '_noemb') + '.json', 'w'), indent=1)
