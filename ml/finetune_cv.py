
"""Fine-tune MiniLM on public course FAQ (question, answer) pairs; grouped K-fold by course.
Query = FAQ question, candidates = all answers of the same held-out course."""
import json, re, os, sys, time, random
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(os.cpu_count() or 4)

def fit(m, pairs, epochs=3, bs=32, lr=2e-5, scale=20.0):
    """Plain-torch MultipleNegativesRankingLoss on CPU (the HF Trainer picked MPS and ran out of memory)."""
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train()
    for _ in range(epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), bs):
            b = pairs[i:i + bs]
            enc = lambda xs: m({k: v for k, v in m.tokenize(xs).items()})['sentence_embedding']
            q = torch.nn.functional.normalize(enc([x[0] for x in b]), dim=-1); a = torch.nn.functional.normalize(enc([x[1] for x in b]), dim=-1)
            loss = torch.nn.functional.cross_entropy(scale * q @ a.T, torch.arange(len(b)))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m
random.seed(0); np.random.seed(0); torch.manual_seed(0)
BASE = 'sentence-transformers/all-MiniLM-L6-v2'
exec(open('reranker_data.py').read())  # R, G (same rows and course groups as train_reranker.py)
CAL = []  # (top-1 cosine, top-1 correct) per held-out query: sets the engine threshold
def ev(m, rows, cal=False):
    out = []
    for g in sorted(set(r['g'] for r in rows)):
        S = [r for r in rows if r['g'] == g]
        eq = m.encode([r['q'] for r in S], normalize_embeddings=True); ea = m.encode([r['a'] for r in S], normalize_embeddings=True)
        sim = eq @ ea.T
        out += [int((sim[i] > sim[i, i]).sum()) + 1 for i in range(len(S))]
        if cal: CAL.extend((float(sim[i].max()), int(sim[i].argmax() == i)) for i in range(len(S)))
    return np.array(out)
summ = lambda rk: dict(top1=round(float((rk == 1).mean()), 3), top3=round(float((rk <= 3).mean()), 3), mrr=round(float((1 / rk).mean()), 3), n=len(rk))
K = 4; rng = random.Random(0); gs = G[:]; rng.shuffle(gs); folds = [gs[i::K] for i in range(K)]
base = SentenceTransformer(BASE, device='cpu'); rb, rf = [], []
for k, test_g in enumerate(folds):
    t0 = time.time()
    tr = [r for r in R if r['g'] not in test_g]; te = [r for r in R if r['g'] in test_g]
    m = SentenceTransformer(BASE, device='cpu')
    fit(m, [(r['q'], r['a']) for r in tr])
    rb.append(ev(base, te)); rf.append(ev(m, te, cal=True))
    print(f'fold {k}: {len(tr)} train / {len(te)} test, {time.time()-t0:.0f}s', summ(rb[-1])['top1'], '->', summ(rf[-1])['top1'], file=sys.stderr, flush=True)
out = {'calibration': CAL, 'minilm_base': summ(np.concatenate(rb)), 'minilm_finetuned': summ(np.concatenate(rf)), 'folds': folds}
print(json.dumps({k: out[k] for k in ['minilm_base', 'minilm_finetuned']}, indent=1))
json.dump(out, open('../results/finetune_cv.json', 'w'), indent=1)
