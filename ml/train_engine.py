"""Train the final engine retriever and set its threshold.
- Fine-tune MiniLM on all public course FAQ pairs + COMP1945 train-split lookup questions (question, gold chunk).
  COMP1945 tune/test questions stay out of training.
- Threshold: highest held-out-course coverage with top-1 precision >= PREC, from finetune_cv.py calibration.
- Question bank for FAQ generation: every public FAQ question + COMP1945 train questions, deduplicated."""
import json, csv, random, os, re
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np, torch
from sentence_transformers import SentenceTransformer
exec(open('reranker_data.py').read())  # R
src = open('finetune_cv.py').read(); exec(src[src.index('def fit('):src.index('def ev(') if 'def ev(' in src else None].split('CAL = []')[0])
random.seed(0); torch.manual_seed(0)
PREC = 0.80
cal = np.array(json.load(open('../results/finetune_cv.json'))['calibration'])
best = None
for t in sorted(set(cal[:, 0])):
    a = cal[:, 0] >= t
    if a.sum() >= 20 and cal[a, 1].mean() >= PREC:
        best = (float(t), float(a.mean()), float(cal[a, 1].mean())); break
chunks = {json.loads(l)['id']: json.loads(l)['text'] for l in open('data/chunks.jsonl')}
split = {r['id']: r['split'] for r in csv.DictReader(open('data/split.csv'))}
qs = list(csv.DictReader(open('data/questions.csv')))
c1945 = [(q['q'], chunks[g]) for q in qs if q['label'] == 'lookup' and split.get(q['id']) == 'train' for g in q['gold_chunk_ids'].split(';')[:1] if g in chunks]
pairs = [(r['q'], r['a']) for r in R] + c1945
m = fit(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'), pairs)
m.save('engine_model/minilm')
bank = sorted({r['q'] for r in R if r['q'].endswith('?') and len(r['q']) <= 120} | {q['q'] for q in qs if q['label'] == 'lookup' and split.get(q['id']) == 'train' and q['kind'] == 'clean'})
json.dump({'threshold': best[0], 'cv_coverage': best[1], 'cv_precision': best[2], 'precision_target': PREC,
           'train_pairs': {'public_faq': len(R), 'comp1945_train': len(c1945)}, 'question_bank': bank}, open('engine_model/engine.json', 'w'), indent=1)
print(f'threshold {best[0]:.3f}: held-out-course coverage {best[1]:.3f}, precision {best[2]:.3f}; trained on {len(R)} public + {len(c1945)} COMP1945 pairs; bank {len(bank)} questions')
