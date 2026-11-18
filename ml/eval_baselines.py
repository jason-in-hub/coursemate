"""Compare CourseMate with standard FAQ-bot baselines on the same questions. Each baseline answers its top passage when the
score clears a threshold chosen on the tune split (max right + correct escalations - wrong). Run: .venv/bin/python eval_baselines.py"""

import sys, csv, pickle, json, numpy as np
from pathlib import Path
H=Path(__file__).resolve().parent; sys.path.insert(0,str(H))
from common import load_chunks, load_questions, TfidfRetriever, MiniLMRetriever
from pipeline import CourseMate
from engine import Engine
ch=load_chunks(); ids=[c['id'] for c in ch]; texts=[c['text'] for c in ch]
cm=CourseMate.from_pickle(pickle.load(open(H/'artifacts'/'model.pkl','rb')), ch)
ft=Engine().m
C=ft.encode([(c.get('source','')+': '+c['text']) for c in ch], normalize_embeddings=True)
tf=TfidfRetriever('word1').fit(texts); ots=MiniLMRetriever().fit(texts)
rows=load_questions()
def lab(r): return r['gold'] if r['label']=='lookup' else (['ESC'] if r['label']=='oos' else ['CALC'])
L={s:[dict(q=r['q'],gold=lab(r)) for r in rows if r['split']==s] for s in ('tune','test')}
L['demo']=[dict(q=r['q'],gold=r['gold'].split(';')) for r in csv.DictReader(open(H/'data'/'demo_questions.csv'))]
L['demo-blind']=[dict(q=r['q'],gold=r['gold'].split(';')) for r in csv.DictReader(open(H/'data'/'demo_questions_blind_full.csv'))]  # independent full-text relabel
SPL=('tune','test','demo','demo-blind')
def ours(q):
    o=cm.answer(q)
    if o['stamp']=='SOURCED': return ('S',o['top_chunk'])
    if o['stamp']=='CALCULATED': return ('C',None)
    e=ft.encode([q],normalize_embeddings=True)[0]; mt=ids[int((C@e).argmax())]
    low=o['label']=='lookup' and o.get('reason')=='low confidence'
    cni=o['label']=='calc' and o.get('reason')=='no project item named'
    so=o['label']=='oos' and max(o['p'])<0.5
    return ('S',o['top_chunk']) if (low or cni or so) and mt==o['top_chunk'] else ('E',None)
def bm25only(q):
    o=cm.answer(q); return ('S',o['top_chunk']) if o['stamp']=='SOURCED' else (('C',None) if o['stamp']=='CALCULATED' else ('E',None))
def thr(ret):
    def f(q,t):
        s=ret.scores(q); i=int(np.argmax(s)); return ('S',ids[i]) if s[i]>=t else ('E',None)
    return f
def score(dec, items):
    right=wrong=escok=0
    for r in items:
        st,top=dec(r['q'])
        if st=='E': escok+= ('ESC' in r['gold'])
        elif st=='S':
            if top in r['gold']: right+=1
            else: wrong+=1
        else: right+= ('CALC' in r['gold']); wrong+= ('CALC' not in r['gold'])
    return right,wrong,escok,len(items)
res={}
for name,ret in [('TF-IDF cosine (classic FAQ bot)',tf),('off-the-shelf all-MiniLM-L6-v2',ots)]:
    f=thr(ret); best=None
    for t in np.unique(np.round([max(ret.scores(r['q'])) for r in L['tune']],4)):
        rt,w,e,n=score(lambda q:f(q,t),L['tune']); key=(rt+e-w, -w)
        if best is None or key>best[0]: best=(key,t)
    t=best[1]; res[name]={'threshold (chosen on tune: max right+correct escalations-wrong)':float(t)}
    for s in SPL: res[name][s]=score(lambda q:f(q,t),L[s])
    # upper bound for the baseline: threshold picked on each evaluation split itself (peeks at the answers)
    for s in ('test','demo','demo-blind'):
        sc=[]
        for t2 in np.unique(np.round([max(ret.scores(r['q'])) for r in L[s]],4)).tolist()+[9e9]:
            rt,w,e,n=score(lambda q:f(q,t2),L[s]); sc.append(((rt+e-w,-w),(rt,w,e,n)))
        res[name][s+' (oracle threshold)']=max(sc)[1]
for name,dec in [('BM25 pipeline only (v2)',bm25only),('CourseMate (BM25 + router + MiniLM second opinion)',ours)]:
    res[name]={s:score(dec,L[s]) for s in SPL}
print('right/wrong/correct-escalations/n')
for k,v in res.items():
    print(k)
    for s,x in v.items(): print('   %-34s %s'%(s,x))
json.dump(res,open(H.parent/'results'/'baselines.json','w'),indent=1,default=list)
