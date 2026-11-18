"""Second-opinion rules for the 1945 page (index.html answer2). The fine-tuned MiniLM (engine_model/minilm) must rank the
same passage first as BM25 ("agree"). A: answer a low-confidence lookup on agreement. B: also a calc-routed question with no
project item named. C: also an oos-routed question with router max-probability < 0.5. Chosen on tune; reported on test and on
the demo questions (gold may include ESC = escalating is also acceptable).  Run: .venv/bin/python eval_agree.py -> results/agree.json"""

import sys, json, csv, pickle, numpy as np
from pathlib import Path
H=Path(__file__).resolve().parent; sys.path.insert(0,str(H))
from common import load_chunks, load_questions
from pipeline import CourseMate
from engine import Engine
ch=load_chunks(); ids=[c['id'] for c in ch]
cm=CourseMate.from_pickle(pickle.load(open(H/'artifacts'/'model.pkl','rb')), ch)
m=Engine().m
C=m.encode([(c.get('source','')+': '+c['text']) for c in ch], normalize_embeddings=True)
rows=load_questions()
train=[r for r in rows if r['split']=='train' and r['label']=='lookup']
TQ=m.encode([r['q'] for r in train], normalize_embeddings=True)
def votes(q):
    o=cm.answer(q); e=m.encode([q],normalize_embeddings=True)[0]
    s=C@e; mt=ids[int(s.argmax())]
    qs=TQ@e; k=int(qs.argmax()); qb=train[k]['gold'][0]; qbs=float(qs[k]); qgold=train[k]['gold']
    return o, mt, qb, qbs, qgold

def decide(rule, q):
    o, mt, qb, qbs, qgold = votes(q)
    cur=('S',o['top_chunk']) if o['stamp']=='SOURCED' else (('C',None) if o['stamp']=='CALCULATED' else ('E',None))
    if cur[0]!='E': return cur
    b=o['top_chunk']; agree = mt==b
    if o['label']=='lookup' and o.get('reason')=='low confidence': return ('S',b) if (rule!='current' and agree) else cur
    if rule in ('B','C') and o['label']=='calc' and o.get('reason')=='no project item named': return ('S',b) if agree else cur
    if rule=='C' and o['label']=='oos' and max(o['p'])<0.5: return ('S',b) if agree else cur
    return cur
RULES=['current','A','B','C']

def score(items):
    out={}
    for rule in RULES:
        ans=right=0
        for r in items:
            st,top=decide(rule,r['q'])
            if st=='E' and 'ESC' in r['gold']: continue
            if r['gold'][0]=='ESC': ans+= st!='E'
            elif r['gold'][0]=='CALC': ans+= st!='E'; right+= st=='C'
            else: ans+= st!='E'; right+= (st=='S' and top in r['gold'])
        out[rule]=(ans,right,ans-right)
    return out
lab=[dict(q=r['q'],gold=(r['gold'] if r['label']=='lookup' else (['ESC'] if r['label']=='oos' else ['CALC'])),split=r['split']) for r in rows]
demo=[dict(q=r['q'],gold=r['gold'].split(';')) for r in csv.DictReader(open(H/'data'/'demo_questions.csv'))]
RES={}
for name,items in [('tune',[r for r in lab if r['split']=='tune']),('test',[r for r in lab if r['split']=='test']),('demo',demo)]:
    print(name,len(items))
    for k,(a,rt,w) in score(items).items():
        RES.setdefault(name,{})[k]=dict(answered=a,right=rt,wrong=w); print('   %-10s answered %2d right %2d wrong %2d'%(k,a,rt,w))
json.dump(RES, open(H.parent/'results'/'agree.json','w'), indent=1); print('wrote results/agree.json')
