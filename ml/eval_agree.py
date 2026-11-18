"""Agreement rule for the 1945 page (index.html answer2): a lookup that BM25 escalates for low confidence is answered
when the fine-tuned MiniLM (engine_model/minilm) ranks the same passage first. Variants that also answer on a MiniLM
cosine threshold are compared; the rule is chosen on the tune split (fewest wrong answers, then most right).
Run: .venv/bin/python eval_agree.py  -> results/agree.json"""

import sys, json, csv, pickle, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import load_chunks, load_questions
from pipeline import CourseMate
from engine import Engine
ch=load_chunks(); ids=[c['id'] for c in ch]
cm=CourseMate.from_pickle(pickle.load(open(__import__('pathlib').Path(__file__).resolve().parent/'artifacts'/'model.pkl','rb')), ch)
m=Engine().m
C=m.encode([(c.get('source','')+': '+c['text']) for c in ch], normalize_embeddings=True)
def mtop(q):
    s=C@m.encode([q],normalize_embeddings=True)[0]; o=np.argsort(-s); return ids[o[0]], float(s[o[0]]), [ids[i] for i in o[:3]]
rows=load_questions()
demo=[dict(q=r['q'],gold=r['gold'].split(';'),split='demo-'+r['seen']) for r in csv.DictReader(open(__import__('pathlib').Path(__file__).resolve().parent/'data'/'demo_questions.csv'))]
lab=[dict(q=r['q'],gold=(r['gold'] if r['label']=='lookup' else (['ESC'] if r['label']=='oos' else ['CALC'])),split=r['split'],label=r['label']) for r in rows]
def rules(q):
    o=cm.answer(q); mt,mc,m3=mtop(q)
    cur=('S',o['top_chunk']) if o['stamp']=='SOURCED' else (('C',None) if o['stamp']=='CALCULATED' else ('E',None))
    out={'current':cur}
    agree = o['label']!='oos' and o['top_chunk']==mt and o['label']!='calc'
    out['agree']= cur if cur[0]!='E' else (('S',mt) if agree else cur)
    for t in (0.45,0.5,0.55):
        hi = o['label']!='oos' and o['label']!='calc' and mc>=t
        out['agree|mini>=%.2f'%t]= out['agree'] if out['agree'][0]!='E' else (('S',mt) if hi else cur)
    return out
def score(items):
    agg={}
    for r in items:
        for k,(st,top) in rules(r['q']).items():
            a=agg.setdefault(k,[0,0,0,0])  # answered, right answers, escalated-correctly, n
            a[3]+=1
            if r['gold'][0]=='ESC': a[2]+= st=='E'; a[0]+= st!='E'
            elif r['gold'][0]=='CALC': a[0]+= st!='E'; a[1]+= st=='C'
            else:
                a[0]+= st!='E'; a[1]+= (st=='S' and top in r['gold'])
    return agg
RES={}
for name,items in [('tune',[r for r in lab if r['split']=='tune']),('test',[r for r in lab if r['split']=='test']),('demo',demo),('demo-new',[r for r in demo if r['split']=='demo-new'])]:
    print(name, len(items))
    for k,(ans,right,escok,n) in score(items).items():
        nonesc=sum(1 for r in items if r['gold'][0]!='ESC'); wrong=ans-right
        RES.setdefault(name,{})[k]=dict(answered=ans,right=right,wrong=wrong)
        print('  %-20s answered %2d  right %2d  wrong-answers %2d  precision %.2f  oos-escalated %d/%d'%(k,ans,right,wrong,right/max(ans,1),escok,n-nonesc))
json.dump(RES, open(__import__('pathlib').Path(__file__).resolve().parent.parent/'results'/'agree.json','w'), indent=1)
print('wrote results/agree.json')
