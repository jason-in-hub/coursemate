
import json,re,sys
from pathlib import Path
from bs4 import BeautifulSoup
D=Path(sys.argv[1]); src=json.load(open(sys.argv[2])); out=[]
def clean(s): return re.sub(r'\s+',' ',s).strip().rstrip(ANCH).strip()
ANCH='#¶ ' + chr(0x200b) + chr(0xf0c1) + chr(0x1f517)
for s in src:
    raw=(D/f"{int(s['i']):03d}.html").read_text(errors='ignore')
    if s['url'].endswith('.md'):
        import markdown; raw=markdown.markdown(raw)
    soup=BeautifulSoup(raw,'html.parser')
    for t in soup(['script','style','nav','footer','header']): t.decompose()
    n=0
    # headings / summary / dt / strong-paragraph that end with '?'
    for h in soup.find_all(['h1','h2','h3','h4','h5','summary','dt','strong','b']):
        q=clean(h.get_text(' '))
        if not q.endswith('?') or not (8<=len(q)<=200): continue
        # answer: following siblings until next same-kind tag
        parts=[]
        if h.name=='summary':
            parts=[clean(x.get_text(' ')) for x in h.parent.find_all(recursive=False) if x is not h]
        elif h.name=='dt':
            dd=h.find_next_sibling('dd'); parts=[clean(dd.get_text(' '))] if dd else []
        else:
            node=h if h.name.startswith('h') else (h.parent if h.parent.name=='p' else h)
            if h.name in('strong','b') and h.parent.name=='p':
                rest=clean(h.parent.get_text(' '))[len(q):].strip()
                if rest: parts.append(rest)
            for sib in node.find_next_siblings():
                if sib.name and (sib.name.startswith('h') or sib.name=='details'): break
                if sib.name=='p' and sib.find(['strong','b']) and clean(sib.find(['strong','b']).get_text(' ')).endswith('?'): break
                parts.append(clean(sib.get_text(' ')))
                if sum(len(p) for p in parts)>1500: break
        a=clean(' '.join(p for p in parts if p))
        if len(a)<15: continue
        out.append({'src':s['url'],'q':q,'a':a[:1500]}); n+=1
    print(n, s['url'], file=sys.stderr)
# dedupe
seen=set(); ded=[]
for r in out:
    k=(r['src'],r['q'].lower())
    if k in seen: continue
    seen.add(k); ded.append(r)
json.dump(ded,open(sys.argv[3],'w'),indent=1,ensure_ascii=False)
print('total',len(ded),file=sys.stderr)
