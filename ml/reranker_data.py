import json, re
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
