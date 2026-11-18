"""Reference outputs of the Python engine (engine.py, as served by serve.py) for the browser parity check (parity_engine.cjs).
Usage: python engine_dump.py out.json file1 [file2 ...]   (questions as JSON list on stdin)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine

E = Engine()
qs = json.load(sys.stdin)
out = {'threshold': E.t, 'bank': E.bank, 'bank_emb': E.bank_emb.tolist(), 'files': {}}
for f in sys.argv[2:]:
    r = E.ingest(Path(f).name, Path(f).read_bytes())
    allq = list(dict.fromkeys(qs + [x['q'] for x in r['faq']]))
    out['files'][f] = {'chunks': E.chunks, 'chunk_emb': E.emb.tolist(), 'faq': r['faq'], 'n': r['chunks'],
                       'asks': {q: E.ask(q) for q in allq}, 'q_emb': E.m.encode(allq, normalize_embeddings=True).tolist()}
json.dump(out, open(sys.argv[1], 'w'))
