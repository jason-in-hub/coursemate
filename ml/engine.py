"""CourseMate engine: any course's syllabus in, FAQ and answers out. No LLM.
Retriever = MiniLM fine-tuned on public course FAQs (train_engine.py). Threshold from held-out-course CV."""
import json, re, os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np

HERE = Path(__file__).resolve().parent
# not under artifacts/: run_all.sh deletes artifacts/ on every run
MODEL_DIR = HERE / 'engine_model' / 'minilm'
CFG = HERE / 'engine_model' / 'engine.json'


def read_text(name, data):
    """Bytes of a .pdf/.html/.htm/.md/.txt file -> plain text with line breaks kept."""
    n = name.lower()
    if n.endswith('.pdf'):
        import io
        from pypdf import PdfReader
        return '\n'.join((p.extract_text() or '') for p in PdfReader(io.BytesIO(data)).pages)
    t = data.decode('utf-8', errors='ignore')
    if n.endswith(('.html', '.htm')):
        from bs4 import BeautifulSoup
        s = BeautifulSoup(t, 'html.parser')
        for x in s(['script', 'style', 'nav', 'footer']): x.decompose()
        return s.get_text('\n')
    return t


def chunk(text, max_words=120, overlap=20):
    """Paragraph-aware chunks: short lines that look like headings start a new section; long sections are windowed."""
    secs, cur, head = [], [], ''
    for line in (l.strip() for l in text.splitlines()):
        if not line: continue
        is_head = len(line.split()) <= 8 and not line.endswith(('.', ',', ';')) and line[:1].isupper()
        if is_head and cur:
            secs.append((head, ' '.join(cur))); cur = []
        if is_head: head = line
        else: cur.append(line)
    if cur: secs.append((head, ' '.join(cur)))
    out = []
    for h, body in secs:
        w = body.split()
        for i in range(0, max(len(w) - overlap, 1), max_words - overlap):
            piece = ' '.join(w[i:i + max_words])
            if len(piece.split()) >= 5: out.append({'section': h, 'text': (h + ': ' if h else '') + piece})
    return out


class Engine:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(str(MODEL_DIR), device='cpu')
        cfg = json.loads(CFG.read_text())
        # drop questions tied to one course (codes like "CS 289", "Data 8", "COMP1945", or named tools)
        specific = re.compile(r'\b[A-Z]{2,5} ?\d{2,4}\b|\bData ?8\b|check50|submit50|edX|Ed\b|Gradescope|JupyterHub|Piazza|DataHub|Canvas|okpy|\bOH\b', re.I)
        self.t = cfg['threshold']
        self.bank = [q for q in cfg['question_bank'] if not specific.search(q)]
        self.bank_emb = self.m.encode(self.bank, normalize_embeddings=True)
        self.chunks, self.emb, self.faq = [], None, []

    def ingest(self, name, data, max_faq=25):
        self.chunks = chunk(read_text(name, data))
        if not self.chunks: raise ValueError('no text found in ' + name)
        self.emb = self.m.encode([c['text'] for c in self.chunks], normalize_embeddings=True)
        S = self.bank_emb @ self.emb.T
        best, sc = S.argmax(1), S.max(1)
        faq, used = [], set()
        for i in np.argsort(-sc):  # best-scoring bank question per chunk, above threshold
            if sc[i] < self.t or len(faq) >= max_faq: break
            if best[i] in used: continue
            used.add(int(best[i]))
            faq.append({'q': self.bank[i], 'a': self.chunks[best[i]]['text'], 'section': self.chunks[best[i]]['section'], 'conf': round(float(sc[i]), 3)})
        self.faq = faq
        return {'chunks': len(self.chunks), 'faq': faq, 'threshold': self.t}

    def ask(self, q):
        if self.emb is None: raise ValueError('no syllabus loaded')
        e = self.m.encode([q], normalize_embeddings=True)[0]
        s = self.emb @ e; i = int(s.argmax()); conf = float(s[i])
        if conf < self.t:
            return {'stamp': 'ESCALATED', 'answer': 'Please ask the TA', 'conf': round(conf, 3), 'reason': 'low confidence'}
        c = self.chunks[i]
        return {'stamp': 'SOURCED', 'answer': c['text'], 'section': c['section'], 'conf': round(conf, 3)}
