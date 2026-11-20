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


HMARK = '\N{SYMBOL FOR RECORD SEPARATOR}'  # marks a heading line that came from an HTML heading tag
HEADS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dt', 'summary', 'caption']
INLINE = ['a', 'b', 'strong', 'em', 'i', 'u', 'code', 'span', 'small', 'sup', 'sub', 'abbr', 'mark', 'kbd', 'var', 'q', 'cite', 'time', 'font']


def read_text(name, data, inline=True):
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
        if inline:  # v2: real heading tags (and a bold lead-in ending in ':') become marked heading lines
            for h in s.find_all(HEADS):  # heading text on one marked line, even if the source wraps it
                t = ' '.join(h.get_text(' ').split()); h.clear(); h.append('\n' + HMARK + t + '\n')
            for b in s.find_all(['strong', 'b']):
                t = b.get_text(' ').strip()
                if t.endswith(':') and len(t.split()) <= 8 and b.parent and b.parent.name in ('p', 'li') and b.parent.get_text().strip().startswith(t):
                    b.clear(); b.append('\n' + HMARK + ' '.join(t.split()) + '\n')
        if inline:  # v2: links, bold, code etc. stay inside their sentence instead of becoming one-line "headings"
            for x in s.find_all(INLINE): x.unwrap()
            s = BeautifulSoup(str(s), 'html.parser')  # merge the text nodes that unwrap left side by side
        return s.get_text('\n')
    return t


SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9("\N{LEFT DOUBLE QUOTATION MARK}])')


def chunk(text, max_words=120, overlap=20, mode='v1'):
    if mode == 'v2': return chunk_v2(text, max_words)
    return chunk_v1(text, max_words, overlap)


def chunk_v2(text, max_words=80):
    """Sections from marked HTML headings when the file has them (else the v1 heading heuristic); each section body is
    packed sentence by sentence into windows of <= max_words; a short last sentence is repeated at the next window's start."""
    marked = HMARK in text
    secs, cur, head = [], [], ''
    for line in (l.strip() for l in text.splitlines()):
        if not line: continue
        if marked:
            is_head = line.startswith(HMARK); line = line.lstrip(HMARK).strip()
            if not line: continue
        else:
            is_head = len(line.split()) <= 8 and not line.endswith(('.', ',', ';')) and line[:1].isupper()
        if is_head and cur:
            secs.append((head, ' '.join(cur))); cur = []
        if is_head: head = line.rstrip(':').strip()
        else: cur.append(line)
    if cur: secs.append((head, ' '.join(cur)))
    out = []
    for h, body in secs:
        pieces, win = [], []
        for s in SENT.split(' '.join(body.split())):
            if win and len(' '.join(win + [s]).split()) > max_words:
                pieces.append(' '.join(win)); win = win[-1:] if len(win[-1].split()) < max_words // 2 else []
            win.append(s)
        if win: pieces.append(' '.join(win))
        for p in pieces:
            w = p.split()
            for i in range(0, len(w), max_words):  # one very long "sentence" (tables, lists): hard split
                piece = ' '.join(w[i:i + max_words])
                if len(piece.split()) >= 5: out.append({'section': h, 'text': (h + ': ' if h else '') + piece})
    return out


def chunk_v1(text, max_words=120, overlap=20):
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
    def __init__(self, model_dir=None, faq_mode='v2', inline=True, chunk_mode='v2', max_words=120, mutual=3, cue=True, min_words=12):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(str(model_dir or MODEL_DIR), device='cpu')
        self.faq_mode, self.inline, self.chunk_mode, self.max_words, self.mutual, self.cue, self.min_words = faq_mode, inline, chunk_mode, max_words, mutual, cue, min_words
        cfg = json.loads(CFG.read_text())
        # drop questions tied to one course (codes like "CS 289", "Data 8", "COMP1945", or named tools)
        specific = re.compile(r'\b[A-Z]{2,5} ?\d{2,4}\b|\bData ?8\b|check50|submit50|edX|Ed\b|Gradescope|JupyterHub|Piazza|DataHub|Canvas|okpy|\bOH\b', re.I)
        self.t = cfg['threshold']
        self.bank = [q for q in cfg['question_bank'] if not specific.search(q)]
        self._bank_emb = None  # legacy bank, only used by the v1 FAQ path; embedded on first use
        self.chunks, self.emb, self.faq = [], None, []
        # v2: canonical student intents (data/intents.json + phrasings mined from scraped FAQs, build_intents.py)
        self.intents = cfg.get('intents', [])
        self.t_faq, self.kw_bonus = cfg.get('faq_threshold', self.t), cfg.get('faq_kw_bonus', 0.0)
        ph = [(k, p) for k, it in enumerate(self.intents) for p in [it['q']] + it['para']]
        self.ph_intent = np.array([k for k, _ in ph], dtype=int)
        self.ph_emb = self.m.encode([p for _, p in ph], normalize_embeddings=True) if ph else None

    def ingest(self, name, data, max_faq=25):
        self.chunks = chunk(read_text(name, data, self.inline), self.max_words if self.chunk_mode == 'v2' else 120, mode=self.chunk_mode)
        if not self.chunks: raise ValueError('no text found in ' + name)
        self.emb = self.m.encode([c['text'] for c in self.chunks], normalize_embeddings=True)
        if self.faq_mode == 'v2' and self.intents:
            self.faq = self.faq_v2(max_faq)
            return {'chunks': len(self.chunks), 'faq': self.faq, 'threshold': self.t}
        if self._bank_emb is None: self._bank_emb = self.m.encode(self.bank, normalize_embeddings=True)
        S = self._bank_emb @ self.emb.T
        best, sc = S.argmax(1), S.max(1)
        faq, used = [], set()
        for i in np.argsort(-sc):  # best-scoring bank question per chunk, above threshold
            if sc[i] < self.t or len(faq) >= max_faq: break
            if best[i] in used: continue
            used.add(int(best[i]))
            faq.append({'q': self.bank[i], 'a': self.chunks[best[i]]['text'], 'section': self.chunks[best[i]]['section'], 'conf': round(float(sc[i]), 3)})
        self.faq = faq
        return {'chunks': len(self.chunks), 'faq': faq, 'threshold': self.t}

    def faq_v2(self, max_faq=25):
        """One FAQ item per intent the syllabus answers: intent score on a chunk = best cosine over the intent's
        phrasings, plus a bonus when the chunk's section heading names the topic. Several intents may share a chunk."""
        P = self.ph_emb @ self.emb.T  # phrasings x chunks
        K, C = len(self.intents), len(self.chunks)
        S = np.full((K, C), -1.0)
        for k in range(K):
            S[k] = P[self.ph_intent == k].max(0)
        for k, it in enumerate(self.intents):
            for j, c in enumerate(self.chunks):
                h = c['section'].lower()
                if h and any(re.search(r'\b' + re.escape(w) + r'\b', h) for w in it['kw']):
                    S[k, j] += self.kw_bonus
        # mutual check: the chunk must also rank this intent among its top-m intents (stops a generic intent such as
        # "prerequisites" landing on whichever chunk is least bad)
        # answer cue: the passage body must contain the kind of fact that answers the intent (a time for "when",
        # "record"/"video" for recordings); otherwise the nearest topical passage gets shown although it says nothing
        if self.cue:
            for k, it in enumerate(self.intents):
                rx = re.compile(it.get('cue') or '.', re.I)
                for j, c in enumerate(self.chunks):
                    body = c['text'][len(c['section']) + 2:] if c['section'] else c['text']
                    if len(body.split()) < self.min_words or not rx.search(body): S[k, j] = -1.0
        R = (S[None, :, :] > S[:, None, :]).sum(1)  # R[k, j] = number of intents scoring higher than k on chunk j
        faq = []
        for k in range(K):
            j = int(S[k].argmax()); s = float(S[k, j])
            if s >= self.t_faq and (not self.mutual or R[k, j] < self.mutual):
                faq.append({'q': self.intents[k]['q'], 'a': self.chunks[j]['text'], 'section': self.chunks[j]['section'], 'conf': round(s, 3), 'intent': self.intents[k]['id'], 'chunk': j})
        faq.sort(key=lambda f: (-f['conf'], f['intent']))
        return faq[:max_faq]

    def ask(self, q):
        if self.emb is None: raise ValueError('no syllabus loaded')
        nq = ' '.join(q.lower().split())
        for f in self.faq:  # the student clicked (or typed) a FAQ question: answer with the passage the FAQ shows
            if ' '.join(f['q'].lower().split()) == nq:
                return {'stamp': 'SOURCED', 'answer': f['a'], 'section': f['section'], 'conf': f['conf'], 'reason': 'faq'}
        e = self.m.encode([q], normalize_embeddings=True)[0]
        s = self.emb @ e; i = int(s.argmax()); conf = float(s[i])
        if conf < self.t:
            return {'stamp': 'ESCALATED', 'answer': 'Please ask the TA', 'conf': round(conf, 3), 'reason': 'low confidence'}
        c = self.chunks[i]
        return {'stamp': 'SOURCED', 'answer': c['text'], 'section': c['section'], 'conf': round(conf, 3)}
