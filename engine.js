// CourseMate engine in the browser: a port of ml/engine.py (+ the text extraction ml/serve.py does), no server, no LLM.
// Same steps as Python: read_text -> chunk -> embed with the fine-tuned MiniLM (ONNX, int8 weights) -> cosine -> threshold.
// Libraries come from jsDelivr at pinned versions: transformers.js (tokenizer + onnxruntime-web) and pdf.js (PDF only, loaded on demand).
// Model files load from ml/engine_model/onnx/ next to this file, so any static server works (python3 -m http.server).
const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const PDFJS = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@5.7.284/build/';
const BASE = new URL('ml/engine_model/', import.meta.url).href;
const MAX_BYTES = 20 * 1024 * 1024, MAX_TOKENS = 256, BATCH = 16;
const MAX_WORDS = 120, MUTUAL = 3, MIN_WORDS = 12;  // engine.Engine defaults (chosen on DEV syllabi, ml/eval_faq.py)

// Python str semantics the chunker relies on: str.split()/strip() whitespace and str.splitlines() separators
const WS = '\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000';
const STRIP = new RegExp('^[' + WS + ']+|[' + WS + ']+$', 'g'), SPLIT = new RegExp('[' + WS + ']+');
const strip = s => s.replace(STRIP, '');
const words = s => { s = strip(s); return s ? s.split(SPLIT) : []; };
const splitlines = s => s.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/);
const isUpper = ch => !!ch && ch !== ch.toLowerCase() && ch === ch.toUpperCase();

const HMARK = '\u{241E}';  // marks a heading line that came from an HTML heading tag (engine.HMARK)
const HEADS = 'h1,h2,h3,h4,h5,h6,dt,summary,caption';
const INLINE = 'a,b,strong,em,i,u,code,span,small,sup,sub,abbr,mark,kbd,var,q,cite,time,font';
const textJoin = (node, sep) => { const w = node.ownerDocument.createTreeWalker(node, NodeFilter.SHOW_TEXT), out = []; while (w.nextNode()) out.push(w.currentNode.data); return out.join(sep); };

/** Bytes of a .pdf/.html/.htm/.md/.txt file -> plain text with line breaks kept (engine.read_text). */
export async function readText(name, data) {
  const n = name.toLowerCase();
  if (n.endsWith('.pdf')) {
    const pdfjs = await import(PDFJS + 'pdf.min.mjs');
    // cross-origin worker scripts are blocked, so start the CDN worker from a same-origin blob
    if (!pdfjs.GlobalWorkerOptions.workerPort) pdfjs.GlobalWorkerOptions.workerPort = new Worker(URL.createObjectURL(new Blob(['import "' + PDFJS + 'pdf.worker.min.mjs";'], { type: 'text/javascript' })), { type: 'module' });
    const doc = await pdfjs.getDocument({ data: data.slice(), disableNormalization: true }).promise, pages = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const tc = await (await doc.getPage(i)).getTextContent({ disableNormalization: true });
      let s = ''; for (const it of tc.items) { if (!('str' in it)) continue; s += it.str; if (it.hasEOL) s += '\n'; }
      pages.push(s);
    }
    return pages.join('\n');  // pypdf: page texts joined by newlines
  }
  const t = new TextDecoder('utf-8', { ignoreBOM: true }).decode(data).replace(/\uFFFD/g, '');  // decode(errors='ignore')
  if (n.endsWith('.html') || n.endsWith('.htm')) {
    const d = new DOMParser().parseFromString(t, 'text/html');
    d.querySelectorAll('script,style,nav,footer').forEach(x => x.remove());
    // v2: heading tags (and a bold lead-in ending in ':') become one marked line; inline tags are unwrapped
    d.querySelectorAll(HEADS).forEach(h => { const t = words(textJoin(h, ' ')).join(' '); h.textContent = '\n' + HMARK + t + '\n'; });
    d.querySelectorAll('strong,b').forEach(b => {
      if (!b.isConnected) return;
      const t = strip(textJoin(b, ' ')), p = b.parentElement;
      if (t.endsWith(':') && words(t).length <= 8 && p && (p.tagName === 'P' || p.tagName === 'LI') && strip(p.textContent).startsWith(t)) b.textContent = '\n' + HMARK + words(t).join(' ') + '\n';
    });
    d.querySelectorAll(INLINE).forEach(x => { if (x.isConnected) x.replaceWith(...x.childNodes); });
    d.normalize();
    const w = d.createTreeWalker(d, NodeFilter.SHOW_TEXT), out = [];
    while (w.nextNode()) out.push(w.currentNode.data);
    return out.join('\n');  // BeautifulSoup get_text('\n')
  }
  return t;
}

const SENT = /(?<=[.!?])\s+(?=[A-Z0-9("\u{201C}])/u;
const isHeadLine = line => words(line).length <= 8 && !/[.,;]$/.test(line) && isUpper(String.fromCodePoint(line.codePointAt(0)));

/** v2 chunks (engine.chunk_v2): sections from marked HTML headings when present, else the heading heuristic; each section
 *  body packed sentence by sentence into windows of <= maxWords; a short last sentence starts the next window too. */
export function chunkV2(text, maxWords = 120) {
  const marked = text.includes(HMARK), secs = []; let cur = [], head = '';
  for (const raw of splitlines(text)) {
    let line = strip(raw); if (!line) continue;
    let isHead;
    if (marked) { isHead = line.startsWith(HMARK); line = strip(line.replace(/^\u{241E}+/u, '')); if (!line) continue; }
    else isHead = isHeadLine(line);
    if (isHead && cur.length) { secs.push([head, cur.join(' ')]); cur = []; }
    if (isHead) head = strip(line.replace(/:+$/, '')); else cur.push(line);
  }
  if (cur.length) secs.push([head, cur.join(' ')]);
  const out = [], nw = s => words(s).length;
  for (const [h, body] of secs) {
    const pieces = []; let win = [];
    for (const s of words(body).join(' ').split(SENT)) {
      if (!s) continue;
      if (win.length && nw([...win, s].join(' ')) > maxWords) { pieces.push(win.join(' ')); win = nw(win[win.length - 1]) < Math.floor(maxWords / 2) ? win.slice(-1) : []; }
      win.push(s);
    }
    if (win.length) pieces.push(win.join(' '));
    for (const p of pieces) {
      const w = words(p);
      for (let i = 0; i < w.length; i += maxWords) { const piece = w.slice(i, i + maxWords).join(' '); if (nw(piece) >= 5) out.push({ section: h, text: (h ? h + ': ' : '') + piece }); }
    }
  }
  return out;
}

/** Paragraph-aware chunks: short lines that look like headings start a new section; long sections are windowed (engine.chunk). */
export function chunk(text, maxWords = 120, overlap = 20) {
  const secs = []; let cur = [], head = '';
  for (const raw of splitlines(text)) {
    const line = strip(raw); if (!line) continue;
    const isHead = words(line).length <= 8 && !/[.,;]$/.test(line) && isUpper(String.fromCodePoint(line.codePointAt(0)));
    if (isHead && cur.length) { secs.push([head, cur.join(' ')]); cur = []; }
    if (isHead) head = line; else cur.push(line);
  }
  if (cur.length) secs.push([head, cur.join(' ')]);
  const out = [];
  for (const [h, body] of secs) {
    const w = words(body);
    for (let i = 0; i < Math.max(w.length - overlap, 1); i += maxWords - overlap) {
      const piece = w.slice(i, i + maxWords).join(' ');
      if (words(piece).length >= 5) out.push({ section: h, text: (h ? h + ': ' : '') + piece });
    }
  }
  return out;
}

const round3 = x => Math.round(x * 1000) / 1000;
const dot = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
const argmax = (rows, v) => { let i = 0, s = -Infinity; rows.forEach((r, j) => { const x = dot(r, v); if (x > s) { s = x; i = j; } }); return [i, s]; };

async function webgpuOK() {
  try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; }
}

/** Load model, tokenizer, threshold and question bank; embed the bank. onStatus(text) reports progress.
 *  opts.device: 'webgpu' | 'wasm' forces a backend (default: WebGPU when an adapter exists, else WASM). */
export async function createEngine(onStatus = () => {}, opts = {}) {
  const t0 = performance.now();
  const T = await import(TJS);
  T.env.allowRemoteModels = false; T.env.allowLocalModels = true; T.env.localModelPath = BASE;
  const cfg = await (await fetch(BASE + 'engine.json')).json();
  const tok = await T.AutoTokenizer.from_pretrained('onnx');
  let model = null, backend = 'wasm';
  if (opts.device !== 'wasm' && await webgpuOK()) {
    onStatus('Loading the model (WebGPU)');
    try { model = await T.AutoModel.from_pretrained('onnx', { device: 'webgpu', dtype: 'q8' }); backend = 'webgpu'; } catch (e) { console.warn('WebGPU failed, using WASM', e); }
  }
  if (!model) { onStatus('Loading the model (WASM)'); model = await T.AutoModel.from_pretrained('onnx', { device: 'wasm', dtype: 'q8' }); }
  async function encode(texts) {  // mean pooling over tokens, then L2 normalise (sentence-transformers Pooling + Normalize)
    const out = [];
    for (let i = 0; i < texts.length; i += BATCH) {
      const x = tok(texts.slice(i, i + BATCH), { padding: true, truncation: true, max_length: MAX_TOKENS });
      const h = (await model(x)).last_hidden_state, [B, L, D] = h.dims, hd = h.data, m = x.attention_mask.data;
      for (let b = 0; b < B; b++) {
        const e = new Float32Array(D); let n = 0;
        for (let l = 0; l < L; l++) if (Number(m[b * L + l])) { n++; const o = (b * L + l) * D; for (let d = 0; d < D; d++) e[d] += hd[o + d]; }
        let z = 0; for (let d = 0; d < D; d++) { e[d] /= n; z += e[d] * e[d]; }
        z = Math.max(Math.sqrt(z), 1e-12); for (let d = 0; d < D; d++) e[d] /= z;
        out.push(e);
      }
    }
    return out;
  }
  // drop questions tied to one course (codes like "CS 289", "Data 8", "COMP1945", or named tools)
  const specific = /\b[A-Z]{2,5} ?\d{2,4}\b|\bData ?8\b|check50|submit50|edX|Ed\b|Gradescope|JupyterHub|Piazza|DataHub|Canvas|okpy|\bOH\b/i;
  const bank = cfg.question_bank.filter(q => !specific.test(q));
  // the legacy bank is only needed by the v1 FAQ path (no intents in engine.json): embed it lazily
  const t1 = performance.now(); let bankEmb = null;
  const getBankEmb = async () => bankEmb || (bankEmb = await encode(bank));
  if (!(cfg.intents || []).length) { onStatus('Embedding ' + bank.length + ' common questions'); await getBankEmb(); }
  // v2 FAQ: canonical student intents, each with phrasings and an answer cue (engine.Engine.faq_v2)
  const intents = cfg.intents || [], phIntent = [], phText = [];
  intents.forEach((it, k) => [it.q, ...it.para].forEach(p => { phIntent.push(k); phText.push(p); }));
  if (phText.length) onStatus('Embedding ' + phText.length + ' student-question phrasings');
  const phEmb = phText.length ? await encode(phText) : [];
  const E = { t: cfg.threshold, tFaq: cfg.faq_threshold ?? cfg.threshold, kwBonus: cfg.faq_kw_bonus ?? 0, intents, bank, backend, loadMs: performance.now() - t0, bankMs: performance.now() - t1, chunks: [], emb: null, faq: [], encode };
  const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\  const E = { t: cfg.threshold, bank, backend, loadMs: performance.now() - t0, bankMs: performance.now() - t1, chunks: [], emb: null, faq: [], encode };');
  E.faqV2 = (maxFaq = 25) => {
    const K = intents.length, C = E.chunks.length, S = Array.from({ length: K }, () => new Float64Array(C).fill(-Infinity));
    phEmb.forEach((pe, p) => { const k = phIntent[p]; for (let j = 0; j < C; j++) { const s = dot(pe, E.emb[j]); if (s > S[k][j]) S[k][j] = s; } });
    intents.forEach((it, k) => {
      const kws = it.kw.map(w => new RegExp('\\b' + reEsc(w) + '\\b')), cue = new RegExp(it.cue || '.', 'i');
      E.chunks.forEach((c, j) => {
        const h = c.section.toLowerCase();
        if (h && kws.some(r => r.test(h))) S[k][j] += E.kwBonus;
        const body = c.section ? c.text.slice(c.section.length + 2) : c.text;
        if (words(body).length < MIN_WORDS || !cue.test(body)) S[k][j] = -1;
      });
    });
    const faq = [];
    for (let k = 0; k < K; k++) {
      let j = 0; for (let x = 1; x < C; x++) if (S[k][x] > S[k][j]) j = x;
      const s = S[k][j]; let rank = 0; for (let q = 0; q < K; q++) if (S[q][j] > s) rank++;
      if (s >= E.tFaq && rank < MUTUAL) faq.push({ q: intents[k].q, a: E.chunks[j].text, section: E.chunks[j].section, conf: round3(s), intent: intents[k].id, chunk: j });
    }
    faq.sort((a, b) => b.conf - a.conf || (a.intent < b.intent ? -1 : a.intent > b.intent ? 1 : 0));
    return faq.slice(0, maxFaq);
  };
  E.ingest = async (name, data, maxFaq = 25) => {
    if (data.length > MAX_BYTES) throw new Error('file too large (20 MB max)');
    const chunks = chunkV2(await readText(name, data), MAX_WORDS);
    if (!chunks.length) throw new Error('no text found in ' + name);
    const emb = await encode(chunks.map(c => c.text));
    E.chunks = chunks; E.emb = emb;
    if (intents.length) { E.faq = E.faqV2(maxFaq); return { chunks: chunks.length, faq: E.faq, threshold: E.t }; }
    const top = (await getBankEmb()).map(q => argmax(emb, q)), order = top.map((_, i) => i).sort((a, b) => top[b][1] - top[a][1]);
    const faq = [], used = new Set();
    for (const i of order) {  // best-scoring bank question per chunk, above threshold
      const [j, s] = top[i];
      if (s < E.t || faq.length >= maxFaq) break;
      if (used.has(j)) continue;
      used.add(j);
      faq.push({ q: bank[i], a: chunks[j].text, section: chunks[j].section, conf: round3(s) });
    }
    E.faq = faq;
    return { chunks: chunks.length, faq, threshold: E.t };
  };
  E.ask = async q => {
    if (!E.emb) throw new Error('no syllabus loaded');
    const nq = words(q.toLowerCase()).join(' '), hit = E.faq.find(f => words(f.q.toLowerCase()).join(' ') === nq);
    if (hit) return { stamp: 'SOURCED', answer: hit.a, section: hit.section, conf: hit.conf, reason: 'faq' };  // the FAQ already shows this passage
    const [i, conf] = argmax(E.emb, (await encode([q]))[0]);
    if (conf < E.t) return { stamp: 'ESCALATED', answer: 'Please ask the TA', conf: round3(conf), reason: 'low confidence' };
    const c = E.chunks[i];
    return { stamp: 'SOURCED', answer: c.text, section: c.section, conf: round3(conf) };
  };
  return E;
}
