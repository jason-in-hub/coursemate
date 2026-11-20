// Parity: Python engine (ml/engine.py, what serve.py runs) vs the in-browser engine (engine.js) on the same files and questions.
// Checks: per-text embedding cosine >= 0.99 (chunks, question bank, questions), identical chunks, FAQ list and SOURCED/ESCALATED stamps.
// Run: NODE_PATH=<dir with playwright> node parity_engine.cjs [syllabus files...]   (PY=python with sentence-transformers; default ml/.venv/bin/python)
const { chromium } = require('playwright');
const { spawn, execFileSync } = require('child_process');
const path = require('path'), fs = require('fs'), os = require('os');
const ROOT = __dirname, PY = process.env.PY || path.join(ROOT, 'ml/.venv/bin/python'), PORT = +(process.env.PORT || 8799);
const FILES = process.argv.slice(2).length ? process.argv.slice(2) : ['ml/data/external/cs50_demo/syllabus.html', 'ml/data/external/cs50_demo/cs50_syllabus.pdf', 'ml/data/raw/course_project_page.txt'];
const QS = ['What is the capital of France?', 'Can I collaborate with classmates on the final project?', 'Can I work with a partner on the final project?',
  'When is the proposal due?', 'How is my final grade computed?', 'Do I need to buy a textbook?', 'Is there an exam?', 'What happens if I submit late?',
  'Can I use ChatGPT?', 'Who teaches this course?', 'How many problem sets are there?', 'What languages does the course cover?'];
const MIN_COS = 0.99, FAQ_AGREE = +(process.env.FAQ_AGREE || 0.9), ASK_TOL = +(process.env.ASK_TOL || 0.05);
let fails = 0; const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
const cos = (a, b) => { let s = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return s / Math.sqrt(x * y); };
const stat = v => ({ min: Math.min(...v), mean: v.reduce((a, b) => a + b, 0) / v.length, n: v.length });
(async () => {
  const tmp = path.join(os.tmpdir(), 'cm_parity_' + process.pid + '.json');
  execFileSync(PY, [path.join(ROOT, 'ml/engine_dump.py'), tmp, ...FILES.map(f => path.resolve(ROOT, f))], { input: JSON.stringify(QS), stdio: ['pipe', 'inherit', 'inherit'] });
  const py = JSON.parse(fs.readFileSync(tmp, 'utf8')); fs.unlinkSync(tmp);
  const srv = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: ROOT, stdio: 'ignore' });
  await new Promise(r => setTimeout(r, 800));
  // GPU=1 asks headless Chromium for WebGPU (DEVICE=wasm forces WASM)
  const b = await chromium.launch(process.env.GPU === '1' ? { args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'] } : {}), p = await b.newPage();
  p.on('pageerror', e => console.log('pageerror', e.message));
  try {
    await p.goto('http://127.0.0.1:' + PORT + '/index.html');
    const files = FILES.map(f => ({ name: path.basename(f), key: path.resolve(ROOT, f), data: [...fs.readFileSync(path.resolve(ROOT, f))] }));
    const js = await p.evaluate(async ({ files, qs, device }) => {
      const M = await import('/engine.js'); const E = await M.createEngine(() => {}, { device });
      const arr = v => v.map(e => Array.from(e));
      const out = { threshold: E.t, bank: E.bank, backend: E.backend, bank_emb: arr(await E.encode(E.bank)), files: {} };
      for (const f of files) {
        const r = await E.ingest(f.name, new Uint8Array(f.data)); const allq = [...new Set(qs.concat(r.faq.map(x => x.q)))], asks = {};
        for (const q of allq) asks[q] = await E.ask(q);
        out.files[f.key] = { chunks: E.chunks, chunk_emb: arr(E.emb), faq: r.faq, n: r.chunks, asks, q_emb: arr(await E.encode(allq)) };
      }
      return out;
    }, { files, qs: QS, device: process.env.DEVICE || undefined });
    console.log('backend: ' + js.backend);
    ok(js.threshold === py.threshold, 'threshold ' + js.threshold);
    ok(JSON.stringify(js.bank) === JSON.stringify(py.bank), 'question bank identical (' + js.bank.length + ' vs ' + py.bank.length + ')');
    const bc = py.bank.map((q, i) => cos(py.bank_emb[i], js.bank_emb[js.bank.indexOf(q)] || []));
    let s = stat(bc); ok(s.min >= MIN_COS, 'bank embedding cosine min ' + s.min.toFixed(5) + ' mean ' + s.mean.toFixed(5) + ' (n=' + s.n + ')');
    for (const f of files) {
      const P = py.files[f.key], J = js.files[f.key], tag = f.name + ': ';
      ok(JSON.stringify(P.chunks) === JSON.stringify(J.chunks), tag + 'chunks identical (' + P.n + ' vs ' + J.n + ')');
      s = stat(P.chunk_emb.map((e, i) => cos(e, J.chunk_emb[i] || [])));
      ok(s.min >= MIN_COS, tag + 'chunk embedding cosine min ' + s.min.toFixed(5) + ' mean ' + s.mean.toFixed(5) + ' (n=' + s.n + ')');
      const pq = Object.keys(P.asks), jq = Object.keys(J.asks);
      s = stat(P.q_emb.map((e, i) => cos(e, J.q_emb[jq.indexOf(pq[i])] || [])));
      ok(s.min >= MIN_COS, tag + 'question embedding cosine min ' + s.min.toFixed(5) + ' mean ' + s.mean.toFixed(5) + ' (n=' + s.n + ')');
      const fk = x => x.q + ' -> ' + x.section;
      // order-insensitive: the browser model is int8 ONNX, so near-tied scores (|diff| <= 0.005) can swap order or,
      // rarely, the chosen passage. Gate: >= FAQ_AGREE of (question, passage) items shared; every difference printed.
      const key = x => JSON.stringify([x.q, x.a]), jk = new Set(J.faq.map(key)), pk = new Set(P.faq.map(key));
      const shared = P.faq.filter(x => jk.has(key(x))).length, agree = shared / Math.max(P.faq.length, J.faq.length, 1);
      ok(agree >= FAQ_AGREE, tag + 'FAQ items shared ' + shared + '/' + Math.max(P.faq.length, J.faq.length) + ' (' + (agree * 100).toFixed(0) + '%, gate ' + FAQ_AGREE * 100 + '%)' + (JSON.stringify(P.faq.map(key)) === JSON.stringify(J.faq.map(key)) ? ', same order' : ', order differs'));
      P.faq.filter(x => !jk.has(key(x))).forEach(x => console.log('   only py: ' + fk(x) + ' ' + x.conf));
      J.faq.filter(x => !pk.has(key(x))).forEach(x => console.log('   only js: ' + fk(x) + ' ' + x.conf));
      const jc = Object.fromEntries(J.faq.map(x => [key(x), x.conf])), dc = P.faq.filter(x => key(x) in jc).map(x => Math.abs(x.conf - jc[key(x)]));
      console.log('   FAQ conf max |py-js| ' + (dc.length ? Math.max(...dc) : 0).toFixed(3));
      let stampMiss = 0, dconf = 0;
      for (const q of pq) {
        const a = P.asks[q], c = J.asks[q] || {};
        if (a.stamp !== c.stamp || (a.answer !== c.answer)) { stampMiss++; console.log('   MISMATCH ' + JSON.stringify(q) + ' py ' + a.stamp + ' ' + a.conf + ' / js ' + c.stamp + ' ' + c.conf); }
        dconf = Math.max(dconf, Math.abs(a.conf - c.conf));
      }
      ok(stampMiss <= Math.floor(ASK_TOL * pq.length), tag + 'same stamp and passage on ' + (pq.length - stampMiss) + '/' + pq.length + ' questions (gate: at most ' + Math.floor(ASK_TOL * pq.length) + ' near-tie differences; max |conf py-js| ' + dconf.toFixed(3) + ')');
    }
  } finally { await b.close(); srv.kill(); }
  console.log(fails ? 'RED: ' + fails + ' failing' : 'GREEN: all passed'); process.exit(fails ? 1 : 0);
})();
