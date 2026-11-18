// E2E for engine mode: real server (ml/serve.py) + real browser. Uploads a CS50 syllabus as HTML and as PDF.
// Run: start the server (ml/.venv/bin/python ml/serve.py), then node e2e_engine.cjs <syllabus.html>
// Static mode (no serve.py; the engine runs in the page via engine.js): STATIC=1 node e2e_engine.cjs <syllabus.html>
//   starts python3 -m http.server itself, runs the same checks, and prints load and per-question timings.
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs'), { spawn } = require('child_process');
const STATIC = process.env.STATIC === '1', PORT = +(process.env.PORT || 8798);
const BASE = STATIC ? 'http://127.0.0.1:' + PORT + '/index.html' : 'http://127.0.0.1:8765/index.html', SYL = process.argv[2];
const srv = STATIC ? spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '--directory', __dirname], { stdio: 'ignore' }) : null;
const ms = t => Math.round(performance.now() - t), times = [];
let fails = 0; const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
const last = p => p.locator('.receipt').last();
async function ask(p, q) { const n = await p.locator('.receipt').count(); await p.fill('#in', q); const t = performance.now(); await p.press('#in', 'Enter'); await p.locator('.receipt').nth(n).waitFor(); times.push(ms(t)); return { stamp: (await last(p).locator('.stamp').innerText()).trim(), text: await last(p).innerText() }; }
(async () => {
  if (STATIC) await new Promise(r => setTimeout(r, 800));
  // GPU=1 asks headless Chromium for WebGPU; the page still falls back to WASM when no adapter is found
  const b = await chromium.launch(process.env.GPU === '1' ? { args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'] } : {}); const p = await b.newPage({ viewport: { width: 1440, height: 900 } }); const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  // make a PDF of the same syllabus with the browser itself
  const pdfPath = path.join(process.env.SHOTS || require('os').tmpdir(), 'cs50_syllabus.pdf'); // not the tracked copy: p.pdf() bytes differ per run
  await p.goto('file://' + path.resolve(SYL)); await p.pdf({ path: pdfPath });
  for (const file of [SYL, pdfPath]) {
    let t = performance.now();
    await p.goto(BASE); await p.waitForFunction(() => !document.getElementById('syl').disabled, null, { timeout: 120000 });
    if (STATIC) {
      const load = ms(t), eng = await p.evaluate(async () => ({ backend: document.body.dataset.engine, engineMs: document.body.dataset.engineMs, bankMs: document.body.dataset.bankMs, gpu: !!navigator.gpu, health: await fetch('/api/health').then(r => r.status) }));
      console.log('TIME ' + path.extname(file) + ': page load to upload enabled ' + load + ' ms (engine init ' + eng.engineMs + ' ms, of which embedding the question bank ' + eng.bankMs + ' ms, ' + (file === SYL ? 'cold cache' : 'warm cache') + '), backend ' + eng.backend + ', navigator.gpu ' + eng.gpu);
      ok(eng.health === 404 && !!eng.backend, path.extname(file) + ': no serve.py (/api/health ' + eng.health + '), in-browser engine loaded (' + eng.backend + ')');
    }
    await p.setInputFiles('#syl', file); t = performance.now();
    await p.waitForFunction(() => /FAQ from/.test(document.getElementById('engsub').textContent) || /Could not/.test(document.getElementById('engsub').textContent), null, { timeout: 60000 });
    if (STATIC) console.log('TIME ' + path.extname(file) + ': upload to FAQ shown ' + ms(t) + ' ms');
    const sub = await p.locator('#engsub').innerText(), nfaq = await p.locator('.fq').count(), ext = path.extname(file);
    ok(/FAQ from \d+ passages/.test(sub) && nfaq > 0, ext + ': upload builds FAQ (' + sub + ')');
    ok((await p.locator('.course').innerText()).trim() !== 'COMP1945', ext + ': header switches to the uploaded course');
    ok(!(await p.locator('#hint').isVisible()), ext + ': COMP1945 sample questions hidden in engine mode');
    ok(/MiniLM/.test(await p.locator('#mdl').innerText()) && /MiniLM/.test(await p.locator('#trace').innerText()), ext + ': model and trace panels describe the engine, not the COMP1945 model');
    if (STATIC) ok(/in your browser/.test(await p.locator('#mdl').innerText()), ext + ': model panel says the engine runs in the browser');
    console.log('   FAQ: ' + (await p.locator('.fq summary').allInnerTexts()).slice(0, 6).join(' | '));
    if (ext === '.pdf') await p.screenshot({ path: path.join(process.env.SHOTS || __dirname, 'e2e-engine-mode.png') });
    await p.locator('.fq').first().locator('summary').click(); await p.locator('.fq').first().locator('button').click();
    await p.locator('.receipt').first().waitFor();
    ok((await last(p).locator('.stamp').innerText()).trim() === 'SOURCED', ext + ': "Ask this" on a FAQ item -> SOURCED');
    let r = await ask(p, 'What is the capital of France?');
    ok(r.stamp === 'ESCALATED', ext + ': off-topic question -> ESCALATED (' + r.text.match(/confidence\s+[\d.]+/)?.[0] + ')');
    r = await ask(p, 'Can I collaborate with classmates on the final project?');
    ok(r.stamp === 'SOURCED' && /Final Project/.test(r.text), ext + ': final-project collaboration question -> SOURCED from the Final Project section');
    // paraphrase the engine is less sure about: escalating is safe, answering from the wrong section is a failure
    r = await ask(p, 'Can I work with a partner on the final project?');
    if (r.stamp === 'ESCALATED') console.log('KNOWN MISS ' + ext + ': "work with a partner" escalates (safe, not wrong)');
    else ok(/Final Project/.test(r.text), ext + ': "work with a partner" answered from the Final Project section');
    await p.click('#back');
    r = await ask(p, 'When is the proposal due?');
    ok(r.stamp === 'SOURCED' && /24 Oct 2026/.test(r.text), ext + ': back to COMP1945, static model answers again');
  }
  // file:// mode: no server, upload disabled with a hint
  await p.goto('file://' + path.resolve(__dirname, 'index.html'));
  ok(!(await p.locator('#back').isVisible()), 'COMP1945 mode: "Back to COMP1945" button hidden');
  ok(await p.locator('#syl').isDisabled() && /serve\.py/.test(await p.locator('#engsub').innerText()), 'file:// without server: upload disabled, hint shown');
  ok(errs.length === 0, 'no page errors ' + JSON.stringify(errs));
  await p.screenshot({ path: path.join(process.env.SHOTS || __dirname, 'e2e-engine.png') });
  if (STATIC) { const s = [...times].sort((x, y) => x - y); console.log('TIME per question (Enter to receipt, n=' + s.length + '): median ' + s[s.length >> 1] + ' ms, max ' + s[s.length - 1] + ' ms'); srv.kill(); }
  await b.close(); console.log(fails ? 'RED: ' + fails + ' failing' : 'GREEN: all passed'); process.exit(fails ? 1 : 0);
})();
