// E2E for engine mode: real server (ml/serve.py) + real browser. Uploads a CS50 syllabus as HTML and as PDF.
// Run: start the server (ml/.venv/bin/python ml/serve.py), then node e2e_engine.cjs <syllabus.html>
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
const BASE = 'http://127.0.0.1:8765/index.html', SYL = process.argv[2];
let fails = 0; const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
const last = p => p.locator('.receipt').last();
async function ask(p, q) { const n = await p.locator('.receipt').count(); await p.fill('#in', q); await p.press('#in', 'Enter'); await p.locator('.receipt').nth(n).waitFor(); return { stamp: (await last(p).locator('.stamp').innerText()).trim(), text: await last(p).innerText() }; }
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1440, height: 900 } }); const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  // make a PDF of the same syllabus with the browser itself
  const pdfPath = path.join(path.dirname(SYL), 'cs50_syllabus.pdf');
  await p.goto('file://' + path.resolve(SYL)); await p.pdf({ path: pdfPath });
  for (const file of [SYL, pdfPath]) {
    await p.goto(BASE); await p.waitForFunction(() => !document.getElementById('syl').disabled);
    await p.setInputFiles('#syl', file);
    await p.waitForFunction(() => /FAQ from/.test(document.getElementById('engsub').textContent) || /Could not/.test(document.getElementById('engsub').textContent), null, { timeout: 60000 });
    const sub = await p.locator('#engsub').innerText(), nfaq = await p.locator('.fq').count(), ext = path.extname(file);
    ok(/FAQ from \d+ passages/.test(sub) && nfaq > 0, ext + ': upload builds FAQ (' + sub + ')');
    ok((await p.locator('.course').innerText()).trim() !== 'COMP1945', ext + ': header switches to the uploaded course');
    ok(!(await p.locator('#hint').isVisible()), ext + ': COMP1945 sample questions hidden in engine mode');
    ok(/MiniLM/.test(await p.locator('#mdl').innerText()) && /MiniLM/.test(await p.locator('#trace').innerText()), ext + ': model and trace panels describe the engine, not the COMP1945 model');
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
  await b.close(); console.log(fails ? 'RED: ' + fails + ' failing' : 'GREEN: all passed'); process.exit(fails ? 1 : 0);
})();
