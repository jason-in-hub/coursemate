// E2E check of index.html in a real browser (Playwright, Chromium headless).
// Run: npm i playwright && npx playwright install chromium-headless-shell && node e2e.cjs
const { chromium } = require('playwright');
const path = require('path');
const URL = 'file://' + path.resolve(__dirname, 'index.html');
let fails = 0;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
async function ask(p, q) {
  const n = await p.locator('.receipt').count();
  await p.fill('#in', q); await p.press('#in', 'Enter');
  await p.locator('.receipt').nth(n).waitFor();
  const r = p.locator('.receipt').nth(n);
  return { stamp: (await r.locator('.stamp').first().innerText()).trim(), text: await r.innerText() };
}
(async () => {
  const b = await chromium.launch();
  for (const [name, vw] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
    const p = await b.newPage({ viewport: vw }); const errs = [];
    p.on('pageerror', e => errs.push(e.message)); p.on('console', m => m.type() === 'error' && !/fonts\.g/.test(m.text()) && errs.push(m.text()));
    await p.goto(URL);
    ok(await p.locator('#hint button').count() === 5, name + ': 5 sample buttons');
    const model = await p.evaluate(() => ({ t: MODEL.threshold, f: MODEL.conf_formula, n: MODEL.chunks.length }));
    // typed questions (Enter key)
    let r = await ask(p, 'When is the proposal due?');
    ok(r.stamp === 'SOURCED' && /24 Oct 2026/.test(r.text), name + ': proposal due -> SOURCED with 24 Oct 2026');
    ok((await p.locator('.side').innerText()).includes(model.f), name + ': trace shows deployed conf formula');
    r = await ask(p, 'How much are the quizzes worth in the course grade?');
    ok(r.stamp === 'SOURCED' && /Syllabus/.test(r.text), name + ': quiz weight -> SOURCED from syllabus chunk');
    r = await ask(p, 'Can you write my assignment?');
    ok(r.stamp === 'ESCALATED' && /ask the TA/i.test(r.text), name + ': out of scope -> ESCALATED, ask the TA');
    r = await ask(p, 'When is the midterm exam?');
    ok(r.stamp === 'ESCALATED', name + ': midterm (not in docs) -> ESCALATED');
    // calculator with the scores table (click a sample button)
    await p.locator('#hint button', { hasText: 'If I get 0 on the poster' }).click();
    await p.waitForTimeout(400);
    let last = p.locator('.receipt').last();
    ok((await last.locator('.stamp').first().innerText()).trim() === 'CALCULATED', name + ': poster what-if -> CALCULATED');
    const before = await last.innerText();
    // edit a score: answer must use the new table
    await p.locator('td input').first().fill('0'); await p.locator('td input').first().dispatchEvent('input');
    await p.locator('#hint button', { hasText: 'If I get 0 on the poster' }).click(); await p.waitForTimeout(400);
    last = p.locator('.receipt').last();
    ok((await last.innerText()) !== before, name + ': editing Proposal score changes the what-if result');
    // layout
    const ov = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    ok(ov <= 0, name + ': no horizontal overflow (' + ov + 'px)');
    ok((await p.locator('#mdl').innerText()).includes('risk-coverage'), name + ': model note matches threshold source');
    ok(errs.length === 0, name + ': no page errors ' + JSON.stringify(errs));
    await p.screenshot({ path: path.join(process.env.SHOTS || __dirname, 'e2e-' + name + '.png'), fullPage: name === 'mobile' });
  }
  await b.close();
  console.log(fails ? 'RED: ' + fails + ' failing' : 'GREEN: all passed'); process.exit(fails ? 1 : 0);
})();
