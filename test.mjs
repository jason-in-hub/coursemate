// Headless test for index.html: run the embedded model + calculator without a browser. Run: node test.mjs
// Accuracy on the labelled questions lives in ml/ (eval scripts + parity.py); this file guards the page itself.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
const html = readFileSync(fileURLToPath(new URL("./index.html", import.meta.url)), "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
let fails = 0, known = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) fails++; };
// 1. whole script must parse
try { new Function(script); ok(true, "script parses"); } catch (e) { ok(false, "script parses: " + e.message); }
// 2. run only the model + core half (no DOM)
const core = script.slice(script.indexOf("// ---- MODEL"), script.indexOf("// ---- UI ----"));
const M = new Function(core + "\nreturn {answer, calc, MODEL};")();
const W = M.MODEL.weights;
ok(M.MODEL && M.MODEL.chunks.length === 37, "inline model loaded, 37 chunks (got " + M.MODEL.chunks.length + ")");
ok(W.items.reduce((a, i) => a + i.weight, 0) === 22, "project weights sum to 22");
ok(JSON.stringify(W.items.map(i => [i.key, i.weight])) === JSON.stringify([["registration", 0], ["proposal", 2], ["poster", 10], ["video", 3], ["report", 7]]),
  "weights table = Important Dates (Registration 0, Proposal 2, Poster 10, Video 3, Final Report 7)");
ok(typeof M.MODEL.conf_formula === "string" && ["v1", "margin", "ideal", "p", "meta"].includes(M.MODEL.conf_mode), "conf_mode " + M.MODEL.conf_mode + ": " + M.MODEL.conf_formula);
ok(typeof M.MODEL.threshold === "number" && M.MODEL.threshold > 0 && M.MODEL.threshold < 1, "threshold in (0,1): " + M.MODEL.threshold.toFixed(4));
ok(JSON.stringify(M.MODEL) === JSON.stringify(JSON.parse(readFileSync(fileURLToPath(new URL("./model.json", import.meta.url)), "utf8"))), "inline model == model.json");
// 3. sample buttons: never a wrong answer. ESCALATED is allowed (risk control); an answer must be the right one.
const gold = {
  "When is the proposal due?": ["SOURCED", ["page:important-dates:0", "page:submission:0", "canvas:proposal-report"]],
  "Is late submission allowed?": ["SOURCED", ["page:submission:0"]],
  "If I get 0 on the poster, what is my project mark?": ["CALCULATED"],
  "Can you write my assignment?": ["ESCALATED"],
  "When is the poster link due?": ["SOURCED", ["page:important-dates:0", "canvas:poster-presentation"]],
};
for (const [q, [want, chunks]] of Object.entries(gold)) {
  const r = M.answer(q);
  const good = want === "ESCALATED" ? r.stamp === "ESCALATED"
    : r.stamp === "ESCALATED" || (r.stamp === want && (!chunks || chunks.includes(r.top_chunk)));
  const line = JSON.stringify(q) + " => " + r.stamp + (r.reason ? " (" + r.reason + ")" : "") + " conf " + r.conf.toFixed(3) + (r.stamp === "SOURCED" ? " top " + r.top_chunk : "");
  if (good && r.stamp !== want) { known++; console.log("KNOWN MISS " + line + "  [wanted " + want + "; escalation is safe, not wrong]"); }
  else ok(good, line);
}
// 4. calculator: hand-computed (demo scores 85/90/80/80 -> 2*85+10*90+3*80+7*80 = 1870 hundredths = 18.7)
const c = M.calc("If I get 0 on the poster");
ok(c.stamp === "CALCULATED" && c.item === "poster" && c.score === 0 && c.before === "18.7" && c.after === "9.7" && c.delta === "-9",
  "poster 0: " + c.before + " -> " + c.after + " of " + c.total + " (want 18.7 -> 9.7, -9)");
ok(M.calc("what if the trailer gets full marks").after === "19.3", "trailer full marks -> 19.3");
ok(M.calc("If I get 0 in Quiz 2, what is my final percentage?").reason === "weight not in table", "quiz what-if -> ESCALATED, weight not in table");
ok(M.calc("tell me a joke").stamp === "ESCALATED", "calc escalates without a project item");
ok(M.answer("When is the midterm exam?").stamp === "ESCALATED", "midterm date (not in docs) -> ESCALATED");
console.log("sample questions answered as intended: " + (5 - known) + "/5 (" + known + " known miss: safe escalation)");
console.log(fails ? "\nRED: " + fails + " failing" : "\nGREEN: all passed");
process.exit(fails ? 1 : 0);
