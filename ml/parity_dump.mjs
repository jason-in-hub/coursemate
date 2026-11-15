// Runs the inference code embedded in ../index.html (or the HTML path given as argv[2]; not a copy) on questions read from stdin (JSON list),
// prints one JSON result per question. Used by parity.py.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
const html = readFileSync(process.argv[2] || fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const core = script.slice(script.indexOf("// ---- MODEL"), script.indexOf("// ---- UI ----"));
const M = new Function(core + "\nreturn {answer, calc, MODEL};")();
const qs = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(qs.map(q => M.answer(q))));
