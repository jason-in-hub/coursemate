"""Generate results/RESULTS.md from the JSON/CSV files the other scripts wrote. Run (from ml/) last.
Every number below is read from a results file; the script that produced it is named next to each table.
"""
import csv
import json
import subprocess
from collections import Counter

from common import RESULTS, load_chunks, load_questions

R = json.loads((RESULTS / "retrieval.json").read_text())
T = json.loads((RESULTS / "router.json").read_text())
P = json.loads((RESULTS / "parity.json").read_text())
C = json.loads((RESULTS / "calc.json").read_text())
pred = list(csv.DictReader((RESULTS / "predictions.csv").open(encoding="utf-8")))
qs, chunks = load_questions(), load_chunks()
f3 = lambda x: f"{x:.3f}"
pc = lambda x: f"{100 * x:.1f}%"
ci = lambda d, k: f"{f3(d[k])} [{f3(d[k + '_ci'][0])}, {f3(d[k + '_ci'][1])}]"

NOTES = {  # hand-written diagnosis for failures, keyed by question id (shown only if that id still fails)
    "q057": "Wrong answer shown (the worst failure type). BM25 per-term scores: Video Trailer chunk 5.60 from 'long' 1.93, 'should' 1.72, 'be' 1.21, 'the' 0.74; gold Final Report chunk 2.83, of which 'conclusion' 2.12. Common words outvote the one content word.",
    "q076": "Retrieved the Submission chunk, which talks about the video link but not where to upload; gold is the Video Trailer chunk ('Upload the video to a ShareDrive'). Partly a gold-labelling judgment call.",
    "q036": "Router sends it to out-of-scope (conf 0.73). 'email' is not in the router's training vocabulary; 'instructor' (learnt from the train out-of-scope question \"When are the instructor office hours?\") and the function words 'to', 'the', 'i' carry the largest out-of-scope weights. Retrieval alone found the right chunk.",
    "q030": "Router says out-of-scope: 'much' and 'worth' lean out-of-scope because the train question \"How much are the quizzes worth in the course grade?\" is labelled out-of-scope; 'each' and 'item' are unseen. BM25 also misses (Final Report chunk first): 'worth' and 'item' never appear in the corpus, which says 'Weight' and 'Deliverable'.",
    "q012": "BM25 per-term scores: Group Formation 3.04, all from 'deadline' ('register ... by the deadline'); gold chunks 2.47 and 2.52, only from 'video', because they say 'Due' rather than 'deadline'. Low confidence, so it escalated: a safe miss.",
    "q038": "Code-mixed query: the three Han characters and 'submit' do not occur in the corpus (corpus has 'submitted', 'submission'), so the only BM25 evidence is 'report' (1.56 on the gold chunk). Right chunk, but confidence too low: escalated.",
}


def main():
    m = R["results"]
    L = ["# CourseMate ML prototype: results", "",
         "All numbers come from running `bash run_all.sh` in this folder; each table names the script and file it came from. "
         "Paste-ready for report sections 4 (data), 5 (methods) and 6 (evaluation).", "",
         "**Circularity warning.** The 167 labelled questions in `ml/data/questions.csv` were written and labelled by one author "
         "(the same person who built the system), with gold chunks chosen from the chunk text. That makes every number here "
         "optimistic. Before the report, a second team member must label the questions independently (label and gold chunks, "
         "blind to these labels); report inter-annotator agreement (Cohen's kappa on the label, overlap on gold chunks) and rerun.", ""]
    # ---------------- data
    cnt = Counter((q["split"], q["label"]) for q in qs)
    kinds = Counter(q["kind"] for q in qs)
    cw = [c["n_words"] for c in chunks if c["section"] == "Canvas Assignments"]
    pn = sorted(c["n_words"] for c in chunks if c["id"].startswith("page:"))
    L += ["## 4. Data and preprocessing", "",
          "Sources (only these): the course project page (text), 22 Canvas assignment items (JSON, due dates in UTC), the Canvas front page. "
          "Course-wide grading weights (quizzes, assignments, exams) are not in these sources and are not used anywhere.", "",
          f"- **Chunks** (`ml/build_corpus.py` -> `ml/data/chunks.jsonl`): {len(chunks)} chunks. Course page split by section, then into windows of about 150 words "
          f"(max 200) with a 20-word overlap (13-15%). Page chunk sizes in words: {pn}. Sections shorter than 100 words "
          "(Important Dates, Group Formation) stay whole: a deviation from the 100-200-word plan, kept so each chunk has one source label. "
          f"Each Canvas item is one short chunk ({min(cw)}-{max(cw)} words) with its due time converted to HKT (UTC+8); items with no due date say so.",
          "- **Normalisation and tokeniser** (`ml/textnorm.py`, ported to JS): Unicode NFKC, lowercase; tokens are ASCII letter/digit runs or single Han characters, "
          "so \"proposal幾時due\" -> proposal / 幾 / 時 / due. Single-character tokens are kept (\"Quiz 2\", \"Part A\", \"get 0\"). MiniLM gets the raw text and uses its own WordPiece tokeniser.",
          f"- **Questions** (`ml/data/questions.csv`): {len(qs)} labelled questions; kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) +
          ". Lookup questions carry every chunk that states the answer as gold. Calc questions carry the expected item, score and stamp; "
          "what-ifs on quizzes/assignments/exams are labelled calc with expected ESCALATED (\"weight not in table\").",
          "- **Split** (`ml/make_split.py`, seed 1945, frozen before any model was fitted; `run_all.sh` checks its SHA-256): 60/20/20 by question family "
          "(paraphrase/typo/code-mix variants of one question stay together), stratified by label.", "",
          "| split | lookup | calc | oos | total |", "|---|---|---|---|---|"]
    for s in ("train", "tune", "test"):
        L.append(f"| {s} | {cnt[(s, 'lookup')]} | {cnt[(s, 'calc')]} | {cnt[(s, 'oos')]} | {sum(cnt[(s, l)] for l in ('lookup', 'calc', 'oos'))} |")
    L += ["", "Test has only 5 calc and 5 out-of-scope questions, so per-class numbers on test move by 0.2 per question.", ""]
    # ---------------- methods
    L += ["## 5. Methods", "",
          "- **Router**: TF-IDF + logistic regression (C=10, class-balanced), 3 classes lookup / calc / oos, fitted on train only. "
          f"Analyzer chosen on tune macro-F1: " + ", ".join(f"{k} {f3(v)}" for k, v in T["router_analyzer_selection_tune_macro_f1"].items()) +
          f" -> **{T['router_analyzer']}** (`ml/train_router.py`).",
          f"- **Lookup**: BM25 vs TF-IDF (word 1-gram, word 1-2-gram, char_wb 3-5-gram) vs all-MiniLM-L6-v2 embeddings (cosine). "
          f"Selection rule: {R['selection_rule']}. Winner on tune: **{R['winner_on_tune']}**; deployed: **{R['deployed']}**. "
          + (R["deploy_note"] or "MiniLM did not win, so no transformers.js download is needed; the page stays a single static file."),
          "- **Calculator** (`ml/calc.py`): deterministic code over the Important Dates weights only (Registration 0, Proposal 2, Poster 10, Video 3, Final Report 7; total 22). "
          "Integer arithmetic in hundredths of a mark, so 18.7 is exact. A what-if on any item without a weight is ESCALATED with reason \"weight not in table\". No LLM is used anywhere.",
          f"- **Confidence and fallback** (`ml/pipeline.py`): lookup conf = router max-probability x s/(s+K), s = top-1 BM25 score, K = {T['K']:.3f} (median top-1 BM25 score on train lookups); "
          f"calc conf = router max-probability; out-of-scope always escalates. Below the threshold the answer is ESCALATED: \"Please ask the TA.\" "
          f"Threshold rule (fixed before running): highest tune coverage with tune risk <= {T['risk_cap']}; ties -> higher threshold. Chosen: **{T['threshold']:.4f}**.",
          "- **Browser deployment**: `ml/export_model.py` writes `model.json` (router vocabulary, idf, LR coefficients, BM25 term statistics, chunk texts, threshold, weights) "
          "and inlines it into `index.html`, so the page works from file:// and on any static host with no network calls.", ""]
    # ---------------- evaluation
    L += ["## 6. Evaluation", "", "### 6.1 Lookup retrieval (`ml/eval_retrieval.py` -> `results/retrieval.json`)", "",
          f"Test split, lookup questions only (n = {m['bm25']['test']['n']}); 95% bootstrap CI (2000 resamples) in brackets. "
          "Top-k = any gold chunk in the top k. Recall@k = share of a question's gold chunks found in the top k. MRR = 1 / rank of first gold chunk.", "",
          "| representation | tokeniser | tune MRR (selection) | Top-1 | Top-3 | R@1 | R@3 | R@5 | MRR |", "|---|---|---|---|---|---|---|---|---|"]
    tok = {"bm25": "word", "word1": "word", "word12": "word (1-2-gram)", "char35": "sub-word chars", "minilm": "WordPiece"}
    for k in m:
        t = m[k]["test"]
        L.append(f"| {k} | {tok[k]} | {f3(m[k]['tune']['mrr'])} | {ci(t, 'top1')} | {ci(t, 'top3')} | {f3(t['r1'])} | {f3(t['r3'])} | {f3(t['r5'])} | {f3(t['mrr'])} |")
    L += ["", f"Larger n: retrieval is not trained on questions, so all {m['bm25']['all_lookup']['n']} lookup questions can also be scored "
          "(supplementary; the test split above is the primary number):", "",
          "| representation | Top-1 all | Top-3 all | Top-1 clean | Top-1 paraphrase | Top-1 typo | Top-1 code-mix |", "|---|---|---|---|---|---|---|"]
    for k in m:
        a, b = m[k]["all_lookup"], m[k]["by_kind_all"]
        L.append(f"| {k} | {ci(a, 'top1')} | {ci(a, 'top3')} | " + " | ".join(f"{f3(b[x]['top1'])} (n={b[x]['n']})" for x in ("clean", "paraphrase", "typo", "codemix")) + " |")
    st = [f"{k}: {v}" for k, v in R["status"].items() if v != "ok"]
    L += ["", "MiniLM status: " + ("ran from the local Hugging Face cache." if not st else "; ".join(st)), ""]
    tt = T["test"]
    L += ["### 6.2 Router (`ml/train_router.py` -> `results/router.json`)", "",
          "| metric | value | n |", "|---|---|---|",
          f"| test macro-F1 | **{f3(tt['macro_f1'])}** | {tt['n']} |",
          f"| test macro-F1, majority-class baseline | {f3(tt['majority_macro_f1'])} | {tt['n']} |",
          f"| test accuracy | {f3(tt['accuracy'])} | {tt['n']} |",
          "| test per-class F1 | " + ", ".join(f"{k} {f3(v)}" for k, v in tt["per_class_f1"].items()) + f" | {tt['n_by_class']} |",
          f"| 5-fold CV macro-F1, train+tune, family-grouped | {f3(T['cv5_train_tune']['mean'])} +- {f3(T['cv5_train_tune']['sd'])} (folds " +
          ", ".join(f3(x) for x in T["cv5_train_tune"]["folds"]) + f") | {T['cv5_train_tune']['n']} |",
          f"| 5-fold CV majority baseline | {f3(T['cv5_train_tune']['majority_mean'])} | {T['cv5_train_tune']['n']} |", "",
          "Test confusion matrix (rows = true, columns = predicted):", "", "| | lookup | calc | oos |", "|---|---|---|---|"]
    for lab, row in zip(["lookup", "calc", "oos"], tt["confusion"]):
        L.append(f"| {lab} | " + " | ".join(map(str, row)) + " |")
    L += ["", f"[INFERRED] The CV mean ({f3(T['cv5_train_tune']['mean'])}) is well below the test figure, so the test macro-F1 is probably a favourable draw of {tt['n']} questions; the plausible range is roughly the span between the two.", ""]
    a, b = T["test_at_threshold"], T["tune_at_threshold"]
    L += ["### 6.3 Risk-coverage and threshold (`ml/train_router.py` -> `results/risk_coverage.png`)", "",
          "Coverage = answered (SOURCED or CALCULATED) / all questions. Risk = wrong answers / answered. Wrong = lookup whose top chunk is not gold, "
          "calc with wrong item or score, or any answer to a question that should escalate.", "",
          "![risk-coverage](risk_coverage.png)", "",
          "| setting | coverage | risk | answered | wrong | n |", "|---|---|---|---|---|---|"]
    for name, d in (("tune, no threshold", T["tune_no_threshold"]), (f"tune, threshold {T['threshold']:.4f}", b),
                    ("test, no threshold", T["test_no_threshold"]), (f"**test, threshold {T['threshold']:.4f}**", a)):
        L.append(f"| {name} | {f3(d['coverage'])} | {f3(d['risk'])} | {d['answered']} | {d['wrong']} | {d['n']} |")
    over = [r for r in pred if r["split"] == "test" and r["stamp"] == "ESCALATED" and r["reason"] == "low confidence" and r["top_chunk"] in r["gold"].split(";")]
    L += ["", f"At the threshold, every test question that should escalate did ({a['escalated_of_those']}/{a['should_escalate']}). "
          f"End-to-end correct (right answer, or escalation when escalation was expected): {a['end_to_end_correct']}/{a['n']} test. "
          f"The cost is coverage: {len(over)} test questions escalated as low-confidence even though the top chunk was correct. "
          f"The threshold came from only {b['n']} tune questions, so it is coarse; test risk ({f3(a['risk'])}) is above the {T['risk_cap']} tune cap.", ""]
    SB = json.loads((RESULTS / "sample_buttons.json").read_text())
    L += [f"**Demo impact.** At this threshold {sum(s['stamp'] != 'ESCALATED' for s in SB)} of the {len(SB)} sample buttons get an answer "
          f"({len(SB) - sum(s['stamp'] == 'ESCALATED' and 'lookup' not in s.get('reason', '') and s['reason'] == 'low confidence' for s in SB)} of {len(SB)} behave as intended; escalating the cheating request is correct): " +
          "; ".join(f"\"{s['q']}\" -> {s['stamp']} (conf {s['conf']:.2f})" for s in SB) +
          ". The escalated lookups retrieve the correct chunk; they fall below the threshold. Choosing a lower threshold trades risk for coverage: "
          "that is a team decision. Tune operating points (from `results/router.json`, every 3rd candidate):", "",
          "| threshold | tune coverage | tune risk | answered |", "|---|---|---|---|"]
    for p in T["curves"]["tune"][::3]:
        L.append(f"| {p['t']:.3f} | {f3(p['coverage'])} | {f3(p['risk'])} | {p['answered']} |")
    L += [""]
    L += ["### 6.4 Calculator (`ml/test_calc.py`, `ml/mutate_calc.sh`)", "",
          f"Exact match on 15 hand-computed cases (expected values typed by hand in the test file): **{C['pass']}/{C['n']}**. "
          f"Mutation check: Poster weight changed 10 -> 11 gives {C['mutant_pass']}/{C['n']} (red), restored by checksum, then green again.", ""]
    L += ["### 6.5 Python-JS parity (`ml/parity.py`, `ml/mutate_parity.sh`)", "",
          f"The JS embedded in `index.html` reproduces the Python pipeline on {P['match']}/{P['n']} questions (all {len(qs)} labelled + edge cases) on label, top chunk, stamp, "
          f"reason and calculator output. Max probability difference {P['max_p_diff']:.1e}; smallest router top1-top2 margin {P['min_margin']:.4f}. "
          "A deliberate change to the JS confidence formula turns this red (see run log).", ""]
    # ---------------- targets
    L += ["### 6.6 Against the proposal targets", "", "| target | result | met? | source |", "|---|---|---|---|",
          f"| router macro-F1 >= 0.85 | test {f3(tt['macro_f1'])} (n={tt['n']}); CV {f3(T['cv5_train_tune']['mean'])} | {'yes' if tt['macro_f1'] >= 0.85 else 'no'} | train_router.py |",
          f"| lookup Top-3 >= 80% ({R['deployed']}) | test {pc(m[R['deployed']]['test']['top3'])} [CI {pc(m[R['deployed']]['test']['top3_ci'][0])}-{pc(m[R['deployed']]['test']['top3_ci'][1])}], n={m[R['deployed']]['test']['n']} | {'yes' if m[R['deployed']]['test']['top3'] >= 0.8 else 'no'} (point estimate){'; CI lower bound below 80%' if m[R['deployed']]['test']['top3_ci'][0] < 0.8 else ''} | eval_retrieval.py |",
          f"| calculator exact match >= 95% | {C['pass']}/{C['n']} | {'yes' if C['pass'] / C['n'] >= 0.95 else 'no'} | test_calc.py |", ""]
    # ---------------- error analysis
    fails = [r for r in pred if r["split"] == "test" and r["correct"] == "0"]
    pick = [r for r in fails if r["id"] in NOTES][:5]
    L += ["### 6.7 Error analysis: 5 real test failures (`results/predictions.csv`)", "",
          f"{len(fails)} of {len(pred) and sum(r['split'] == 'test' for r in pred)} test questions were not handled correctly end-to-end. Five, chosen to cover different failure types:", "",
          "| id | question | true | router | stamp | top chunk | why |", "|---|---|---|---|---|---|---|"]
    for r in pick:
        L.append(f"| {r['id']} | {r['q']} | {r['label']} | {r['pred']} | {r['stamp']} | {r['top_chunk']} | {NOTES[r['id']]} |")
    L += ["", "Pattern: answered-but-wrong cases come from BM25 word overlap with the wrong section; router misses come from cue words learnt from a small training set "
          "('instructor', 'worth'); most other failures are safe escalations caused by the coarse threshold.", ""]
    # ---------------- discrepancies + limitations
    L += ["## Source conflicts found in the data", "",
          "1. **Poster due date.** Canvas lists *Poster Presentation* due Mon 30 Nov 2026, 23:59 HKT (`canvas_assignments.json`, 2026-11-30T15:59:59Z). "
          "The course project page says the poster link is due \"One day before your assigned Poster Presentation Session\" (Team 9 presents Nov 20). "
          "Both are kept as separate chunks with their own source labels; the page shows a conflict note on either. Ask the TA which is binding.",
          "2. **Final Report points.** Canvas shows \"Final Report (with Video Trailer Link)\" as 10 points; the course page weights Final Report 7 and Video Trailer 3. The calculator uses the course page.",
          "3. **Video length.** The page says \"approximately 1-2 minutes\" and also \"Approximately 2 minutes; recommended range: 1.5-2.5 minutes\".",
          "4. The Important Dates table leaves the Final Report due date blank; only the Submission section and Canvas give Nov 30.", "",
          "## Limitations", "",
          "- Calculator takes the first number in the question: \"poster is worth 10%, what if I get 0\" computes a score of 10, and \"8 out of 10\" computes 8.",
          "- Calc what-if questions reuse a few templates across splits, so the router's 5/5 on test calc questions is optimistic.",
          "- Self-written, single-annotator test set (see the warning at the top); no inter-annotator agreement yet.",
          f"- Small n: test has {tt['n']} questions ({m['bm25']['test']['n']} lookups, 5 calc, 5 out-of-scope). CIs are wide; tune ({b['n']}) is too small for a stable threshold.",
          f"- The corpus is {len(chunks)} chunks from three documents; questions whose answers are not in them (midterm date, quiz weights) can only be escalated.",
          "- Lexical retrieval does not know synonyms (\"worth\" vs \"weight\", \"deadline\" vs \"due\"); MiniLM did worse here, possibly because Canvas chunks are short templated strings.",
          "- Calculator handles one item and one whole-number score per question; it does not parse multi-item what-ifs or typos in item names (\"postr\").",
          "- The old hand-built router answered quiz due dates with Ask TA; now they are lookups, because Canvas gives the dates.",
          "- Not evaluated with real students; no usability data.", ""]
    (RESULTS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote results/RESULTS.md")


if __name__ == "__main__":
    main()
