# CourseMate (COMP1945, Group 9 prototype)

Web chatbot for COMP1945 project questions. Every answer has a receipt: SOURCED (a course-document passage with its source), CALCULATED (what-if project mark from deterministic code) or ESCALATED ("Please ask the TA.").

The page runs a real ML pipeline in the browser: a TF-IDF + logistic-regression router (lookup / calc / out-of-scope), BM25 retrieval over 46 chunks of the course documents (project page, Canvas assignments, Canvas front page, and the coursework/syllabus page from `COMP1945_Course_Data.csv`), and a confidence threshold chosen from a risk-coverage curve. The model is exported from Python (`ml/`) and inlined in `index.html` (also saved as `model.json`), so the page is one static file with no network calls. Demo scores are invented; edit them under the input box.

Results, with every number tied to the script that produced it: [results/RESULTS.md](results/RESULTS.md).

## Open locally

Double-click `index.html`, or `open index.html` (macOS).

## Rerun everything

```sh
bash run_all.sh          # creates ml/.venv on first run (python3; built and tested with 3.14), then runs every step below
```

One command per step (from `ml/`, after `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`):

```sh
.venv/bin/python build_corpus.py     # data/raw/* -> data/chunks.jsonl (46 chunks, source labels, HKT dates)
.venv/bin/python make_split.py       # ONE-TIME: freezes data/split.csv (refuses to overwrite); already done
.venv/bin/python eval_retrieval.py   # BM25 vs TF-IDF word/1-2gram/char vs MiniLM -> results/retrieval.json
.venv/bin/python train_router.py     # router, CV, risk-coverage, threshold -> results/router.json, risk_coverage.png
.venv/bin/python export_model.py     # -> ../model.json, inlined into ../index.html
.venv/bin/python test_calc.py        # 15 hand-computed calculator cases
./mutate_calc.sh                     # shows test_calc.py goes red when a weight changes, then restores
.venv/bin/python parity.py           # Python vs the JS inside index.html, every question
./mutate_parity.sh                   # shows parity.py goes red on a mutated temp copy of the page
.venv/bin/python make_results.py     # -> results/RESULTS.md
cd .. && node test.mjs               # headless page test (exit 0 = all pass)
```

`ml/data/questions.csv` and `ml/data/split.csv` are frozen; `run_all.sh` checks their SHA-256 (`ml/data/FROZEN.sha256`). The questions are self-written by one author: have a second team member label them independently before quoting the numbers.

MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) is loaded from the local Hugging Face cache (`HF_HUB_OFFLINE=1`); on a fresh machine, download it once with network access or the script records it as unavailable and continues.

## Host it (pick one; run from this folder)

GitHub Pages (needs `gh auth login`; this folder is not a git repo yet, so init and commit first; the repo is **public**):

```sh
git init -b main && git add index.html model.json README.md test.mjs && git commit -m "CourseMate prototype" && gh repo create coursemate --public --source=. --push && gh api -X POST repos/{owner}/coursemate/pages -f "source[branch]=main" -f "source[path]=/"
```

The site appears at `https://<your-user>.github.io/coursemate/` after a minute or two.

Netlify Drop (it may ask you to sign in or claim the site to keep it live): open https://app.netlify.com/drop and drag this folder onto the page.

Local server, for a phone on the same Wi-Fi:

```sh
npx serve .
```

## Engine mode: any course's syllabus (`ml/serve.py`)

Upload a syllabus (PDF, HTML, Markdown or text) and the page builds that course's FAQ and answers questions about it. No LLM: a MiniLM sentence encoder fine-tuned on public course FAQs ranks the syllabus passages by cosine similarity, and a confidence threshold hands unsure questions to the TA.

```sh
ml/.venv/bin/python ml/serve.py          # then open http://127.0.0.1:8765
```

Opened as a file (`index.html` from disk), the page still runs the COMP1945 model in the browser and shows how to start the engine.

How it was built (all from `ml/`):

| step | script | output |
|---|---|---|
| scrape public course FAQ pages (334 questions, 16 course groups after cleaning) | `extract_faq.py` | `data/external/faq_pairs.json` |
| baselines and a logistic-regression reranker, leave-one-course-out | `train_reranker.py` | `results/reranker.json` |
| fine-tune MiniLM, 4-fold CV grouped by course | `finetune_cv.py` | `results/finetune_cv.json` |
| final model (public FAQ pairs + COMP1945 train-split questions) and threshold | `train_engine.py` | `engine_model/` (the 87 MB model folder is git-ignored; rerun this script to rebuild it) |

Held-out courses (the test course's questions and answers never appear in training), question -> its own FAQ answer among that course's answers:

| retriever | Top-1 | Top-3 |
|---|---|---|
| BM25 | 0.536 | 0.808 |
| TF-IDF word | 0.545 | 0.772 |
| logistic regression over BM25/TF-IDF/MiniLM features | 0.65 | 0.844 |
| MiniLM, off the shelf | 0.668 | 0.85 |
| **MiniLM, fine-tuned on other courses' FAQs (deployed)** | **0.707** | **0.901** |

Threshold 0.497: on held-out courses the engine answers 67.7% of questions, and 80.1% of those answers are the right passage; the rest go to the TA.

Tests: `node e2e_engine.cjs <syllabus.html>` with the server running (uploads the file as HTML and as a browser-made PDF, checks FAQ, answers, escalation, switching back to COMP1945). Playwright is not a repo dependency: install it anywhere and point `NODE_PATH` at it.
