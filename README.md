# CourseMate (COMP1945, Group 9 prototype)

Web chatbot for COMP1945 project questions. Every answer has a receipt: SOURCED (a course-document passage with its source), CALCULATED (what-if project mark from deterministic code) or ESCALATED ("Please ask the TA.").

The page runs a real ML pipeline in the browser: a TF-IDF + logistic-regression router (lookup / calc / out-of-scope), BM25 retrieval over 37 chunks of the course documents, and a confidence threshold chosen from a risk-coverage curve. The model is exported from Python (`ml/`) and inlined in `index.html` (also saved as `model.json`), so the page is one static file with no network calls. Demo scores are invented; edit them under the input box.

Results, with every number tied to the script that produced it: [results/RESULTS.md](results/RESULTS.md).

## Open locally

Double-click `index.html`, or `open index.html` (macOS).

## Rerun everything

```sh
bash run_all.sh          # creates ml/.venv on first run (python3; built and tested with 3.14), then runs every step below
```

One command per step (from `ml/`, after `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`):

```sh
.venv/bin/python build_corpus.py     # data/raw/* -> data/chunks.jsonl (37 chunks, source labels, HKT dates)
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
