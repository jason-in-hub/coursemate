#!/usr/bin/env bash
# Rerun everything from scratch: corpus -> retrieval eval -> router -> export -> tests -> RESULTS.md.
# Usage: bash run_all.sh   (first run also creates ml/.venv and installs ml/requirements.txt)
set -euo pipefail
cd "$(dirname "$0")/ml"
[ -x .venv/bin/python ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
PY=.venv/bin/python
echo "== 0. frozen data check"; shasum -a 256 -c data/FROZEN.sha256
rm -rf ../results artifacts data/chunks.jsonl
echo "== 1. build_corpus.py";   $PY build_corpus.py
echo "== 2. eval_retrieval.py"; $PY eval_retrieval.py 2> >(grep -v -e Warning -e Loading -e redirects -e "Could not reach" >&2)
echo "== 3. train_router.py";   $PY train_router.py
echo "== 4. export_model.py";   $PY export_model.py
echo "== 5. test_calc.py";      $PY test_calc.py | tail -1
echo "== 6. mutate_calc.sh";    ./mutate_calc.sh | grep -e "---" -e "exact match"
echo "== 7. parity.py";         $PY parity.py
echo "== 8. mutate_parity.sh";  ./mutate_parity.sh | grep -e "---" -e MISMATCH -e "parity:" | head -8
echo "== 9. node test.mjs";     (cd .. && node test.mjs | grep -e KNOWN -e FAIL -e "sample questions" -e conf_mode -e GREEN -e RED)
echo "== 10. eval_holdout.py";  $PY eval_holdout.py
echo "== 11. make_results.py";  $PY make_results.py
echo "ALL DONE"
