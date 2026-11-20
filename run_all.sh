#!/usr/bin/env bash
# Rerun everything from scratch: corpus -> retrieval eval -> router -> export -> tests -> RESULTS.md.
# Usage: bash run_all.sh   (first run also creates ml/.venv and installs ml/requirements.txt)
set -euo pipefail
cd "$(dirname "$0")/ml"
[ -x .venv/bin/python ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
PY=.venv/bin/python
echo "== 0. frozen data check"; shasum -a 256 -c data/FROZEN.sha256
# keep results that run_all does not regenerate (slow external-FAQ runs and the pairwise judgement)
mkdir -p ../results && find ../results -type f ! -name finetune_cv.json ! -name 'reranker*.json' ! -name pairwise_judge.json ! -name 'faq_judge_*' ! -name 'faq_*sweep*' ! -path '*/faq_items/*' ! -path '*/e2e/*' ! -path '*/uat/*' ! -name 'parity_engine*' -delete
rm -rf artifacts data/chunks.jsonl
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
echo "== 13. eval_baselines.py"; if [ -d engine_model/minilm ]; then $PY eval_baselines.py 2>/dev/null | tail -0; echo "wrote results/baselines.json"; else echo "skipped: engine_model/minilm not built"; fi
echo "== 12. eval_agree.py";    if [ -d engine_model/minilm ]; then $PY eval_agree.py 2>/dev/null | tail -1; else echo "skipped: engine_model/minilm not built (train_engine.py)"; fi
echo "== 14. eval_faq.py test (syllabus -> FAQ, proxy metric; LLM-judge numbers: score_judge.py)"; if [ -d engine_model/minilm ]; then FAQ_CFG=120,3,0.3,0.05,seeds,1 $PY eval_faq.py test 2>/dev/null; else echo "skipped: engine_model/minilm not built"; fi
echo "ALL DONE"
