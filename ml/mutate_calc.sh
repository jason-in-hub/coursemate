#!/usr/bin/env bash
# Shows test_calc.py can fail: change the Poster weight 10 -> 11 in the single weights source,
# require RED, restore (trap), verify the restore by checksum, require GREEN. Run from ml/.
set -euo pipefail
W=data/weights.json; BAK=$(mktemp); cp "$W" "$BAK"; SUM=$(shasum -a 256 "$W" | cut -d' ' -f1)
trap 'cp "$BAK" "$W"; rm -f "$BAK"' EXIT
.venv/bin/python - <<'EOF'
import json; p = "data/weights.json"; t = json.load(open(p))
next(i for i in t["items"] if i["key"] == "poster")["weight"] = 11
json.dump(t, open(p, "w"), indent=1)
EOF
echo "--- mutated: Poster weight 10 -> 11"
if CALC_MUTANT=1 .venv/bin/python test_calc.py; then echo "MUTATION SURVIVED: test is not checking weights"; exit 1; fi
echo "--- RED as expected; restoring"
cp "$BAK" "$W"
[ "$(shasum -a 256 "$W" | cut -d' ' -f1)" = "$SUM" ] && echo "--- restored, checksum matches"
.venv/bin/python test_calc.py
echo "--- GREEN after restore"
