#!/usr/bin/env bash
# Shows parity.py can fail: copy index.html to a temp file, change the JS confidence formula there
# (lookup conf p x factor -> p^2 x factor), require RED on the copy, then require GREEN on the real page. index.html is never written.
set -euo pipefail
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
.venv/bin/python - "$TMP/index.html" <<'EOF'
import sys
t = open("../index.html", encoding="utf-8").read()
a = "?lookupConf(pmax,q,sc)"; assert t.count(a) == 1
open(sys.argv[1], "w", encoding="utf-8").write(t.replace(a, "?lookupConf(pmax*pmax,q,sc)"))
EOF
echo "--- mutated copy: JS lookup conf uses p^2 instead of p"
if .venv/bin/python parity.py "$TMP/index.html" | tail -3; then echo "MUTATION SURVIVED"; exit 1; fi
echo "--- RED as expected on the mutated copy"
.venv/bin/python parity.py
echo "--- GREEN on the real index.html"
