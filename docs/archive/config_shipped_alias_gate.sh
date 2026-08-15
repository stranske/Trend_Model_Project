#!/usr/bin/env bash
# Regression gate for forbidden shipped configuration aliases (verifier #5875).
set -euo pipefail
rg_exit=0
rg -n -e 'config\.legacy|from \.legacy|(^|[[:space:]])nan_policy:|^jobs:|weighting_(name|method):' config --glob '*.yml' --glob '*.yaml' || rg_exit=$?
case "$rg_exit" in
  0) echo "forbidden shipped configuration alias found" >&2; exit 1 ;;
  1) ;; # rg reports no matches with status 1; that is the passing result.
  *) exit "$rg_exit" ;;
esac
