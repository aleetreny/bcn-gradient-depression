#!/usr/bin/env bash
# Regenerate the final submission (v4) from the competition data and the public Sportec matches, and check its hash.
#
#   bash scripts/reproduce_final.sh            # ~1 h on CPU; competition data expected in data/tactical/
#
# Steps: research environment (Python 3.11, pinned) -> Sportec/DFL open data -> build -> official
# format check -> SHA-256 against the frozen value. The reserve (GBM only, ~5 min):
#   RESERVE=1 bash scripts/reproduce_final.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv311/bin/python"
cd "$ROOT"

[ -f data/tactical/BAR-ATH_tactical_atd.txt ] || { echo "competition data missing: put it in data/tactical/" >&2; exit 1; }
bash scripts/setup_research_env.sh
bash scripts/download_public_datasets.sh sportec

if [ "${RESERVE:-0}" = "1" ]; then
    NAME=reserve_gbm_ctl
    EXPECTED=f067c06e661a403d26dcd4fec375967338ca2a92274787c1b15ea900f9700566
    "$PY" scripts/make_submission.py --name "$NAME" --control --decoder-config configs/decoder_v1ctl_gbm.json
else
    NAME=v4
    EXPECTED=dbc3607092354e4382795491be1e8d419e91c972a608c44815b03adc70eca942
    "$PY" scripts/make_submission.py --name "$NAME" --control --transition --tcn --tcn-norm channel \
        --proba mean --change trans --decoder-config configs/decoder_v1ctl_tcnlocal_ens_m3.json
fi

"$PY" evaluation/validate_submission.py "submissions/$NAME.csv"
GOT="$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "submissions/$NAME.csv")"
echo "sha256 $GOT"
if [ "$GOT" = "$EXPECTED" ]; then echo "OK: identical to the frozen submission"; else echo "MISMATCH: expected $EXPECTED" >&2; exit 2; fi
