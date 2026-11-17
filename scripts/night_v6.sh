#!/usr/bin/env bash
# Overnight chain for the v6 candidate (simulator with ball-player incoherence, log 046).
#
#   bash scripts/night_v6.sh > experiments/outputs/night_v6.log 2>&1
#
# 0. self-training second judge on Sportec (s4_12, ~25 min, independent of v6);
# 1. calibrate the ghost rates without labels (configs/degrade_v6.json), if not done;
# 2. GBM screen with criteria fixed in advance (s4_11) -> stops here unless it passes;
# 3. full nested LOMO (GBM + M3 on CPU in parallel with the local TCN on MPS), decoding;
# 4. final decoders, reproducible CPU build v6 (keeps its models), Metrica Sample with those CPU weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=.venv311/bin/python
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }

[ -f configs/degrade_v6.json ] || { stamp "calibrating ghosts"; $PY scripts/s4_10_ghost_calibration.py; }
stamp "self-training, second judge (Sportec transductive LOMO)"
$PY scripts/s4_12_self_training_lomo.py || stamp "self-training LOMO failed (continuing)"
stamp "screen"
$PY scripts/s4_11_ghost_screen.py
if ! $PY -c 'import json,sys; sys.exit(0 if json.load(open("experiments/outputs/047/summary.json"))["pass"] else 1)'; then
    stamp "screen did not pass the criteria: stopping (no long run)"; exit 0
fi
stamp "screen passed: nested LOMO"
STORE_ARGS="--config configs/degrade_v6.json --features v1ctl --tcn-norm channel"
$PY scripts/s3_7_nested_full.py $STORE_ARGS --models gbm,trans --no-decode > experiments/outputs/night_v6_cpu.log 2>&1 &
CPU=$!
$PY scripts/s3_7_nested_full.py $STORE_ARGS --models tcn --no-decode > experiments/outputs/night_v6_tcn.log 2>&1
wait $CPU
$PY scripts/s3_7_nested_full.py $STORE_ARGS --decode-only
stamp "final decoders"
$PY scripts/s4_3_final.py decoders --store nested_degrade_v6_v1ctl_tcnlocal --name v6_v1ctl_tcnlocal
stamp "reproducible build"
$PY scripts/make_submission.py --name v6 --degrade-config configs/degrade_v6.json --control --transition --tcn --tcn-norm channel \
    --proba mean --change trans --decoder-config configs/decoder_v6_v1ctl_tcnlocal_ens_m3.json --save-models
stamp "Sample with the CPU weights"
$PY scripts/s4_3_final.py sample --models local_data/models/v6/models.pkl --decoder-config configs/decoder_v6_v1ctl_tcnlocal_ens_m3.json \
    --proba mean --change trans --out experiments/outputs/047/sample_v6_cpu.json
stamp "done"
