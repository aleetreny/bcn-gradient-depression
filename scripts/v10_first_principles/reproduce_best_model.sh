#!/usr/bin/env bash
# ==============================================================================
# End-to-End Reproducibility Blueprint & Runner (`Regla 3` / `SUBMISSION.md`)
# Script: scripts/v10_first_principles/reproduce_best_model.sh
#
# Supports three execution modes:
#   1. --mode fast       (Default, < 120s): Replays only the final synthesis from the
#                        intermediate files of an earlier full run (NOT a fresh clone),
#                        then checks byte-for-byte SHA-256 identity of the final file,
#                        0.000e+00 H<->A equivariance, and official validate_submission.py.
#   2. --mode modal      (~20-30 min): Re-runs only the Modal training (Stage 1b) + synthesis;
#                        assumes Stage 0-1 of full-local already ran on this machine,
#                        deterministic CPU decoding & validation.
#   3. --mode full-local (~3.5 hrs + Modal): Complete clean-clone rebuild from
#                        raw tracking + video (`BAR-ATH.mp4`) + Modal training, incorporating
#                        all 3 `PLAN_NOCHE` Section 14 reproducibility fixes:
#                        - `scripts/v9_swarm/reproduce_b4d.py` (`SymBoundary4DirGBM`)
#                        - `scripts/noche/v85_orphans.py` (`SymBiGRU v1ctl`)
#                        - `py_pkgs/xgboost/lib/libomp.dylib` OpenMP runtime link fix
#                        plus `D3` video jersey re-teaming (`scripts/noche/players_reteam.py`).
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv311/bin/python}"
export PYTHONPATH=".:py_pkgs:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

MODE="fast"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    -h|--help)
      sed -n '2,23p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1 (use --mode fast|modal|full-local [--dry-run])" >&2
      exit 2
      ;;
  esac
done

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

verify_environment() {
  echo "------------------------------------------------------------------------------"
  echo "[ENV CHECK] Verifying Python environment & OpenMP library links..."
  echo "------------------------------------------------------------------------------"
  if [[ -d "$ROOT/py_pkgs/xgboost/lib" && ! -e "$ROOT/py_pkgs/xgboost/lib/libomp.dylib" ]]; then
    TORCH_OMP="$ROOT/.venv311/lib/python3.11/site-packages/torch/lib/libomp.dylib"
    if [[ -f "$TORCH_OMP" ]]; then
      echo "  Fixing broken symlink: py_pkgs/xgboost/lib/libomp.dylib -> $TORCH_OMP"
      ln -sf "$TORCH_OMP" "$ROOT/py_pkgs/xgboost/lib/libomp.dylib"
    fi
  fi
  # LightGBM and XGBoost wheels for macOS link against an external OpenMP runtime.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! { command -v brew >/dev/null && [[ -e "$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib" ]]; } \
       && [[ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib && ! -e /usr/local/opt/libomp/lib/libomp.dylib ]]; then
      echo "  ERROR: OpenMP runtime (libomp) not found. LightGBM/XGBoost cannot load without it." >&2
      echo "  Install it with:  brew install libomp" >&2
      exit 1
    fi
  fi
  "$PYTHON" -c "
import lightgbm, catboost, xgboost, torch
print(f'  Python OK | lightgbm={lightgbm.__version__} | catboost={catboost.__version__} | xgboost={xgboost.__version__} | torch={torch.__version__}')
"
}

validate_generated_submissions() {
  echo "------------------------------------------------------------------------------"
  echo "[OFFICIAL VALIDATION] Running evaluation/validate_submission.py on generated CSVs..."
  echo "------------------------------------------------------------------------------"
  for sub_dir in "$ROOT/experiments/outputs/v10_first_principles/submissions" "$ROOT/experiments/outputs/v11_big_leap/submissions"; do
    if [[ -d "$sub_dir" ]]; then
      for csv in "$sub_dir"/submission_*.csv; do
        "$PYTHON" evaluation/validate_submission.py "$csv"
      done
    fi
  done
  if [[ -d "$ROOT/ENTREGA_FINAL_HACKATHON" ]]; then
    for csv in "$ROOT/ENTREGA_FINAL_HACKATHON"/*.csv; do
      [[ -f "$csv" ]] && "$PYTHON" evaluation/validate_submission.py "$csv"
    done
  fi
}

# Expected SHA-256 of the submitted file (ENTREGA_FINAL_HACKATHON/gradient_depression.csv).
EXPECTED_SHA="03a637f62c23510d7900bb1a4895a21f11a5afc9ed2790f4679ba49fa275bc15"

verify_final_sha() {
  local csv="$ROOT/ENTREGA_FINAL_HACKATHON/gradient_depression.csv"
  echo "------------------------------------------------------------------------------"
  echo "[SHA-256] Comparing $csv with the submitted file..."
  echo "------------------------------------------------------------------------------"
  if [[ ! -f "$csv" ]]; then
    echo "  FAIL: $csv was not generated." >&2
    exit 1
  fi
  local got
  got="$(shasum -a 256 "$csv" | awk '{print $1}')"
  if [[ "$got" != "$EXPECTED_SHA" ]]; then
    echo "  FAIL: got $got, expected $EXPECTED_SHA" >&2
    exit 1
  fi
  echo "  OK: byte-for-byte identical to the submitted file ($got)"
}

# Intermediate files read by the final synthesis (run_v9/v10/v11 champion scripts). None of them is
# committed (experiments/outputs/ and local_data/ are git-ignored); they are produced by the
# earlier stages of --mode modal / --mode full-local.
check_required_caches() {
  local stage="$1"; shift
  local missing=0
  for rel in "$@"; do
    if [[ ! -e "$ROOT/$rel" ]]; then
      echo "  MISSING: $rel" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "[$stage] Required intermediate files are missing (listed above)." >&2
    echo "  --mode fast only replays the final synthesis from files a previous full run left on disk;" >&2
    echo "  it cannot run on a fresh clone. Run --mode full-local first." >&2
    exit 1
  fi
}

# TODO(Ale): no script in this repository writes these three files; the v9/v10/v11 champion scripts
# only read them. Commit the script that builds them and call it from --mode full-local, otherwise a
# fresh clone cannot regenerate the submission (RULES.md rule 3).
# Also read by the v8.5 chain with no writer in the repository (found in the fresh-clone review):
#   experiments/outputs/exp_w8b_oof/  (sportec_labels_half.npz and the other cached exp_w8b inputs)
#   experiments/outputs/exp_w7b_oof/v6_genuine_<match>.npz
ORPHAN_CACHES=(
  "experiments/outputs/v9_swarm/_diag_streams_s3.pkl"
  "experiments/outputs/v9_swarm/_diag_enr_cands.pkl"
  "local_data/v9_swarm/v85_streams_and_preds.pkl"
)

SYNTHESIS_CACHES=(
  "${ORPHAN_CACHES[@]}"
  "experiments/outputs/v9_swarm/pset_v2_oof/pset_v2_j23_preds.pkl"
  "experiments/outputs/v9_swarm/l1_trees_oof/l1_trees_j23_preds.pkl"
  "experiments/outputs/v10_first_principles/traj_v10_oof/traj_v10_j23_preds.pkl"
  "experiments/outputs/v11_big_leap/phys_conformer_v11_oof/pc11_j23_preds.pkl"
)

run_modal_training() {
  echo "Step M.0: Prepare & upload public-only Modal training bundle (7 Sportec + 3 Metrica Sample)"
  run_cmd "$PYTHON" scripts/v9_swarm/prepare_modal_bundle.py --upload
  echo "Step M.1: Parallel 8-container GPU training of SymPlayerSetNetV2 + local BAR-ATH inference"
  run_cmd "$PYTHON" scripts/v9_swarm/modal_train_pset_v2.py
  echo "Step M.2: Parallel 8-container 16-CPU training of 481-col L1 Trees & 3D Aerial/Duel + local BAR-ATH inference"
  run_cmd "$PYTHON" scripts/v9_swarm/modal_train_l1_trees_and_aerial.py
  echo "Step M.3: Parallel 8-container GPU training of SymTrajectoryEventNetV10 + local BAR-ATH inference"
  run_cmd "$PYTHON" scripts/v10_first_principles/modal_train_traj_event_net.py
  echo "Step M.4: Parallel 8-container GPU training of SymPhysConformerV11 + local BAR-ATH inference"
  run_cmd "$PYTHON" scripts/v11_big_leap/modal_train_phys_conformer_v11.py
}

finish_and_verify() {
  if [[ "$DRY_RUN" -eq 0 ]]; then
    validate_generated_submissions
    verify_final_sha
  fi
}

require_caches() {
  [[ "$DRY_RUN" -eq 0 ]] && check_required_caches "$@"
  return 0
}

verify_environment

case "$MODE" in
  fast)
    T0=$(date +%s)
    echo "=============================================================================="
    echo "[MODE 1: FAST REPLAY OF THE FINAL SYNTHESIS FROM AN EARLIER FULL RUN (< 120s)]"
    echo "         Needs the intermediate files of a previous --mode full-local run."
    echo "=============================================================================="
    require_caches "MODE 1" "${SYNTHESIS_CACHES[@]}"
    run_cmd "$PYTHON" scripts/v10_first_principles/extract_and_evaluate_d3_reteam.py
    run_cmd "$PYTHON" scripts/v10_first_principles/exp_touch_segment_lattice.py
    run_cmd "$PYTHON" scripts/v10_first_principles/run_v10_first_principles_champion.py
    run_cmd "$PYTHON" scripts/v11_big_leap/run_v11_big_leap_champion.py
    finish_and_verify
    T1=$(date +%s)
    echo "=============================================================================="
    echo "[MODE 1 COMPLETE] Final synthesis replayed, validated and SHA-256 checked in $((T1 - T0))s."
    echo "=============================================================================="
    ;;

  modal)
    T0=$(date +%s)
    echo "=============================================================================="
    echo "[MODE 2: MODAL CLOUD GPU + 128-CORE CPU PARALLEL REBUILD (~25-35 min)]"
    echo "         (Modal trains ONLY on public Sportec + Metrica; BAR-ATH runs 100% locally)"
    echo "=============================================================================="
    echo "         Assumes Stage 0-1 of --mode full-local already ran on this machine."
    run_modal_training
    echo "Step 2.5: Deterministic local CPU synthesis, D3 re-teamed stream fusion & SHA-256 verification"
    require_caches "MODE 2" "${SYNTHESIS_CACHES[@]}"
    run_cmd "$PYTHON" scripts/v10_first_principles/extract_and_evaluate_d3_reteam.py
    run_cmd "$PYTHON" scripts/v10_first_principles/exp_touch_segment_lattice.py
    run_cmd "$PYTHON" scripts/v10_first_principles/run_v10_first_principles_champion.py
    run_cmd "$PYTHON" scripts/v11_big_leap/run_v11_big_leap_champion.py
    finish_and_verify
    T1=$(date +%s)
    echo "=============================================================================="
    echo "[MODE 2 COMPLETE] Modal Cloud GPU/CPU rebuild + deterministic CPU decoding in $((T1 - T0))s."
    echo "=============================================================================="
    ;;

  full-local)
    T0=$(date +%s)
    echo "=============================================================================="
    echo "[MODE 3: COMPLETE LOCAL CLEAN-CLONE REBUILD FROM RAW DATA]"
    echo "=============================================================================="
    echo "Stage 0: D3 Video Jersey Re-Teaming from data/tactical/BAR-ATH_tactical.mp4"
    if [[ ! -f "$ROOT/local_data/video/detections_yolo11m_s2.npz" ]]; then
      run_cmd "$PYTHON" scripts/noche/video_detect.py --stride 2
    fi
    run_cmd "$PYTHON" scripts/noche/players_reteam.py --count
    run_cmd "$PYTHON" scripts/noche/players_reteam.py --all
    run_cmd "$PYTHON" scripts/noche/players_reteam.py --pairs

    echo "Stage 1: Broadcast Alignment + Complete v8.5 Base Pipeline (reproduce_v85_complete.sh)"
    # Broadcast-to-tactical clock alignment (local_data/broadcast_alignment.npz); the broadcast repair
    # in exp_e2_broadcast_repair.py, imported by the v8.5 chain, reads it.
    run_cmd "$PYTHON" scripts/s4_6_broadcast_align.py
    # Builds local_data/models/v4/models.pkl (via reproduce_final.sh) before anything reads it.
    run_cmd bash scripts/v9_swarm/reproduce_v85_complete.sh
    # Stand-alone broadcast-repair report; needs models.pkl, so it runs after the v8.5 chain.
    run_cmd "$PYTHON" scripts/exp_e2_broadcast_repair.py
    run_cmd "$PYTHON" scripts/noche/v85_orphans.py

    echo "Stage 1b: External-data model training on Modal (the v9/v10/v11 synthesis reads its outputs)"
    run_modal_training

    echo "Stage 2: v9 Super-Champion + v10 First-Principles + v11 Big-Leap Synthesis & Validation"
    require_caches "MODE 3" "${SYNTHESIS_CACHES[@]}"
    run_cmd "$PYTHON" scripts/v9_swarm/run_v9_super_champion.py
    run_cmd "$PYTHON" scripts/v10_first_principles/extract_and_evaluate_d3_reteam.py
    run_cmd "$PYTHON" scripts/v10_first_principles/exp_touch_segment_lattice.py
    run_cmd "$PYTHON" scripts/v10_first_principles/run_v10_first_principles_champion.py
    run_cmd "$PYTHON" scripts/v11_big_leap/run_v11_big_leap_champion.py
    finish_and_verify
    T1=$(date +%s)
    echo "=============================================================================="
    echo "[MODE 3 COMPLETE] Full clean-clone rebuild completed in $((T1 - T0))s."
    echo "=============================================================================="
    ;;

  *)
    echo "Invalid --mode '$MODE'. Supported modes: fast, modal, full-local" >&2
    exit 2
    ;;
esac
