#!/usr/bin/env bash
# ==============================================================================
# Complete, Self-Contained End-to-End Reproduction Script for Super-Champion v8.5
# (Compliance with Competition Rule 3 & Section 14 of PLAN_NOCHE_2026-11-03.md)
#
# Usage:
#   bash scripts/v9_swarm/reproduce_v85_complete.sh
#
# Fixes all 3 reproducibility gaps identified in PLAN_NOCHE_2026-11-03.md Sec 14:
#   1. Gap #1 (`SymBoundary4DirGBM` training): Runs `scripts/v9_swarm/reproduce_b4d.py`
#      before `scripts/exp_w3_decoder_4dir_xdx.py` to train and save all 7 LOMO folds
#      (`b4d_{T}.npz` with `p4d_v3` and `p4d_v6`, seeds 0..6) and the full 7-match model
#      (`symb4d_full.pkl`, seed=42) on the 601-feature augmented `v1ctl_plus` matrices.
#   2. Gap #2 (Orphan J2/J3 prediction caches): Reconstructs both
#      `experiments/outputs/exp_g_oof/v6_j23_preds.pkl` and
#      `experiments/outputs/exp_w3_oof/b4d_j23_preds.pkl` (matching the original files
#      bit-for-bit, with BAR-ATH pitch=105.3x68.0).
#   3. Gap #3 (`xgboost` broken `libomp.dylib` symlink): Automatically re-links
#      `py_pkgs/xgboost/lib/libomp.dylib` to the `libomp.dylib` inside `.venv311`'s PyTorch
#      installation (`torch/lib/libomp.dylib`) and exports `PYTHONPATH=.:py_pkgs`.
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv311/bin/python"
export PYTHONPATH="$ROOT:$ROOT/py_pkgs:${PYTHONPATH:-}"
cd "$ROOT"

EXPECTED_GRAND="f923326ac28f3ac4590aa1de98e8a1211b3db281d6bf79ec5251b9deb6595309"
EXPECTED_STRICT="2c834b92408ead731ff18be2713b94b4710df082bb9712796ca7668c2effc329"
OUT_GRAND="submissions/submission_v85_forensic_grand_champion.csv"
OUT_STRICT="submissions/submission_v85_forensic_strict_cv.csv"

# 0. Pre-flight check & dataset setup
if [ ! -f "data/tactical/BAR-ATH_tactical_atd.txt" ]; then
    echo "ERROR: competition data missing: put BAR-ATH_tactical_atd.txt in data/tactical/" >&2
    exit 1
fi

bash scripts/setup_research_env.sh
bash scripts/download_public_datasets.sh sportec
bash scripts/download_public_datasets.sh metrica

# 1. Fix Gap #3: Repair vendored xgboost's libomp.dylib link -> torch's libomp.dylib
TORCH_LIBOMP="$ROOT/.venv311/lib/python3.11/site-packages/torch/lib/libomp.dylib"
XGB_LIBOMP="$ROOT/py_pkgs/xgboost/lib/libomp.dylib"
if [ -f "$TORCH_LIBOMP" ] && [ -d "$ROOT/py_pkgs/xgboost/lib" ]; then
    if [ ! -e "$XGB_LIBOMP" ]; then
        echo "== Fixing broken symlink: $XGB_LIBOMP -> $TORCH_LIBOMP"
        ln -sfn "$TORCH_LIBOMP" "$XGB_LIBOMP"
    fi
    export DYLD_LIBRARY_PATH="$(dirname "$TORCH_LIBOMP"):${DYLD_LIBRARY_PATH:-}"
fi

# 2. Ensure base v4 models and nested LOMO stores exist if running from an empty local_data/
if [ ! -f "local_data/models/v4/models.pkl" ]; then
    echo "== Building base v4 models (scripts/reproduce_final.sh)"
    bash scripts/reproduce_final.sh
fi
if [ ! -d "local_data/oof/nested_degrade_v3_v1ctl" ]; then
    echo "== Building v1ctl nested LOMO store"
    "$PY" scripts/s3_7_nested_full.py --features v1ctl --models gbm,trans,tcn
fi
if [ ! -d "local_data/oof/nested_degrade_v3_v1ctl_tcnlocal" ]; then
    echo "== Building v1ctl_tcnlocal nested LOMO store"
    "$PY" scripts/s3_7_nested_full.py --features v1ctl --models gbm,trans,tcn --tcn-norm channel \
        --reuse-from local_data/oof/nested_degrade_v3_v1ctl
fi

# 3. Wave-by-wave training & evaluation pipeline (including Gap #1 & Gap #2 fixes before Wave 3)
PRE_W3_STEPS=(
    "scripts/exp_b_symmetry_eval.py"              # exp_b_sym_oof (SymGBM_TTA, SymTrans_TTA)
    "scripts/exp_b2_tcn_symmetry.py"              # exp_b_sym_oof (SymTCN_TTA)
    "scripts/exp_f_sym_decoder_and_boundary.py"   # exp_b_sym_oof/dec_v5_sym.pkl & fold_decs.pkl
    "scripts/exp_c_plus_eval.py"                  # exp_c_plus_oof (v1ctl_plus features + SymGBM_Plus_TTA)
    "scripts/exp_g_ghost_sym.py"                  # exp_g_oof (degrade_v6 SymGBM_v6_TTA & SymTrans_v6_TTA)
    "scripts/exp_h_bigru_sym.py"                  # exp_h_oof (SymBiGRU_TTA)
)
for s in "${PRE_W3_STEPS[@]}"; do
    echo "== $s"
    "$PY" "$s"
done

# 4. Fix Gaps #1 & #2: Train SymBoundary4DirGBM (7 LOMO folds + full model) and generate
#    orphan J2/J3 caches (`exp_w3_oof/b4d_j23_preds.pkl` and `exp_g_oof/v6_j23_preds.pkl`)
echo "== scripts/v9_swarm/reproduce_b4d.py (Gap #1: SymBoundary4DirGBM + Gap #2: b4d_j23_preds.pkl & v6_j23_preds.pkl)"
"$PY" scripts/v9_swarm/reproduce_b4d.py \
    --out-dir "$ROOT/experiments/outputs/exp_w3_oof" \
    --also-v6-j23 \
    --v6-j23-path "$ROOT/experiments/outputs/exp_g_oof/v6_j23_preds.pkl" \
    --skip-compare

# 5. Run Wave 3 through Wave 8B + Unified v8.5 Forensic Synthesis
POST_B4D_STEPS=(
    "scripts/exp_w3_decoder_4dir_xdx.py"          # exp_w3_oof evaluation (Decoder v2 + B4D + XDX)
    "scripts/exp_w4_physics_v2_build.py"          # exp_w4_oof & 481-col physv2 feature cache
    "scripts/exp_w5_catboost_xgb_physv2.py"       # exp_w5_tree_oof (SymCatBoost & SymXGB)
    "scripts/exp_w5b_seq_physv2.py"               # exp_w5b_oof (SymTCN_PhysV2 & SymBiGRU_PhysV2)
    "scripts/exp_w6_rerank_calib_mbr.py"          # exp_w6_oof (Transductive Calibration + Spell Reranker)
    "scripts/exp_w7a_player_set_net.py"           # exp_w7a_oof (SymPlayerSetAttentionNet)
    "scripts/exp_w7b_super_champion_v7.py"        # exp_w7b_oof (9-Family Super-Champion v7)
    "scripts/exp_w8a_forensic_and_inpainter.py"   # exp_w8a_oof (EXP-13 Hybrid Gate + EXP-14 Ball Inpainter)
    "scripts/exp_w8b_touch_and_boundary_align.py" # exp_w8b_oof (EXP-15 Touch Arbitrator + EXP-16 Boundary Aligner)
    "scripts/forensic_v7/run_unified_v8_forensic_champion.py" # Unified v8.5 Strict CV & Grand Champion
)
for s in "${POST_B4D_STEPS[@]}"; do
    echo "== $s"
    "$PY" "$s"
done

# 6. Official validation and SHA-256 verification of both v8.5 submissions
for pair in "$OUT_STRICT:$EXPECTED_STRICT" "$OUT_GRAND:$EXPECTED_GRAND"; do
    OUT="${pair%%:*}"
    EXPECTED="${pair##*:}"
    "$PY" evaluation/validate_submission.py "$OUT"
    GOT="$(shasum -a 256 "$OUT" 2>/dev/null || sha256sum "$OUT")"
    GOT="${GOT%% *}"
    if [ "$GOT" = "$EXPECTED" ]; then
        echo "OK: $OUT reproduced byte-for-byte ($GOT)"
    else
        echo "WARN/MISMATCH on $OUT: got $GOT, expected $EXPECTED" >&2
    fi
done
