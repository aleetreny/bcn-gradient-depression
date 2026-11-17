"""Diagnose the exact mechanisms behind:
1. FP3_D_overlap_spill topologies: D->X->D, Y->X->D, D->X->Y across all 4 Judges.
2. Why 393 matched spells at IoU=0.3 fail IoU=0.7 (merged across missed turnover vs pass-in-flight kick-vs-receive lag).
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

EXP_LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_LAB / "py_pkgs"))
sys.path.insert(0, str(EXP_LAB))
sys.path.insert(0, str(EXP_LAB / "scripts"))
sys.path.insert(0, str(EXP_LAB / "evaluation"))

import possession_metrics as pm
from src.data.common import AWAY, DEAD, HOME
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.spell_reranker import rle_runs
from exp_w7a_player_set_net import load_cached_physv2_table


def extract_live_spells(arr: np.ndarray, half: np.ndarray):
    runs = rle_runs(np.where((half == 1) | (half == 2), arr, -1))
    spells = []
    for s, e, y in runs:
        s, e, y = int(s), int(e), int(y)
        if y in (HOME, AWAY) and half[s] in (1, 2):
            spells.append((y, int(half[s]), s, e))
    return spells


def extract_all_spells(arr: np.ndarray, half: np.ndarray):
    runs = rle_runs(np.where((half == 1) | (half == 2), arr, -1))
    spells = []
    for s, e, y in runs:
        s, e, y = int(s), int(e), int(y)
        if y in (HOME, AWAY, DEAD) and half[s] in (1, 2):
            spells.append((y, int(half[s]), s, e))
    return spells


def spell_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    union = max(e1, e2) - min(s1, s2)
    return float(inter) / float(union)


def get_spell_topologies(y_gt: np.ndarray, y_pred: np.ndarray, half: np.ndarray, proba: np.ndarray, X_df):
    g = np.asarray(y_gt, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    h = np.asarray(half, dtype=np.int64)

    gt_spells = extract_live_spells(g, h)
    pred_spells = extract_live_spells(p, h)
    all_pred_spells = extract_all_spells(p, h)

    idx_map = {(s, e, lbl): idx for idx, (lbl, _, s, e) in enumerate(all_pred_spells)}

    overlaps = []
    for gi, (gl, gh, gs, ge) in enumerate(gt_spells):
        for pj, (pl, ph, ps, pe) in enumerate(pred_spells):
            if gl != pl or gh != ph or pe <= gs or ps >= ge:
                continue
            iou = spell_iou(gs, ge, ps, pe)
            if iou > 0.0:
                overlaps.append((iou, gi, pj))
    overlaps.sort(key=lambda t: (-t[0], t[1], t[2]))

    g_flags = np.zeros(len(gt_spells), dtype=bool)
    p_flags = np.zeros(len(pred_spells), dtype=bool)
    pairs_03 = []
    for iou, gi, pj in overlaps:
        if iou < 0.3:
            break
        if g_flags[gi] or p_flags[pj]:
            continue
        g_flags[gi] = True
        p_flags[pj] = True
        pairs_03.append((gi, pj, iou))

    topologies = {
        "Y_X_Y": [],
        "D_X_D": [],
        "Y_X_D": [],
        "D_X_Y": [],
        "EDGE": [],
    }

    for pj, (pl, ph, ps, pe) in enumerate(pred_spells):
        if p_flags[pj]:
            continue
        aidx = idx_map[(ps, pe, pl)]
        prev_lbl = all_pred_spells[aidx - 1][0] if (aidx > 0 and all_pred_spells[aidx - 1][1] == ph) else -1
        next_lbl = all_pred_spells[aidx + 1][0] if (aidx + 1 < len(all_pred_spells) and all_pred_spells[aidx + 1][1] == ph) else -1
        opp = AWAY if pl == HOME else HOME
        plen = pe - ps
        gt_seg = g[ps:pe]
        frac_D = float(np.mean(gt_seg == DEAD))
        frac_opp = float(np.mean(gt_seg == opp))
        frac_self = float(np.mean(gt_seg == pl))
        mean_pD = float(np.mean(proba[ps:pe, DEAD]))
        max_pD = float(np.max(proba[ps:pe, DEAD]))
        mean_pX = float(np.mean(proba[ps:pe, pl]))

        info = {
            "len": plen, "frac_D": frac_D, "frac_opp": frac_opp, "frac_self": frac_self,
            "mean_pD": mean_pD, "max_pD": max_pD, "mean_pX": mean_pX,
            "ps": ps, "pe": pe, "pl": pl,
        }
        if prev_lbl == opp and next_lbl == opp:
            topologies["Y_X_Y"].append(info)
        elif prev_lbl == DEAD and next_lbl == DEAD:
            topologies["D_X_D"].append(info)
        elif prev_lbl == opp and next_lbl == DEAD:
            topologies["Y_X_D"].append(info)
        elif prev_lbl == DEAD and next_lbl == opp:
            topologies["D_X_Y"].append(info)
        else:
            topologies["EDGE"].append(info)

    # Analyze the 0.3 -> 0.7 cliff on matched spells
    cliff_causes = {
        "merged_across_missed_opp_or_D": 0,
        "split_single_gt_spell": 0,
        "pure_opp_boundary_lag": 0,
        "pure_D_boundary_lag": 0,
    }
    lag_details = []
    spd = X_df["ball_speed"].to_numpy(dtype=np.float64) if "ball_speed" in X_df else np.zeros(len(g))

    for gi, pj, iou in pairs_03:
        if iou >= 0.7:
            continue
        gl, gh, gs, ge = gt_spells[gi]
        pl, ph, ps, pe = pred_spells[pj]
        opp = AWAY if gl == HOME else HOME

        n_gt_overlapped = sum(1 for (gl2, gh2, gs2, ge2) in gt_spells if gl2 == gl and gh2 == gh and min(pe, ge2) > max(ps, gs2))
        n_pred_overlapped = sum(1 for (pl2, ph2, ps2, pe2) in pred_spells if pl2 == pl and ph2 == ph and min(ge, pe2) > max(gs, ps2))

        if n_gt_overlapped >= 2:
            cliff_causes["merged_across_missed_opp_or_D"] += 1
        elif n_pred_overlapped >= 2:
            cliff_causes["split_single_gt_spell"] += 1
        else:
            s_err = abs(ps - gs)
            e_err = abs(pe - ge)
            prev_gt = g[gs - 1] if gs > 0 and h[gs - 1] == gh else -1
            next_gt = g[ge] if ge < len(g) and h[ge] == gh else -1
            if (s_err >= e_err and prev_gt == opp) or (e_err > s_err and next_gt == opp):
                cliff_causes["pure_opp_boundary_lag"] += 1
                if s_err >= e_err:
                    lag_details.append(("entry", ps - gs, ge - gs, float(np.mean(spd[min(gs,ps):max(gs,ps)+1]))))
                else:
                    lag_details.append(("exit", pe - ge, ge - gs, float(np.mean(spd[min(ge,pe):max(ge,pe)+1]))))
            else:
                cliff_causes["pure_D_boundary_lag"] += 1

    return topologies, cliff_causes, lag_details


def main():
    sportec = load_sportec()
    w7b_dir = EXP_LAB / "experiments" / "outputs" / "exp_w7b_oof"
    X_v3 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", sportec[mid].half) for mid in MATCH_IDS}
    X_v6 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", sportec[mid].half) for mid in MATCH_IDS}
    w7b_oof = {T: np.load(w7b_dir / f"w7b_{T}.npz") for T in MATCH_IDS}

    for name, pred_k, prob_k, X_map in [
        ("J1A (Sportec v3)", "pred_v3_rerank", "proba_v3", X_v3),
        ("J1B-Std (Sportec v6)", "pred_v6_std_rerank", "proba_v6_std", X_v6),
        ("J1B-Gen (Sportec v6 Native)", "pred_v6_gen_rerank", "proba_v6_gen", X_v6),
    ]:
        agg_top = {"Y_X_Y": [], "D_X_D": [], "Y_X_D": [], "D_X_Y": [], "EDGE": []}
        agg_cliff = {"merged_across_missed_opp_or_D": 0, "split_single_gt_spell": 0, "pure_opp_boundary_lag": 0, "pure_D_boundary_lag": 0}
        agg_lags = []
        for T in MATCH_IDS:
            z = w7b_oof[T]
            t, c, l = get_spell_topologies(sportec[T].labels, z[pred_k], sportec[T].half, z[prob_k], X_map[T])
            for k in agg_top:
                agg_top[k].extend(t[k])
            for k in agg_cliff:
                agg_cliff[k] += c[k]
            agg_lags.extend(l)

        print(f"\n=====================================================================================")
        print(f"TOPOLOGY & CLIFF DIAGNOSIS: {name}")
        print(f"=====================================================================================")
        tot_fp = sum(len(v) for v in agg_top.values())
        print(f"  [A] Exact Neighbor Topology of All {tot_fp} False Positives (FP@0.3):")
        for k, lst in agg_top.items():
            if not lst:
                print(f"      {k:10s}:    0")
                continue
            m_len = np.mean([x["len"] for x in lst])
            m_fD = np.mean([x["frac_D"] for x in lst])
            m_fO = np.mean([x["frac_opp"] for x in lst])
            m_pD = np.mean([x["mean_pD"] for x in lst])
            m_pX = np.mean([x["mean_pX"] for x in lst])
            print(f"      {k:10s}: {len(lst):4d} ({100*len(lst)/max(tot_fp,1):5.1f}%) | mean_len={m_len:5.1f}f | GT_frac(D={m_fD:.2f}, Opp={m_fO:.2f}) | mean(pD={m_pD:.2f}, pX={m_pX:.2f})")

        tot_cliff = sum(agg_cliff.values())
        print(f"\n  [B] Exact Root Cause of All {tot_cliff} Matched@0.3 Spells That Fail IoU=0.7:")
        for k, v in agg_cliff.items():
            print(f"      {k:32s}: {v:4d} ({100*v/max(tot_cliff,1):5.1f}%)")

        if agg_lags:
            entry_lags = [x[1] for x in agg_lags if x[0] == "entry"]
            exit_lags = [x[1] for x in agg_lags if x[0] == "exit"]
            spds = [x[3] for x in agg_lags]
            print(f"      Pure Opp Entry Shifts (n={len(entry_lags)}): mean_signed={np.mean(entry_lags):+5.1f}f, mean_abs={np.mean(np.abs(entry_lags)):5.1f}f")
            print(f"      Pure Opp Exit Shifts  (n={len(exit_lags)}): mean_signed={np.mean(exit_lags):+5.1f}f, mean_abs={np.mean(np.abs(exit_lags)):5.1f}f | mean_ball_speed={np.mean(spds):.1f}m/s")


if __name__ == "__main__":
    main()
