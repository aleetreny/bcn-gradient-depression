"""Reconstruct v9 enriched candidates from existing algorithm-generated inputs.

This is a batch-reproduction utility, not a predictor and not a result gate.
The output may contain match-derived data and must remain on this machine.
"""
from pathlib import Path
import pickle
import numpy as np


def rebuild_enriched_candidates(root: Path, output_path: Path) -> Path:
    ROOT = Path(root).resolve()
    output_path = Path(output_path).resolve()
    try:
        relative_output = output_path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative_output.parts[:1] != ("_local",) and relative_output.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)
    OUT_DIR = ROOT / "experiments" / "outputs" / "v9_swarm"
    HOME_TEAM = 0

    with open(OUT_DIR / "_diag_duel_cands.pkl", "rb") as f:
        duel_cands_cache = pickle.load(f)
    with open(OUT_DIR / "_diag_mbr_cands.pkl", "rb") as f:
        mbr_cands_cache = pickle.load(f)

    matches = sorted(duel_cands_cache["v3"].keys())
    pset_v2_oof = {mid: np.load(OUT_DIR / "pset_v2_oof" / f"pset_v2_{mid}.npz") for mid in matches}
    with open(OUT_DIR / "pset_v2_oof" / "pset_v2_j23_preds.pkl", "rb") as f:
        pset_v2_j23 = pickle.load(f)

    l1_oof = {mid: np.load(OUT_DIR / "l1_trees_oof" / f"l1_trees_{mid}.npz") for mid in matches}
    with open(OUT_DIR / "l1_trees_oof" / "l1_trees_j23_preds.pkl", "rb") as f:
        l1_j23 = pickle.load(f)

    def enrich_duel(c, ps2_p, ps2_c, ps2_p4d, ps2_air, l1_g, l1_cb, l1_xgb, l1_air, l1_duel):
        ca, cb = int(c["ca"]), int(c["cb"])
        L = cb - ca
        y_lbl, x_lbl = int(c["y_lbl"]), int(c["x_lbl"])
        col_to_Y = 1 if y_lbl == HOME_TEAM else 0
        col_to_X = 1 if x_lbl == HOME_TEAM else 0
        pad = max(2, min(6, L // 6))
        ca_i, cb_i = ca + pad, cb - pad
        if cb_i <= ca_i:
            ca_i, cb_i = ca, cb
        T = len(ps2_p)
        wa0, wa1 = max(0, ca - 5), min(T, ca + 6)
        wb0, wb1 = max(0, cb - 5), min(T, cb + 6)
        c = dict(c)
        c["ps2_pY"] = float(np.mean(ps2_p[ca_i:cb_i, y_lbl]))
        c["ps2_max_pY"] = float(np.max(ps2_p[ca:cb, y_lbl]))
        l1_tree_P = (l1_g + l1_cb + l1_xgb) / 3.0
        c["l1_tree_pY"] = float(np.mean(l1_tree_P[ca_i:cb_i, y_lbl]))
        c["ps2_b4d_in"] = float(np.max(ps2_p4d[wa0:wa1, col_to_Y]))
        c["ps2_b4d_out"] = float(np.max(ps2_p4d[wb0:wb1, col_to_X]))
        c["ps2_b4d_geom"] = float(np.sqrt(max(0.0, c["ps2_b4d_in"] * c["ps2_b4d_out"])))
        c["ps2_pc_geom"] = float(np.sqrt(max(0.0, float(np.max(ps2_c[wa0:wa1])) * float(np.max(ps2_c[wb0:wb1])))))
        c["v9_air_int"] = float(0.5 * np.mean(ps2_air[ca_i:cb_i]) + 0.5 * np.mean(l1_air[ca_i:cb_i]))
        c["v9_duel_geom"] = float(np.sqrt(max(0.0, float(np.max(l1_duel[wa0:wa1])) * float(np.max(l1_duel[wb0:wb1])))))
        c["v9_duel_int"] = float(np.mean(l1_duel[ca_i:cb_i]))
        return c

    def enrich_mbr(c, ps2_p, ps2_p4d, ps2_air, l1_g, l1_cb, l1_xgb, l1_air, l1_duel):
        sa, sb = int(c["sa"]), int(c["sb"])
        y_lbl, x_lbl = int(c["y_lbl"]), int(c["x_lbl"])
        col_to_Y = 1 if y_lbl == HOME_TEAM else 0
        col_to_X = 1 if x_lbl == HOME_TEAM else 0
        T = len(ps2_p)
        wa0, wa1 = max(0, sa - 5), min(T, sa + 6)
        wb0, wb1 = max(0, sb - 5), min(T, sb + 6)
        c = dict(c)
        c["ps2_pY"] = float(np.mean(ps2_p[sa:sb, y_lbl]))
        c["ps2_pX"] = float(np.mean(ps2_p[sa:sb, x_lbl]))
        l1_tree_P = (l1_g + l1_cb + l1_xgb) / 3.0
        c["l1_tree_pY"] = float(np.mean(l1_tree_P[sa:sb, y_lbl]))
        c["l1_tree_pX"] = float(np.mean(l1_tree_P[sa:sb, x_lbl]))
        c["ps2_b4d_geom"] = float(np.sqrt(max(0.0, float(np.max(ps2_p4d[wa0:wa1, col_to_Y])) * float(np.max(ps2_p4d[wb0:wb1, col_to_X])))))
        c["v9_air_mean"] = float(0.5 * np.mean(ps2_air[sa:sb]) + 0.5 * np.mean(l1_air[sa:sb]))
        c["ps2_air_mean"] = float(np.mean(ps2_air[sa:sb]))
        c["l1_air_mean"] = float(np.mean(l1_air[sa:sb]))
        c["v9_duel_mean"] = float(np.mean(l1_duel[sa:sb]))
        return c

    duel_enr = {"v3": {}, "v6_std": {}, "v6_gen": {}, "j2": {}, "atd_rep": []}
    mbr_enr = {"v3": {}, "v6_std": {}, "v6_gen": {}, "j2": {}, "atd_rep": []}
    for split_key in ["v3", "v6_std", "v6_gen"]:
        v_tag = "v3" if split_key == "v3" else "v6"
        for mid in matches:
            z_ps = {k: pset_v2_oof[mid][k] for k in pset_v2_oof[mid].files}
            z_l1 = {k: l1_oof[mid][k] for k in l1_oof[mid].files}
            duel_enr[split_key][mid] = [
                enrich_duel(
                    c, z_ps[f"{v_tag}_proba"], z_ps[f"{v_tag}_change"], z_ps[f"{v_tag}_p4d"], z_ps[f"{v_tag}_p_aerial"],
                    z_l1[f"gbm_{v_tag}"], z_l1[f"cb_{v_tag}"], z_l1[f"xgb_{v_tag}"], z_l1[f"air_{v_tag}"], z_l1[f"duel_{v_tag}"]
                )
                for c in duel_cands_cache[split_key][mid]
            ]
            mbr_enr[split_key][mid] = [
                enrich_mbr(
                    c, z_ps[f"{v_tag}_proba"], z_ps[f"{v_tag}_p4d"], z_ps[f"{v_tag}_p_aerial"],
                    z_l1[f"gbm_{v_tag}"], z_l1[f"cb_{v_tag}"], z_l1[f"xgb_{v_tag}"], z_l1[f"air_{v_tag}"], z_l1[f"duel_{v_tag}"]
                )
                for c in mbr_cands_cache[split_key][mid]
            ]

    for gid in [1, 2, 3]:
        z_ps = pset_v2_j23["sample"][gid]
        z_l1 = l1_j23["sample"][gid]
        duel_enr["j2"][gid] = [
            enrich_duel(
                c, z_ps["pset_proba"], z_ps["pset_change"], z_ps["pset_p4d"], z_ps["pset_p_aerial"],
                z_l1["gbm"], z_l1["cb"], z_l1["xgb"], z_l1["air"], z_l1["duel"]
            )
            for c in duel_cands_cache["j2"][gid]
        ]
        mbr_enr["j2"][gid] = [
            enrich_mbr(
                c, z_ps["pset_proba"], z_ps["pset_p4d"], z_ps["pset_p_aerial"],
                z_l1["gbm"], z_l1["cb"], z_l1["xgb"], z_l1["air"], z_l1["duel"]
            )
            for c in mbr_cands_cache["j2"][gid]
        ]

    z_ps_a = pset_v2_j23["atd_rep"]
    z_l1_a = l1_j23["atd_rep"]
    duel_enr["atd_rep"] = [
        enrich_duel(
            c, z_ps_a["pset_proba"], z_ps_a["pset_change"], z_ps_a["pset_p4d"], z_ps_a["pset_p_aerial"],
            z_l1_a["gbm"], z_l1_a["cb"], z_l1_a["xgb"], z_l1_a["air"], z_l1_a["duel"]
        )
        for c in duel_cands_cache["atd_rep"]
    ]
    mbr_enr["atd_rep"] = [
        enrich_mbr(
            c, z_ps_a["pset_proba"], z_ps_a["pset_p4d"], z_ps_a["pset_p_aerial"],
            z_l1_a["gbm"], z_l1_a["cb"], z_l1_a["xgb"], z_l1_a["air"], z_l1_a["duel"]
        )
        for c in mbr_cands_cache["atd_rep"]
    ]

    with open(output_path, "xb") as f:
        pickle.dump({"duel": duel_enr, "mbr": mbr_enr}, f)
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root containing the input caches")
    parser.add_argument("--output", type=Path, required=True, help="new local output file; must not already exist")
    args = parser.parse_args()
    output = rebuild_enriched_candidates(args.root, args.output)
    print("Wrote local cache:", output)


if __name__ == "__main__":
    main()
