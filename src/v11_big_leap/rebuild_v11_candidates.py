"""Rebuild the local v11 enriched candidate cache without scoring J3.

Requires existing algorithm-generated intermediate caches. Match-derived
outputs must remain on this machine; this does not establish a clean build.
"""
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import load_all as load_sportec
from src.decode.viterbi_hsmm import decode as decode_v4
from src.v10_first_principles.rebuild_b1_b2 import _features, _fold_predictions
from src.v10_first_principles.d3_reteam_and_repro import load_v85_fitted_shield_models
from src.v10_first_principles.touch_segment_lattice import extract_b2_v10_boundary_candidates, extract_mbr_xyx_candidates_v10, extract_touch_segment_proposals_v10
from src.v11_big_leap.unified_stream_and_lattice_v11 import enrich_b2_cand_v11, enrich_candidates_with_v11, extract_viterbi_discrepancy_candidates_v11, synthesize_unified_stream_v11


def rebuild_v11_candidates(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)
    outputs = root / "experiments/outputs"
    swarm = outputs / "v9_swarm"
    v10 = outputs / "v10_first_principles"
    v11 = outputs / "v11_big_leap"
    def read(path):
        with path.open("rb") as handle:
            return pickle.load(handle)
    streams = read(swarm / "_diag_streams_s3.pkl")["streams"]
    enr = read(swarm / "_diag_enr_cands.pkl")
    v9_grand = read(v10 / "v9_preds_cache.pkl")["grand"]
    b1_c = read(v10 / "b1_v10_cands.pkl")
    b2_c = read(v10 / "b2_v10_cands.pkl")
    sportec = load_sportec()
    samples = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    mids, gids = sorted(sportec), sorted(samples)
    suite = SimpleNamespace(
        half_v3={mid: m.half for mid, m in sportec.items()},
        gt_v3={mid: m.labels for mid, m in sportec.items()},
        half_sample={gid: m.half for gid, m in samples.items()},
        gt_sample={gid: m.labels for gid, m in samples.items()},
    )
    half_atd_dec = np.where(atd.graded, atd.half, 0)
    X_v3 = {mid: _features(root, f"sportec_{mid}_degrade_v3_s0", sportec[mid].half) for mid in mids}
    X_v6 = {mid: _features(root, f"sportec_{mid}_degrade_v6_s0", sportec[mid].half) for mid in mids}
    X_sample = {gid: _features(root, f"sample_{gid}_raw", samples[gid].half) for gid in gids}
    X_atd_rep = _features(root, "atd_rep_True_True_None", atd.half)
    ps_v3, ps_v6, l1_v3, l1_v6, t10_v3, t10_v6, pc11_v3, pc11_v6 = ({}, {}, {}, {}, {}, {}, {}, {})
    for mid in mids:
        for version, ps, l1, traj, phys in (("v3", ps_v3, l1_v3, t10_v3, pc11_v3),
                                            ("v6", ps_v6, l1_v6, t10_v6, pc11_v6)):
            ps[mid] = _fold_predictions(swarm / "pset_v2_oof" / f"pset_v2_{mid}.npz", version)
            l1[mid] = _fold_predictions(swarm / "l1_trees_oof" / f"l1_trees_{mid}.npz", version, suffix=True)
            traj[mid] = _fold_predictions(v10 / "traj_v10_oof" / f"traj_v10_{mid}.npz", version)
            phys[mid] = _fold_predictions(v11 / "phys_conformer_v11_oof" / f"pc11_{mid}.npz", version)
    ps_j23 = read(swarm / "pset_v2_oof/pset_v2_j23_preds.pkl")
    l1_j23 = read(swarm / "l1_trees_oof/l1_trees_j23_preds.pkl")
    t10_j23 = read(v10 / "traj_v10_oof/traj_v10_j23_preds.pkl")
    pc11_j23 = read(v11 / "phys_conformer_v11_oof/pc11_j23_preds.pkl")
    ps_sample = {g: {key[5:]: val for key, val in ps_j23["sample"][g].items()} for g in gids}
    l1_sample = l1_j23["sample"]
    t10_sample = t10_j23["sample"]
    pc11_sample = pc11_j23["sample"]
    ps_atd = {key[5:]: val for key, val in ps_j23["atd_rep"].items()}
    l1_atd, t10_atd, pc11_atd = l1_j23["atd_rep"], t10_j23["atd_rep"], pc11_j23["atd_rep"]
    v85_sh = load_v85_fitted_shield_models(root)
    mbr_c = {sp: {} for sp in ("v3", "v6_std", "v6_gen", "j2", "atd_rep")}
    vit_c = {sp: {} for sp in ("v3", "v6_std", "v6_gen", "sample")}
    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, c, p4d, fams = streams[sp_key][m]
            ps = ps_v3[m] if sp_key == "v3" else ps_v6[m]
            l1 = l1_v3[m] if sp_key == "v3" else l1_v6[m]
            t10 = t10_v3[m] if sp_key == "v3" else t10_v6[m]
            pc11 = pc11_v3[m] if sp_key == "v3" else pc11_v6[m]
            p_u, c_u, p4d_u, fams_u = synthesize_unified_stream_v11(
                p, c, p4d, fams, suite.half_v3[m], ps, l1, t10=t10, pc11=pc11
            )
            mbr_c[sp_key][m] = enrich_candidates_with_v11(
                [dict(x) for x in enr["mbr"][sp_key][m]], pc11=pc11, p4d_v11=p4d_u, t10=t10
            )
            for kind in ("XYX", "XYD", "DYX"):
                enrich_candidates_with_v11(b1_c[kind][sp_key][m], pc11=pc11, p4d_v11=p4d_u, t10=t10)
            for bc in b2_c[sp_key][m]:
                enrich_b2_cand_v11(bc, t10, pc11)
            pr_vit = decode_v4(p_u, suite.half_v3[m], v85_sh["loo_dec_hyb"][m], p_change=c_u)
            vit_c[sp_key][m] = extract_viterbi_discrepancy_candidates_v11(
                v9_grand[b_key][m], pr_vit, p_u, c_u, p4d_u, fams_u,
                suite.half_v3[m], X_dict[m], ps, l1, t10=t10, pc11=pc11, gt=suite.gt_v3[m]
            )
    for g in gids:
        p, c, p4d, fams = streams["sample"][g]
        p_u, c_u, p4d_u, fams_u = synthesize_unified_stream_v11(
            p, c, p4d, fams, suite.half_sample[g], ps_sample[g], l1_sample[g],
            t10=t10_sample[g], pc11=pc11_sample[g]
        )
        mbr_c["j2"][g] = enrich_candidates_with_v11(
            [dict(x) for x in enr["mbr"]["j2"][g]], pc11=pc11_sample[g], p4d_v11=p4d_u, t10=t10_sample[g]
        )
        for kind in ("XYX", "XYD", "DYX"):
            enrich_candidates_with_v11(b1_c[kind]["sample"][g], pc11=pc11_sample[g], p4d_v11=p4d_u, t10=t10_sample[g])
        for bc in b2_c["sample"][g]:
            enrich_b2_cand_v11(bc, t10_sample[g], pc11_sample[g])
        pr_vit_g = decode_v4(p_u, suite.half_sample[g], v85_sh["dec_v5_hyb"], p_change=c_u)
        vit_c["sample"][g] = extract_viterbi_discrepancy_candidates_v11(
            v9_grand["preds_sample"][g], pr_vit_g, p_u, c_u, p4d_u, fams_u,
            suite.half_sample[g], X_sample[g], ps_sample[g], l1_sample[g],
            t10=t10_sample[g], pc11=pc11_sample[g], gt=suite.gt_sample[g]
        )
    p_atd, c_atd, p4d_atd, fams_atd = streams["atd_rep"]
    p_u_a, c_u_a, p4d_u_a, fams_u_a = synthesize_unified_stream_v11(
        p_atd, c_atd, p4d_atd, fams_atd, half_atd_dec, ps_atd, l1_atd, t10=t10_atd, pc11=pc11_atd
    )
    mbr_atd = enrich_candidates_with_v11(
        extract_mbr_xyx_candidates_v10(
            v9_grand["pred_atd_rep"], p_atd, c_atd, p4d_atd, fams_atd, half_atd_dec, X_atd_rep, ps_atd, l1_atd
        ),
        pc11=pc11_atd, p4d_v11=p4d_u_a, t10=t10_atd
    )
    mbr_c["atd_rep"] = mbr_atd
    b1_atd = extract_touch_segment_proposals_v10(
        v9_grand["pred_atd_rep"], p_atd, c_atd, p4d_atd, fams_atd, half_atd_dec, X_atd_rep, ps_atd, l1_atd, t10=t10_atd
    )
    for kind in ("XYX", "XYD", "DYX"):
        enrich_candidates_with_v11(b1_atd[kind], pc11=pc11_atd, p4d_v11=p4d_u_a, t10=t10_atd)
    b2_atd = extract_b2_v10_boundary_candidates(
        v9_grand["pred_atd_rep"], half_atd_dec, p_atd, c_atd, p4d_atd, fams_atd, ps_atd, l1_atd, X_atd_rep
    )
    for bc in b2_atd:
        enrich_b2_cand_v11(bc, t10_atd, pc11_atd)
    pr_vit_atd = decode_v4(p_u_a, half_atd_dec, v85_sh["dec_v5_hyb"], p_change=c_u_a)
    vit_atd = extract_viterbi_discrepancy_candidates_v11(
        v9_grand["pred_atd_rep"], pr_vit_atd, p_u_a, c_u_a, p4d_u_a, fams_u_a,
        half_atd_dec, X_atd_rep, ps_atd, l1_atd, t10=t10_atd, pc11=pc11_atd
    )
    c11 = {
        "mbr": mbr_c,
        "b1": b1_c,
        "b2": b2_c,
        "vit": vit_c,
        "atd_ctrl": {"mbr": mbr_atd, "b1": b1_atd, "b2": b2_atd, "vit": vit_atd},
    }
    with output_path.open("xb") as handle:
        pickle.dump(c11, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
