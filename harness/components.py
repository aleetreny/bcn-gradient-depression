"""Built-in components: thin adapters over src/, imported lazily so `python -m harness list` stays fast
and CI (no lightgbm) can still import everything.

Adding a component = one adapter here (or in your own module imported from here) + its contract
tests run automatically (tests/test_harness_contracts.py iterates over every registered decoder).
"""
from __future__ import annotations

import json

import numpy as np

from harness import ROOT
from harness.contracts import check_features, check_labels, check_proba
from harness.registry import get, missing, register


def load_params(params: dict) -> dict:
    """`config = "configs/x.json"` loads that file (keys starting with _ are provenance); inline keys win."""
    params = dict(params or {})
    path = params.pop("config", None)
    base = {}
    if path:
        raw = json.loads((ROOT / path).read_text())
        base = {k: v for k, v in raw.items() if not k.startswith("_")}
    return {**base, **params}


# -- decoders: fn(proba, half, params, change=None) -> int8 labels ---------------------------------

@register("decoder", "argmax")
def argmax_decoder(proba, half, params, change=None):
    """Per-frame argmax; the floor every decoder must beat."""
    return np.argmax(proba, axis=1).astype(np.int8)


@register("decoder", "smooth", requires=("src/models/gbm.py",))
def smooth_decoder(proba, half, params, change=None):
    """Centred moving average inside each half, then argmax (src/models/gbm.smooth_argmax). params: window."""
    from src.models.gbm import smooth_argmax
    return smooth_argmax(np.asarray(proba, np.float64), np.asarray(half), int(params.get("window", 12)))


@register("decoder", "hsmm", requires=("src/decode/viterbi_hsmm.py",))
def hsmm_decoder(proba, half, params, change=None):
    """v4 semi-Markov decoder (src/decode/viterbi_hsmm.py). params: DecoderParams fields or config=."""
    from src.decode.viterbi_hsmm import DecoderParams, decode
    p = load_params(params)
    p = DecoderParams(**{k: p[k] for k in DecoderParams.__dataclass_fields__ if k in p})
    pc = None if change is None else np.asarray(change, np.float64)
    return decode(np.asarray(proba, np.float64), np.asarray(half, np.int8), p, p_change=pc)


# -- features: fn(match) -> DataFrame ------------------------------------------------------------

@register("features", "v1", requires=("src/features/v1.py",))
def v1_features(match):
    """160 slot-invariant features (src/features/v1.py), parquet-cached by match content."""
    from src.features.v1 import cached_features
    return cached_features(match, "harness_v1")


# -- models: factory(**params) -> fit(Xs, ys, groups) / predict_proba(X) --------------------------

@register("model", "gbm", requires=("src/models/gbm.py",))
def gbm_model(**params):
    """TwoHeadGBM (alive head x team head, isotonic-calibrated), src/models/gbm.py."""
    from src.models.gbm import TwoHeadGBM
    return TwoHeadGBM(**params)


# -- predictors: factory(**params) -> fit_predict(train, test) -> labels --------------------------

@register("predictor", "cached_external_report", requires=("src/eval/cached_external_report.py",))
def cached_external_report(which: str, bundle_sha256: str, fingerprints_sha256: str):
    """Score fingerprinted cached v11 predictions on external J1/J2 folds."""
    from src.eval.cached_external_report import external_bundle_predictor
    return external_bundle_predictor(which, bundle_sha256, fingerprints_sha256)

@register("predictor", "public_csv_report", requires=("src/eval/public_csv_report.py",))
def public_csv_report(which: str, sha256: str):
    """Report-only J3 evaluation of an explicitly hashed local CSV."""
    from src.eval.public_csv_report import public_csv_predictor
    return public_csv_predictor(which, sha256)

@register("predictor", "baseline", requires=("src/models/baseline.py",))
def baseline_predictor(window: int = 50):
    """Official nearest-player + majority vote baseline (src/models/baseline.py). params: window."""
    from src.models.baseline import predict

    def fit_predict(train, test):
        return predict(test, window=window)
    return fit_predict


@register("predictor", "staged")
def staged_predictor(features: str = "v1", model: str = "gbm", decoder: str = "hsmm",
                     model_params: dict | None = None, decoder_params: dict | None = None):
    """features -> model -> decoder, each looked up by name and checked at its boundary."""
    for kind, name in (("features", features), ("model", model), ("decoder", decoder)):
        gone = missing(kind, name)
        if gone:
            raise RuntimeError("{} {!r} needs {} (not on this branch)".format(kind, name, ", ".join(gone)))
    feat_fn, model_factory, dec_fn = get("features", features), get("model", model), get("decoder", decoder)

    def fit_predict(train, test):
        Xs, ys, groups = [], [], []
        for mid, m in train.items():
            Xs.append(check_features(feat_fn(m), m.n_frames, "features {} on {}".format(features, mid)))
            ys.append(np.asarray(m.labels))
            groups.append(mid)
        est = model_factory(**(model_params or {})).fit(Xs, ys, groups)
        X = check_features(feat_fn(test), test.n_frames, "features {} on test".format(features))
        proba = check_proba(est.predict_proba(X), test.n_frames, "model {}".format(model))
        return check_labels(dec_fn(proba, test.half, decoder_params or {}), test.n_frames, "decoder {}".format(decoder))
    return fit_predict


@register("predictor", "oracle_noise")
def oracle_noise_predictor(flip: float = 0.1, seed: int = 0, decoder: str = "smooth",
                           decoder_params: dict | None = None):
    """Truth + noise, then a decoder. Judge plumbing and decoder tests only; refuses unlabelled matches."""
    dec_fn = get("decoder", decoder)

    def fit_predict(train, test):
        if test.labels is None or test.source != "synthetic":
            raise RuntimeError("oracle_noise reads the test labels; it only runs on synthetic judges")
        rng = np.random.default_rng(seed)
        y = np.clip(test.labels, 0, 2)
        proba = np.full((len(y), 3), flip / 2)
        proba[np.arange(len(y)), y] = 1 - flip
        proba = 0.5 * proba + 0.5 * rng.dirichlet(np.ones(3), len(y))
        return dec_fn(proba, test.half, decoder_params or {"window": 12})
    return fit_predict


@register("reconstruction", "v9_enriched_candidates",
          requires=("src/v9_swarm/rebuild_enriched_candidates.py",))
def rebuild_v9_enriched_candidates(root, output_path):
    """Rebuild local enriched candidates from existing intermediate caches."""
    from src.v9_swarm.rebuild_enriched_candidates import rebuild_enriched_candidates
    return rebuild_enriched_candidates(root, output_path)


@register("reconstruction", "v9_duel_candidates", requires=("src/v9_swarm/rebuild_candidates.py",))
def rebuild_v9_duel_candidates(root, output_path):
    """Rebuild local duel candidates from pre-existing streams and feature caches."""
    from src.v9_swarm.rebuild_candidates import rebuild_candidates
    return rebuild_candidates(root, output_path, "duel")


@register("reconstruction", "v9_mbr_candidates", requires=("src/v9_swarm/rebuild_candidates.py",))
def rebuild_v9_mbr_candidates(root, output_path):
    """Rebuild local MBR candidates from pre-existing streams and feature caches."""
    from src.v9_swarm.rebuild_candidates import rebuild_candidates
    return rebuild_candidates(root, output_path, "mbr")


@register("reconstruction", "v9_diagnostic_streams", requires=("src/v9_swarm/rebuild_diagnostic_streams.py",))
def rebuild_v9_diagnostic_streams(root, output_path):
    """Rebuild local historical diagnostic streams from existing intermediate caches."""
    from src.v9_swarm.rebuild_diagnostic_streams import rebuild_diagnostic_streams
    return rebuild_diagnostic_streams(root, output_path)


@register("reconstruction", "v85_streams", requires=("src/v9_swarm/rebuild_v85_streams.py",))
def rebuild_v85_streams(root, output_path):
    """Rebuild local historical v8.5 streams from existing intermediate caches."""
    from src.v9_swarm.rebuild_v85_streams import rebuild_v85_streams as rebuild
    return rebuild(root, output_path)


@register("reconstruction", "v9_sportec_cache", requires=("src/v9_swarm/rebuild_sportec_cache.py",))
def rebuild_v9_sportec_cache(root, output_path):
    """Rebuild local W8B Sportec inputs from existing public OOF caches."""
    from src.v9_swarm.rebuild_sportec_cache import rebuild_sportec_cache
    return rebuild_sportec_cache(root, output_path)


@register("reconstruction", "v6_genuine_fold", requires=("src/v9_swarm/rebuild_v6_genuine.py",))
def rebuild_v6_genuine_fold(root, output_path):
    """Retrain one historical Sportec fold; probabilities may differ numerically."""
    from src.v9_swarm.rebuild_v6_genuine import rebuild_v6_genuine
    return rebuild_v6_genuine(root, output_path)


@register("reconstruction", "w8b_mode1_candidates", requires=("src/v9_swarm/rebuild_w8b_candidates.py",))
def rebuild_w8b_mode1(root, output_path):
    """Rebuild local W8B mode 1 candidates from the Sportec cache."""
    from src.v9_swarm.rebuild_w8b_candidates import rebuild_w8b_candidates
    return rebuild_w8b_candidates(root, output_path, "mode1")


@register("reconstruction", "w8b_mode2_candidates", requires=("src/v9_swarm/rebuild_w8b_candidates.py",))
def rebuild_w8b_mode2(root, output_path):
    """Rebuild local W8B mode 2 candidates from the Sportec cache."""
    from src.v9_swarm.rebuild_w8b_candidates import rebuild_w8b_candidates
    return rebuild_w8b_candidates(root, output_path, "mode2")


@register("reconstruction", "w8b_boundary_candidates", requires=("src/v9_swarm/rebuild_w8b_candidates.py",))
def rebuild_w8b_boundaries(root, output_path):
    """Rebuild local W8B boundary candidates from the Sportec cache."""
    from src.v9_swarm.rebuild_w8b_candidates import rebuild_w8b_candidates
    return rebuild_w8b_candidates(root, output_path, "boundaries")


@register("reconstruction", "pset_v2_fold", requires=("src/v9_swarm/rebuild_pset_fold.py",))
def rebuild_pset_v2_fold(root, output_path):
    """Retrain one public Sportec PSet fold locally; historical weights are not recovered."""
    from src.v9_swarm.rebuild_pset_fold import rebuild_pset_fold
    return rebuild_pset_fold(root, output_path)


@register("reconstruction", "pset_sportec_tensor", requires=("src/v9_swarm/rebuild_pset_tensor.py",))
def rebuild_pset_sportec_tensor(root, output_path):
    """Rebuild a public Sportec PlayerSet tensor cache locally."""
    from src.v9_swarm.rebuild_pset_tensor import rebuild_pset_tensor
    return rebuild_pset_tensor(root, output_path)


@register("reconstruction", "d3_verified_streams", requires=(
    "src/v10_first_principles/d3_input_preflight.py",
    "src/v10_first_principles/d3_verified_extraction.py",
    "src/v10_first_principles/rebuild_verified_streams.py",
))
def rebuild_d3_verified_streams(root, output_path):
    """Rebuild local D3 streams from explicitly verified inputs; excludes v11 heads."""
    from src.v10_first_principles.rebuild_verified_streams import rebuild_verified_streams
    return rebuild_verified_streams(root, output_path)


@register("reconstruction", "v10_b1_candidates", requires=("src/v10_first_principles/rebuild_b1_b2.py",))
def rebuild_v10_b1(root, output_path):
    """Rebuild local B1 candidates without public-window evaluation."""
    from src.v10_first_principles.rebuild_b1_b2 import rebuild_b1_b2
    return rebuild_b1_b2(root, output_path, "b1")


@register("reconstruction", "v10_b2_candidates", requires=("src/v10_first_principles/rebuild_b1_b2.py",))
def rebuild_v10_b2(root, output_path):
    """Rebuild local B2 candidates without public-window evaluation."""
    from src.v10_first_principles.rebuild_b1_b2 import rebuild_b1_b2
    return rebuild_b1_b2(root, output_path, "b2")


@register("reconstruction", "trajectory_local_inference", requires=("src/v10_first_principles/rebuild_trajectory_inference.py",))
def rebuild_trajectory_local_inference(root, output_path):
    """Infer local trajectory heads from explicitly hashed local models and tensors."""
    from src.v10_first_principles.rebuild_trajectory_inference import rebuild_trajectory_inference
    return rebuild_trajectory_inference(root, output_path)


@register("reconstruction", "trajectory_cpu_inference", requires=("src/v11_big_leap/rebuild_cpu_inference.py",))
def rebuild_trajectory_cpu_inference(root, output_path):
    """Run verified local trajectory inference on CPU."""
    from src.v11_big_leap.rebuild_cpu_inference import rebuild_trajectory_cpu
    return rebuild_trajectory_cpu(root, output_path)


@register("reconstruction", "v9_baseline", requires=("src/v9_swarm/rebuild_v9_baseline.py",))
def rebuild_v9_baseline(root, output_path):
    """Rebuild local v9 baseline predictions without public-window evaluation."""
    from src.v9_swarm.rebuild_v9_baseline import rebuild_v9_baseline as rebuild
    return rebuild(root, output_path)


@register("reconstruction", "phys_local_inference", requires=("src/v11_big_leap/rebuild_phys_inference.py",))
def rebuild_phys_local_inference(root, output_path):
    """Infer local PhysConformer heads from explicitly hashed models and features."""
    from src.v11_big_leap.rebuild_phys_inference import rebuild_phys_inference
    return rebuild_phys_inference(root, output_path)


@register("reconstruction", "phys_cpu_inference", requires=("src/v11_big_leap/rebuild_cpu_inference.py",))
def rebuild_phys_cpu_inference(root, output_path):
    """Run verified local PhysConformer inference on CPU."""
    from src.v11_big_leap.rebuild_cpu_inference import rebuild_phys_cpu
    return rebuild_phys_cpu(root, output_path)


@register("reconstruction", "trajectory_v10_fold", requires=("src/v10_first_principles/rebuild_trajectory_fold.py",))
def rebuild_trajectory_v10_fold(root, output_path):
    """Retrain one public Sportec trajectory fold locally with historical settings."""
    from src.v10_first_principles.rebuild_trajectory_fold import rebuild_trajectory_fold
    return rebuild_trajectory_fold(root, output_path)


@register("reconstruction", "phys_v11_fold", requires=("src/v11_big_leap/rebuild_phys_fold.py",))
def rebuild_phys_v11_fold(root, output_path):
    """Retrain one public Sportec PhysConformer fold locally with historical settings."""
    from src.v11_big_leap.rebuild_phys_fold import rebuild_phys_fold
    return rebuild_phys_fold(root, output_path)


@register("reconstruction", "full_public_model", requires=("src/v9_swarm/train_full_local.py",))
def train_full_public_model(root, output_path):
    """Train a full public-Sportec model and save its weights locally only."""
    from src.v9_swarm.train_full_local import train_full_local
    return train_full_local(root, output_path)


@register("reconstruction", "full_phys_model", requires=("src/v11_big_leap/train_phys_full_local.py",))
def train_full_phys_model(root, output_path):
    """Train a full public-Sportec PhysConformer and keep its weights local."""
    from src.v11_big_leap.train_phys_full_local import train_phys_full_local
    return train_phys_full_local(root, output_path)


@register("reconstruction", "pset_local_inference", requires=("src/v9_swarm/rebuild_pset_inference.py",))
def rebuild_pset_local_inference(root, output_path):
    """Infer local PSet heads from explicitly hashed local models and tensors."""
    from src.v9_swarm.rebuild_pset_inference import rebuild_pset_inference
    return rebuild_pset_inference(root, output_path)


@register("reconstruction", "l1_local_inference", requires=("src/v9_swarm/rebuild_l1_inference.py",))
def rebuild_l1_local_inference(root, output_path):
    """Infer local L1 trees from explicitly hashed models and features."""
    from src.v9_swarm.rebuild_l1_inference import rebuild_l1_inference
    return rebuild_l1_inference(root, output_path)


@register("reconstruction", "pset_cpu_inference", requires=("src/v11_big_leap/rebuild_cpu_inference.py",))
def rebuild_pset_cpu_inference(root, output_path):
    """Run verified local PlayerSet inference on CPU."""
    from src.v11_big_leap.rebuild_cpu_inference import rebuild_pset_cpu
    return rebuild_pset_cpu(root, output_path)


@register("reconstruction", "v85_baseline", requires=("src/v9_swarm/rebuild_v85_baseline.py",))
def rebuild_v85_baseline(root, output_path):
    """Rebuild the local strict v8.5 baseline from verified intermediate streams."""
    from src.v9_swarm.rebuild_v85_baseline import rebuild_v85_baseline as rebuild
    return rebuild(root, output_path)


@register("reconstruction", "v11_enriched_candidates", requires=("src/v11_big_leap/rebuild_v11_candidates.py",))
def rebuild_v11_enriched_candidates(root, output_path):
    """Rebuild local v11 candidates from explicit prerequisite caches without J3 scoring."""
    from src.v11_big_leap.rebuild_v11_candidates import rebuild_v11_candidates
    return rebuild_v11_candidates(root, output_path)


@register("reconstruction", "v11_local_synthesis", requires=("src/v11_big_leap/rebuild_v11_synthesis.py",))
def rebuild_v11_local_synthesis(root, output_path):
    """Synthesize local v11 prediction bundles without scoring the public window."""
    from src.v11_big_leap.rebuild_v11_synthesis import rebuild_v11_synthesis
    return rebuild_v11_synthesis(root, output_path)


@register("reconstruction", "v85_candidates", requires=("src/v9_swarm/rebuild_v85_candidates.py",))
def rebuild_v85_candidates(root, output_path):
    """Rebuild local v8.5 Sportec candidates from pre-existing model outputs."""
    from src.v9_swarm.rebuild_v85_candidates import rebuild_v85_candidates as rebuild
    return rebuild(root, output_path)


@register("reconstruction", "v11_export_local", requires=("src/v11_big_leap/export_local_submission.py",))
def export_v11_local(root, output_path):
    """Export a validated local CSV from an explicitly identified synthesis bundle."""
    from src.v11_big_leap.export_local_submission import export_local_submission
    return export_local_submission(root, output_path)


@register("reconstruction", "pset_training_inputs", requires=("src/v9_swarm/rebuild_pset_training_inputs.py",))
def rebuild_pset_training_inputs(root, output_path):
    """Build local public Sportec tensors for training and held-out inference."""
    from src.v9_swarm.rebuild_pset_training_inputs import rebuild_pset_training_inputs as build
    return build(root, output_path)


@register("reconstruction", "phys_sportec_features", requires=("src/v11_big_leap/rebuild_sportec_features.py",))
def rebuild_phys_sportec_features(root, output_path):
    """Build a local public Sportec PhysConformer feature realization."""
    from src.v11_big_leap.rebuild_sportec_features import rebuild_sportec_features
    return rebuild_sportec_features(root, output_path)


@register("reconstruction", "pset_raw_inference_tensor", requires=("src/v9_swarm/rebuild_inference_tensor.py",))
def rebuild_pset_raw_inference_tensor(root, output_path):
    """Build an inference tensor from raw local or public sample tracking."""
    from src.v9_swarm.rebuild_inference_tensor import rebuild_inference_tensor
    return rebuild_inference_tensor(root, output_path)


@register("reconstruction", "pset_repaired_inference_tensor", requires=("src/v9_swarm/rebuild_repaired_inference_tensor.py",))
def rebuild_pset_repaired_inference_tensor(root, output_path):
    """Build a local broadcast-repaired match inference tensor."""
    from src.v9_swarm.rebuild_repaired_inference_tensor import rebuild_repaired_inference_tensor
    return rebuild_repaired_inference_tensor(root, output_path)


@register("reconstruction", "pset_reteamed_inference_tensor", requires=("src/v9_swarm/rebuild_reteamed_inference_tensor.py",))
def rebuild_pset_reteamed_inference_tensor(root, output_path):
    """Build a local video-reteamed and broadcast-repaired inference tensor."""
    from src.v9_swarm.rebuild_reteamed_inference_tensor import rebuild_reteamed_inference_tensor
    return rebuild_reteamed_inference_tensor(root, output_path)


@register("reconstruction", "raw_inference_features", requires=("src/v11_big_leap/rebuild_inference_features.py",))
def rebuild_raw_inference_features(root, output_path):
    """Build local raw-match or public-sample inference feature caches."""
    from src.v11_big_leap.rebuild_inference_features import rebuild_inference_features
    return rebuild_inference_features(root, output_path)


@register("reconstruction", "repaired_inference_features", requires=("src/v11_big_leap/rebuild_repaired_features.py",))
def rebuild_repaired_inference_features(root, output_path):
    """Build local broadcast-repaired match inference feature caches."""
    from src.v11_big_leap.rebuild_repaired_features import rebuild_repaired_features
    return rebuild_repaired_features(root, output_path)


@register("reconstruction", "plus_public_fold", requires=("src/v9_swarm/rebuild_plus_oof.py",))
def rebuild_plus_public_fold(root, output_path):
    """Build one public Sportec plus-model fold required by the v11 chain."""
    from src.v9_swarm.rebuild_plus_oof import rebuild_plus_oof
    return rebuild_plus_oof(root, output_path)
