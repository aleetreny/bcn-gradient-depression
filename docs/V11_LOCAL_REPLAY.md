# Local v11 reconstruction: current scope

This is an intermediate reconstruction guide, not a verified clean-clone recipe. The registered components have been exercised as described in `V11_RECOVERY_STATUS.md`; several still require earlier model predictions, feature caches, the match bundle and local manifests. Keep all datasets, tracking-derived artifacts, predictions and models on this machine. Do not run the historical `full-local` or `modal` modes: they invoke a Modal upload from local data.

Run `python -m harness list` first and `python -m harness check` at the end. Use the project Python 3.11 environment. In a separate checkout, all destinations below must be absent before running: the reconstructors refuse to replace files. Create the required output directories locally first. The suggested destinations are private generated caches, never Git inputs.

## Earlier inputs

Before the recovered v8.5 steps can run, the earlier model predictions, feature tables, v8a inputs, degradation parameters, public Sportec and Metrica data, and licensed match bundle must already be available. Their generation from a clean checkout has not been verified. `v6_genuine_fold` can rebuild each Sportec `v6_genuine_<match>.npz` in `experiments/outputs/exp_w7b_oof/`; it retrains models and may produce numerically different probabilities.

`pset_sportec_tensor` can rebuild one public Sportec tensor input from a private `_local/pset_tensor_request.json` containing `match_id` and `tag`. The supported tags are the v3/v6 held-out realizations and recorded training realizations. The output must be a new `_local/` NPZ named with the degraded match fingerprint expected by the fold trainer; a mismatched name is rejected with the expected name. These tensors remain local. Synthetic tests cover the request and seeds. One complete public Sportec realization was regenerated and every tensor array matched its retained cache exactly; other realizations and a clean-clone sequence remain unverified.

`v9_sportec_cache` can build `experiments/outputs/exp_w8b_oof/sportec_fast_cache.pkl` from existing public folds and features. `w8b_mode1_candidates`, `w8b_mode2_candidates`, and `w8b_boundary_candidates` can then build the respective local candidate caches. These are separate from the later v8.5 candidate cache below.

## Local chain with retained prerequisites

For each command, use `python -m harness reconstruct NAME --root . --output PATH`. The order matters:

| Name | New output path |
| --- | --- |
| `v85_candidates` | `experiments/outputs/forensic_v7/v85_cands_cache.pkl` |
| `v85_streams` | `_local/v85_streams.pkl` |
| `v9_diagnostic_streams` | `experiments/outputs/v9_swarm/_diag_streams_s3.pkl` |
| `v9_duel_candidates` | `experiments/outputs/v9_swarm/_diag_duel_cands.pkl` |
| `v9_mbr_candidates` | `experiments/outputs/v9_swarm/_diag_mbr_cands.pkl` |
| `v9_enriched_candidates` | `experiments/outputs/v9_swarm/_diag_enr_cands.pkl` |

`d3_verified_streams` separately reconstructs control and count streams from explicit local source paths and a private hash manifest. Its `_local/d3_reconstruction.json` specifies the variant and roots; run it separately for `main` and `count`, with distinct outputs. Hash matching confirms consistency with the nominated files, not their historical provenance. The D3 reconstruction does not rebuild the v11 model heads.

Create `_local/v85_baseline_inputs.json` with `{path, sha256}` entries for the generated `v85_streams`, `diagnostic_streams` and verified main `control_streams`. Then continue:

| Name | New output path |
| --- | --- |
| `v85_baseline` | `experiments/outputs/v9_swarm/baseline_preds_cache.pkl` |
| `v9_baseline` | `experiments/outputs/v10_first_principles/v9_preds_cache.pkl` |
| `v10_b1_candidates` | `experiments/outputs/v10_first_principles/b1_v10_cands.pkl` |
| `v10_b2_candidates` | `experiments/outputs/v10_first_principles/b2_v10_cands.pkl` |
| `v11_enriched_candidates` | `experiments/outputs/v11_big_leap/v11_enriched_cands.pkl` |

The v9 and later steps additionally require the expected PSet, L1, trajectory and PhysConformer fold and inference caches. A complete regenerated set is not yet available. Whenever one of those inputs changes, rebuild every dependent cache: existing candidate files do not prove that they correspond to the new inputs.

Create `_local/v11_synthesis_inputs.json` with `{path, sha256}` entries named `v9`, `candidates` and `d3_streams`, pointing respectively to the generated v9 baseline, v11 candidate cache and verified count stream. `v11_local_synthesis` writes a new local prediction bundle, without running the historical public-window evaluator. The hashes identify the chosen files but do not prove that candidates and model predictions share the same provenance.

Finally, `_local/v11_export_inputs.json` identifies the synthesis bundle as a `{path, sha256}` entry and sets `variant` explicitly to `strict`, `grand` or `d3`. `v11_export_local` creates a new CSV under `_local/` or `submissions/` and runs the official format validator. Format validity alone does not establish competition eligibility, quality or identical-file reproduction.

## New model inputs

`pset_v2_fold`, `trajectory_v10_fold` and `phys_v11_fold` each train one public Sportec held-out fold from existing local tensors or features and write a new local NPZ. All seven PlayerSet and trajectory folds have been trained locally and checked for expected keys, shapes and finite values. PhysConformer folds are not yet complete. `full_public_model` and `full_phys_model` can train newly initialized full-Sportec models and write weights only under `_local/`; these particular trainer components have not yet been exercised in a full real run. Separately trained local full models for each family were used in the hybrid replay described in `V11_RECOVERY_STATUS.md`. The three `*_local_inference` components use private configurations with explicitly hashed model and tensor or feature paths. These are new models, not the missing historical weights, and the hybrid replay is not a clean-clone result.

The historical selection of DeadWedge and the second-half kickoff boundary remain unresolved. In the current local outputs, the requested inclusive pre-kickoff range is not entirely `D`. Do not describe these points as settled or select a model using J3.
