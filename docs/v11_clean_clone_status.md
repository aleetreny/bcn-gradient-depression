# Retrained v11: clean-clone reconstruction status

The delivered file is the retrained v11 CSV, `grand` variant, with SHA-256
`cc5bf45fc040eead80a3d66e7e974192b3a600f29d80f58d5bbb489bafbca2a3` (published second-half kickoff boundary applied).
An earlier local export of the `d3` variant from the same synthesis had SHA-256
`b19368564f1e031986d6003117dfc085a343960468186a5ced141d602f001db5`; it was not delivered. The pre-boundary candidate had
SHA-256 `7b893cecb18121f459bfa04d2035b61a10884089287eeb4b98a951be20f5834d`.
The pre-boundary candidate's registered external gate is REJECT against the
historical v11 reference; it is not an improvement. Nothing in this document
selects a model using J3.

**Summary.** The complete registered chain, from the v8.5 candidates to the
official-kickoff export, regenerates the delivery CSV byte for byte in a fresh
clone. The chain still consumes a declared set of retained intermediate
predictions (listed below) that cannot be regenerated bit for bit by retraining.
An outside party with only this repository and the data bundle therefore cannot
yet obtain the identical file; retraining those pieces would yield a different
candidate.

## End-to-end chain in a fresh clone

On 2026-11-17 a new clone of the source branch was created on the development
Mac (Apple M5 Pro, macOS, Python 3.11 research environment with CatBoost and
XGBoost from a local package directory). Every product of the chain (45 cached
files, including historical submission folders) was removed from its
`experiments/outputs/` before running, so none could be reused. The remaining
inputs were staged explicitly and recorded with SHA-256 hashes in a private local
manifest. `scripts/v11_big_leap/run_v11_chain.py` (and an earlier equivalent
local driver) then ran, in order, `v85_candidates`, `v85_streams`,
`v9_diagnostic_streams`, `v9_duel_candidates`, `v9_mbr_candidates`,
`v9_enriched_candidates`, `d3_verified_streams` (main and count),
`v85_baseline`, `v9_baseline`, `v10_b1_candidates`, `v10_b2_candidates`,
`v11_enriched_candidates`, `v11_local_synthesis` and
`v11_official_kickoff_export`. The whole chain takes about six minutes on CPU.

Four runs produced the same synthesis bundle
(`73bd2b9d80d23b790ede3b47dd54d78e0c3941440309a37ddf49933a6d483067`). The
first three exported the `d3` variant (`b19368564f1e031986d6003117dfc085a343960468186a5ced141d602f001db5`); the fourth, with the committed
runner and `variant` set to `grand`, exported the delivered file (`cc5bf45fc040eead80a3d66e7e974192b3a600f29d80f58d5bbb489bafbca2a3`).
Both passed the official format validator. The external harness predictions of
the two variants are identical; they differ only on the match. No public-window
score was computed.

### Inputs rebuilt from data in clean clones

- PlayerSet, trajectory and `v6_genuine` held-out folds (7 each), retrained from
  public Sportec data and regenerated tensors/features.
- Match and Metrica-sample inference of the L1 tree, PlayerSet, trajectory and
  PhysConformer heads, recomputed on CPU from the retained local full-model
  weights and regenerated inputs.
- Plus, ghost and physics-v2 tree held-out folds (7 each): each array identical
  to the retained reference.
- Boundary-model (`b4d`) folds (7): three identical; four differ numerically
  (in three by at most 2e-4 in a handful of values, in one substantially). With
  all seven rebuilt folds the CSV is unchanged.

### Inputs still retained (not regenerable bit for bit)

- PhysConformer held-out folds: retraining from identical inputs gives
  different outputs, also between repeats.
- Earlier neural heads (symmetric TCN and BiGRU, physics-v2 TCN/BiGRU) and their
  Metrica/match inference, originally trained on Apple MPS; no per-fold weights
  survive.
- The count-variant streams read by the synthesis step; they do not affect the
  delivered `grand` variant.
- Other earlier cached intermediates in `experiments/outputs/` (calibration and
  re-ranker summaries, W8 candidates) that the chain reads.

These are model predictions or match-derived caches and are not published.

## Earlier clean-clone checks (superseded by the run above)

The clone was made from local commit `4a52b66`; it is ahead of the published
branch. It used the existing local Python 3.11 environment, the original tactical
match bundle and the public Sportec dataset. Neither predictions nor previously
computed feature caches were copied into the clone for this check.

1. `scripts/make_submission_v1.py::ensure_degradation_params()` ran the existing
   S1.3 and S1.4 scripts against those data. The generated local
   `degradation_params.json` was byte-identical to the previous local file
   (SHA-256 `ed08956d7d847cb276aafd112e193a7dc60ceb870423687dbfce194004806d71`).
   S1.3 also emits a fixed public-window diagnostic; it was not used to adjust
   or select the candidate.
2. `python -m harness reconstruct pset_sportec_tensor` reconstructed the
   `J03WMX`, `v3_s0` PlayerSet tensor from the public Sportec raw files and
   those degradation parameters. All its arrays matched the retained tensor.
   This is one input realization, not a full model or pipeline reproduction.
3. The new CSV exporter fails without `_local/v11_export_inputs.json` and a
   verified synthesis bundle. The documented historical fast script checks for
   a system OpenMP installation and targets the historical CSV, not this one.

## What would close the gap

Make PhysConformer and the earlier neural heads deterministic (CPU, fixed
threads and seeds), retrain them and their full models, regenerate the D3 count
streams, rerun this chain and register the resulting candidate with the harness
gate. That candidate would be a different file from the current one.

## Follow-up clean-clone execution

The public Sportec PlayerSet training and test tensor realizations were built in
this clone from the raw public files and the regenerated degradation parameters;
their arrays matched the retained local inputs. The full set of PhysConformer
feature realizations was also rebuilt, and each feature table matched the local
reference. The sample and raw, repaired and video-reteamed match inference
tensors matched their references. For the last tensor, the complete video
detection and reteam process was subsequently rerun from the supplied video;
its reteam arrays matched the reference. The video run used the existing local
video Python environment and a local detector weight file, so standalone
installation and the applicable weight distribution terms remain to be tested.

All seven PlayerSet and trajectory held-out outputs were retrained in this clone
and matched the corresponding local candidate inputs. The first PhysConformer
held-out output did **not** match the candidate input, despite identical
feature tables. The remaining PhysConformer training queue stopped on that
mismatch. A repeat of that fold also produced different predictions. These are
intermediate checks; the end-to-end run above supersedes them.

## Third-party detector licensing

The video reconstruction used `ultralytics` 8.4.187 and the local `yolo11m.pt`
file. The competition repository's MIT notice applies to that repository's
contents, not automatically to third-party software, models, or match data.
Ultralytics describes its YOLO software and trained models as available under
AGPL-3.0 or an Enterprise license: see the [Ultralytics licensing page](https://www.ultralytics.com/license)
and its [repository license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
The applicability of each license and any obligations to the eventual delivery
need review in the context of the actual distribution. The match-data terms
remain separate; do not upload the match assets or derived tracking data.
The local weight file was used as an input; a fresh environment obtaining the
same authorized detector weights and installing the pinned video stack has not
been verified.

The clone also regenerated raw and repaired inference feature tables for all
three public samples and the supplied match; each matched its retained local
counterpart. Given the retained local tree model, its regenerated-input
inference matched the previous CPU inference. That invocation required CatBoost
and XGBoost from an existing local package directory, so it does not establish a
fresh dependency installation or a distributable model.

With the retained PlayerSet models, CPU inference on the regenerated tensors
was not bitwise identical to the prior local CPU inference. An isolated synthesis
replay using the regenerated PlayerSet, trajectory, PhysConformer and tree
inferences exported the same pre-boundary SHA-256 shown above and passed the
official format validator. This replay reused retained out-of-fold predictions,
other historical intermediate caches and model weights. It therefore does not
establish complete clean-clone reproducibility or a portable distribution. No
score or model selection was performed by this replay.

The new export component applies the organiser's published second-half timing
boundary to the synthesis before writing a new CSV. Repeated exports from the
same local synthesis match the current candidate hash; this does not remove the
dependencies on local weights or earlier intermediate caches.
