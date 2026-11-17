# Retrained v11: clean-clone reconstruction status

The retrained candidate is not yet reproducible from a clean clone. After applying
the published second-half kickoff boundary, its local CSV SHA-256 is
`b19368564f1e031986d6003117dfc085a343960468186a5ced141d602f001db5`.
The pre-boundary candidate had SHA-256
`7b893cecb18121f459bfa04d2035b61a10884089287eeb4b98a951be20f5834d`.
These are targets for verification, not a claim that the historical submission
or an independently trained model is identical. The pre-boundary candidate's
registered external gate is REJECT; this document does not select a model using J3.

## Checked on the local clean clone

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

## Remaining work

- Build every required tensor and feature realization, and the earlier model
  and out-of-fold predictions used by the v8.5/v9 chain, from accessible data.
- Connect these artifacts, the new three-family models and the D3 inputs to a
  complete reproducible local workflow; remove reliance on ad-hoc local scripts
  and manifests. Test whether retraining can reproduce the frozen candidate's
  actual weights and output on a separately installed environment.
- Run the complete chain in a clean clone with the permitted original inputs,
  validate the generated CSV with the official validator and compare its hash.
  An exact hash match, not partial tensor equality, is the acceptance evidence.
- Document installation, pinned dependencies, runtime, and permitted inputs for
  another participant; make the necessary code available in the delivery repo.
  Match data must not be redistributed, and derived tracking data must not be
  published if it would reconstruct that data (RULES.md). A public model is
  allowed, but whether any particular weights are distributable depends on
  their training inputs and applicable terms; no match-derived model or cache
  should be published on the basis of this partial check.

The historical fast replay and the two matching local candidate exports reused
intermediate artifacts. Neither establishes this clean-clone requirement.

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
intermediate checks only; no clean end-to-end CSV has been produced.

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
