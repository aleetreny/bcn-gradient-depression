# Gradient Depression — retrained v11 candidate (draft)

**Task:** More than a Hack 2027, Phase 1 possession segmentation for BAR–ATH.  
**Candidate:** the locally generated retrained-v11 CSV with SHA-256
`b19368564f1e031986d6003117dfc085a343960468186a5ced141d602f001db5`.  
**Repository:** to be completed with the public URL and exact commit.  
**Status:** draft. Exact reproduction from a clean clone remains unverified; do
not present this document as proof that the submission requirements are met.

## Approach and data

The system combines possession, event and boundary predictions from several
tracking models with a segment-level decoding pipeline. The newer families
include PlayerSetNetV2, TrajectoryEventNetV10 and PhysConformerV11. Their
retrained local weights are not recovered historical weights. The provided
tracking, tactical video, homography and smart tags are used as inputs. The
pipeline also uses broadcast tracking to repair some missing tactical tracking
and a video-based team-colour check to correct certain tracking-slot assignments.
The person detector is an unmodified pretrained Ultralytics YOLO model.

Sportec/DFL open matches are used for supervised model training and external
leave-one-match-out evaluation. Metrica open sample matches are used for
transfer evaluation; a domain calibration was fitted and its activation selected
using these games, so they are not untouched for that decision. The D3 colour
classifier uses tracking-assigned team slots, not hand-labelled possession.
The public BAR–ATH sample is report-only under this repository's protocol; the
full historical selection history remains under review.

## Validation and limitations

The candidate was produced by an automatic export using the published
second-half timing boundary and passed the official submission-format validator.
Two local exports from the same synthesis input produced the same hash above.
The synthesis input was rebuilt using the retained local models and reused
historical intermediate caches. The prior candidate, before applying the
published timing boundary, was compared against the historical v11 reference;
its registered harness gate was:

- `J1 vs external_original_v11: REJECT delta -0.0062  90% CI [-0.0087, -0.0040]  folds improved 0/7 (need 5)`
- `J2 vs external_original_v11: REJECT delta -0.0036  90% CI [-0.0047, -0.0026]  folds improved 0/3 (need 3)`

These tests refer to the prior candidate, used fingerprinted external prediction bundles and are not a
clean-clone training or reproduction proof. The candidate should not be
represented as an improvement over the reference. J2 is not fully independent
of the calibration-selection decision described above.

The clean-clone work has regenerated several input representations and the
PlayerSet and trajectory held-out predictions, but the full chain still requires
intermediate artifacts. A repeated PhysConformer fold differed from the local
candidate's fold and from its own repeat. Numerical CPU inference outputs also
varied; a partial replay with all regenerated model inferences preserved the
final match labels while reusing other retained inputs. None of these partial
checks demonstrates exact reproduction of the final CSV on another computer.

## Reproduction and data access

The competition bundle must be obtained through the authorised competition
channel. The match assets and tracking-derived intermediate data must not be
redistributed. Sportec and Metrica public datasets must be obtained under their
own terms. The candidate currently depends on local weights, cached predictions,
feature blocks and synthesis inputs; a complete command from original inputs
is **not yet available**. See [`docs/v11_clean_clone_status.md`](../docs/v11_clean_clone_status.md)
for the verified steps and outstanding work. Before submission, verify whether
any necessary weights can be distributed and review third-party licenses,
including Ultralytics; the repository's MIT notice does not grant rights to the
match data or third-party models.
