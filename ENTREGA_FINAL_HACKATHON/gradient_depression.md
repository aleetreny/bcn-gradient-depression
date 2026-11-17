# Gradient Depression — technical document

**Challenge:** More than a Hack 2027, Phase 1: ball possession, BAR–ATH.  
**Team:** Gradient Depression.  
**Possession file:** `gradient_depression.csv`, SHA-256
`cc5bf45fc040eead80a3d66e7e974192b3a600f29d80f58d5bbb489bafbca2a3`
(199,319 rows; passes `evaluation/validate_submission.py`).  
**Code:** https://github.com/aleetreny/gradient-depression-barca-2027 (commit `8567d6188c7020070702d67e6edc99fe94b6db75`).

## 1. Approach

The labels come from an algorithm in three stages. No step reads possession labels
for BAR–ATH.

1. **Frame-level models.** Several model families estimate, for every frame, the
   probability of H, A and dead ball (D), plus the probability of a possession
   change and of a four-way boundary (H→A, A→H, live→dead, dead→live). The
   families are gradient-boosted trees on hand-built physics features (ball–player
   distances, control and contact proxies, ball speed and acceleration),
   symmetric TCN and BiGRU sequence models, a permutation-invariant
   player-set network (PlayerSetNetV2), a trajectory transformer
   (TrajectoryEventNetV10) and a conformer over the physics features
   (PhysConformerV11). Every model is made symmetric under swapping H and A, so
   the two teams are treated identically.
2. **Stream synthesis and decoding.** The families are combined into one stream.
   The per-half marginals of P(D) and of the change probability are kept fixed,
   and the stream is decoded with a semi-Markov (HSMM) Viterbi decoder that
   imposes realistic spell lengths.
3. **Segment-level corrections.** Candidate short episodes are re-scored with
   multi-family consensus: deflections absorbed into the ongoing spell, genuine
   short steals carved out, last touches before a stoppage, the first touch after
   a restart, and boundaries snapped to the touch impulse frame. All thresholds of
   these corrections were fitted on external Sportec leave-one-match-out folds.

We chose an ensemble of different families because no single family was robust to
the tracking problems of the provided match (ball dropouts, slot swaps,
off-camera players). The segment stage exists because spell-F1 punishes small
spurious possessions much more than frame accuracy suggests.

## 2. Assets used, and assets dropped

| Asset | How it was used |
| :--- | :--- |
| DFL/Sportec open data, 7 matches (CC-BY 4.0, doi:10.6084/m9.figshare.28196177) | Supervised training of all models and decoder or segment thresholds; leave-one-match-out evaluation (judge J1), with clean and synthetically degraded tracking. |
| Metrica open sample, 3 matches | Transfer evaluation (judge J2). One domain-calibration step was fitted and selected on these games, so they are not untouched for that decision. Labels come from our own rules applied to Metrica's published events. |
| Provided BAR–ATH tracking, broadcast tracking, smart tags and match audio | Inference inputs only. Broadcast tracking repairs gaps in the tactical tracking. Smart tags and a fixed 3.95–4.15 kHz whistle filter on the match audio locate restarts and stoppages. |

**Dropped:** SkillCorner broadcast data as extra training data (it sees only
about 12 players per frame and made both judges worse); full-match instance
normalisation in sequence models (replaced by a 60-second local window); naive
linear blending of P(D) across heads (replaced by marginal-preserving synthesis).

No model, parameter or threshold was trained on the BAR–ATH possession ground
truth. Under this repository's protocol the public five minutes are
report-only. Historical development records show that public-window results were
visible while the dead-ball "DeadWedge" component was developed; we declare this
rather than claim that every historical design decision was blind to it.

## 3. Possession rules implemented

1. **Deflections and one-touch blocks.** A brief contact (up to about 2.2 s) by
   an opponent without proximity control does not break possession; the spell
   stays with the team that had it.
2. **Short recoveries.** When an opponent establishes close control (about
   0.45 m) and several model families agree on a clean transition in and out, a
   separate short possession is created (X → Y → X).
3. **Loose balls and passes in flight.** Possession stays with the last team in
   control until the opponent establishes controlled contact or play stops.
4. **Last touch before a stoppage, first touch after a restart.** A final touch
   by the other team just before a whistle or exit (X → Y → D), and control of the
   first phase after a set piece (D → Y → X), are labelled as possessions.
5. **Restarts and dead ball.** Throw-ins, corners, free kicks and goal kicks are
   D during preparation until the release impulse. Frames outside the tracked
   halves are D; the file contains no X labels. The second half starts at the
   first-play frame given in the organisers' FAQ, so frames 110,808–113,741 are D
   and frame 113,742 is the first live frame. The FAQ and `games.py` place this
   kickoff one frame apart; we follow the FAQ.

## 4. Validation

The final models are local retrainings on public data. The original v11 weights
were lost, so they could not be reused. Measured with the repository's harness
on external data, before the official kickoff boundary was applied, the
retrained version is slightly **worse** than the original v11 predictions:

- `J1 vs external_original_v11: REJECT delta -0.0062  90% CI [-0.0087, -0.0040]  folds improved 0/7 (need 5)`
- `J2 vs external_original_v11: REJECT delta -0.0036  90% CI [-0.0047, -0.0026]  folds improved 0/3 (need 3)`

We submit it anyway because, unlike the original, its match predictions can be
regenerated from models we still hold (section 6). We do not claim it is an
improvement. J2 is not fully independent of the Metrica calibration decision.

## 5. Limitations and what we would fix with more time

- **Exact retraining.** PhysConformer folds give different numbers when retrained
  from identical inputs, and the early TCN/BiGRU heads were trained on Apple GPU
  without surviving per-fold weights. With more time we would make these models
  deterministic on CPU, retrain them and register the resulting candidate.
- **Off-camera players.** Tight camera crops remove players from tracking; a
  trajectory-completion model trained on full-pitch data would help in aerial
  duels.
- **Boundary precision.** Joint training of the networks with the semi-Markov
  decoder on a spell-F1 surrogate would reduce sub-second boundary offsets.
- **Selection history.** A full audit of when each historical design choice was
  made, relative to the public window, is incomplete.

## 6. Reproduction

Requirements: macOS on Apple Silicon, Python 3.11 (`uv venv --python 3.11 .venv311`,
`uv pip install --python .venv311/bin/python -r requirements-submission.txt`,
`brew install libomp`), the competition bundle in `data/` (see `DATA.md`) and the
public datasets (`bash scripts/download_public_datasets.sh sportec` and
`... metrica`). The public smoke check is `python scripts/check_official.py`.

Every step is a registered component (`python -m harness list`, kind
`reconstruction`). `scripts/v11_big_leap/run_v11_chain.py` runs the full chain,
from the v8.5 candidates to the export with the official kickoff boundary, in
one checkout and exports the delivered variant (`grand`). It takes about six minutes on CPU. It checks the SHA-256 of
every staged input, refuses to reuse any previously generated chain product and
never scores the public window. The status of each step and input is in
`docs/v11_clean_clone_status.md`.

**What we verified.** In a fresh clone, with every chain product removed before
each run, the full chain regenerated this exact file (SHA-256 above); repeated
runs produced the same synthesis bundle.
The inputs staged for those runs are of two kinds:

- *Rebuilt in clean clones from the data:* PlayerSet, trajectory, `v6`, plus,
  ghost, physics-tree and boundary-model held-out folds. Also the match and
  Metrica-sample inference of the tree, PlayerSet, trajectory and PhysConformer
  models, using our locally held full-model weights.
- *Retained from earlier runs:* PhysConformer held-out folds, the early
  TCN/BiGRU heads and their inference, count-variant streams that the synthesis
  step reads (they do not affect the exported variant) and some earlier
  calibration and candidate caches. Retraining these does not
  reproduce them bit for bit.

**Intermediate files.** Model weights, predictions and match-derived caches are
not published in the repository. Retraining the retained pieces from scratch
gives a slightly different file, so the retained intermediate files, listed with
their SHA-256 in a local manifest, are needed to regenerate this exact file. We
can provide them to the organisers on request.
