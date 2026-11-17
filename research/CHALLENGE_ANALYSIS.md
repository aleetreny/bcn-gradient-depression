# Challenge analysis — 21 October 2026

**Source of truth:** official [CHALLENGE.md](../CHALLENGE.md), [DATA.md](../DATA.md), [EVALUATION.md](../EVALUATION.md), [RULES.md](../RULES.md) and source implementation [possession_metrics.py](../evaluation/possession_metrics.py).

## 1. What is really being predicted?

Every one of **199,319** tactical-video frames (25 fps) receives H, A, D or X. The graded intervals are **11,185–90,651** and **110,808–190,197** inclusive, totalling **158,857** graded frames. Frames outside those intervals are ungraded. The second graded half starts **2,934 frames** before the second-half kickoff (113,742): these frames are labelled D in the organisers' description. Do not mistake the start of the second graded block for kick-off.

A possession in the scoring reference is a maximal uninterrupted H or A run while the ball is alive. A stoppage (D) separates possessions even if the same team resumes. A loose ball does **not** automatically change ownership; the opposing side must gain controlled possession. The reference comes mechanically from corrected tracking, not from watching the video and arbitrarily assigning possession events.

## 2. Data structure and what can go wrong

| Asset | What it offers | Main weakness |
| --- | --- | --- |
| Tactical ATD | Player/ball locations at 25 fps, 199,319 records; team membership by track slot | Ball is missing on about **41% of graded frames**; 46 fragmented slots for 22 players, not persistent identities |
| Smart Tagging | 638 automatically detected phases; team and restart / phase tags | Noisy, not corrected; phase boundaries are not possession boundaries |
| Tactical homography | Pitch-image mapping on every frame | Projection is useful, but coordinate errors remain |
| Tactical video | Full reference video | Not provided publicly; expensive visual modelling |
| Broadcast video + ATD | Alternative angles and tracking | Separate **164,488-frame timeline** with cuts/replays and no supplied mapping to tactical |
| Public ground truth | Five minutes; **17 possessions** | Too small for reliable hyperparameter selection |

**Timing trap:** tracking IDs and tactical video frames are different indices. In official 'games.py', 'id_frame = video_frame + offset[half]' and the two half offsets are 1,702,057 and 1,702,058. Respect the organiser's alignment helpers instead of applying one scalar throughout. Smart Tagging intervals are seconds from tactical-video frame zero; multiply by 25 fps, subject to rounding conventions.

The full-match ground truth is private. The supplied five-minute ground truth is publicly usable for validating alignment and understanding definitions. It is not an unbiased holdout for selecting many hyperparameters.

## 3. The objective function matters more than headline frame accuracy

**macroF1** averages class F1 over H, A and D per graded frame. **spellF1**, the ranking metric, extracts contiguous H and A possession spells, matches them one-to-one with the ground truth by temporal intersection-over-union, and averages spell F1 at IoU thresholds **0.3, 0.4, 0.5, 0.6, 0.7**.

That means:
- A few flickers can split one real possession into several predicted possessions and sharply reduce spellF1 despite a decent frame-level score.
- Heavy smoothing can erase short real possessions and shift boundaries too far.
- Detecting actual changes of team control and stoppages is more valuable than maximising raw per-frame accuracy.
- X does not behave as ordinary missingness under spellF1: it is forward-filled within each half (with back-fill at the start). During a stable possession this can protect continuity; across a true transition it can extend the wrong ownership. Under macroF1, graded X is a miss.

Published full-match official baseline results:

| Method | macroF1 | spellF1 |
| --- | ---: | ---: |
| Nearest-player raw | 0.6532 | 0.0486 |
| Smooth +/-25 frames | **0.6752** | 0.1996 |
| Smooth +/-50 frames | 0.6724 | **0.2433** |
| Smooth +/-125 frames | 0.6356 | 0.2242 |

The **public** five-minute test ranks smooth125 over smooth50 (spellF1 0.4242 vs 0.3256), the **opposite** of the full-match outcome (0.2242 vs 0.2433). This is an empirical warning against fine-tuning the five-minute sample.

## 4. Diagnosis of the official baseline

Official 'starter/baseline.py' finds the nearest observed player to the observed ball, scales x and y to real pitch metres, emits D when the ball or nearby player cannot be seen, then uses a centred majority vote.

It is simple and transparent, but its critical problems are:
1. **Ball occlusion is not dead ball.** Calling missing ball D creates false stoppages.
2. The nearest player at an instant may belong to a defender pressing the possessor or making a brief deflection.
3. It cannot explicitly model controlled possession, loose balls, restarts or phases.
4. Symmetric smoothing provides no uncertainty and can blur real changes.
5. It ignores the tagging and homography, which may supply evidence about possession and restarts.

We should first diagnose *where* possession fragments arise, not immediately train an expensive deep model.

## 5. A plausible strong approach

### Stage A: geometry and reliability-aware frame evidence

Compute side-level features in pitch metres (not raw normalised coordinates): closest-player distance to the ball for each team; gap between distances; distance and speed of the nearest 2–3 players; ball movement and velocity when present; spatial compactness, out-of-bounds evidence and distance to touchlines. Missing observations are masks, **not a synonym for D**. Treat track slots as team-level samples, not stable person IDs.

### Stage B: controlled-ball / stoppage evidence

Fuse tracking with Smart Tagging: tagged build-up/progression phases as noisy priors on the attacking team, restarts to localise D-to-H/A transitions, set pieces for possible stoppages. Check timestamp and team label orientation. Do not copy entire tag intervals into output uncritically.

### Stage C: sequence model

Start with an interpretable **three-state (H/A/D) HMM, Viterbi or hysteresis decoder** rather than a transformer. Use emission confidence from Stage A/B and transition penalties informed by controlled possession. Make switching to the opposing side harder than staying in the current possession, but permit short genuine turnovers and restarts. Compare fixed, event-sensitive and duration-aware penalties.

A semi-Markov model with realistic spell durations is a possible extension, but duration priors should not be fit tightly to the 17 public possessions.

### Stage D: optional local video model

Only if diagnosis proves tracking/tagging cannot resolve transitions, experiment with a video-based ball / referee / restart detector.

## 6. Validation discipline

- Verify 199,319 outputs, consecutive frame IDs, H/A orientation and full-half alignment before evaluating a model.
- Run the exact official validator on every candidate.
- Use public spellF1 and macroF1 as **sanity checks**, not as a leaderboard of hyperparameters.
- Inspect **predicted possession counts, median/mean durations, class proportions, D fragment lengths** and whether transitions coincide with plausible events. Plausibility checks are not substitute ground truth.
- Record each change with a fixed configuration, deterministic seed if relevant, full generation command and code commit hash.
- Look for robust gains under modest perturbations of window, penalty and missing-ball assumptions; beware scores driven by one or two public possessions.
- Finish with an end-to-end reproducibility test from original gated assets to byte-identical submission CSV.

## 7. Practical priorities

**Highest likely value:** correctly handling D and long ball-occlusion segments, preserving possession continuity while detecting authentic transitions. **Secondary value:** Smart Tagging-assisted restarts and duration-aware decoding. **Lower immediate priority:** rebuilding all computer vision/tracking or training a deep temporal classifier from scratch with only five minutes of public ground truth.

This is a *research plan*, not evidence of outperformance. We cannot assess the models on the private full-match labels before submission.

## 8. Rules and leakage risks

Official restrictions prohibit manually editing submitted frame labels, using undisclosed private ground truth or derived versions, and shipping a result that the public code cannot regenerate. Gated match assets are for the competition only and must not be publicly redistributed. Public five-minute GT and organiser-published baseline CSVs are distinct from the gated assets.
