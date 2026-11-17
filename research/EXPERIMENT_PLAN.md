# Experiment plan and log

The aim is **full-match spellF1**, not maximising accuracy on the public 5-minute slice. Use this file to record hypotheses and results only after they have actually been measured.

## Test stages

> **Update (21 Oct 2026):** the verified methods research in [metodos/05_plan_para_ganar.md](metodos/05_plan_para_ganar.md) supersedes the table below with a more detailed plan (E0–E6): train/validate on Sportec Open DFL with leave-one-match-out spellF1, then tune the temporal decoder on that metric.

| ID | Change compared with baseline | Hypothesis | Required checks | Status |
| --- | --- | --- | --- | --- |
| E00 | Reproduce official smooth50 | Verify starting point and alignment | Same 199,319-frame CSV and public score as official reference | Not run locally yet |
| E01 | Treat missing ball as uncertain / propagate last confident team | Reduce artificial possession fragmentation from occlusions | Changes in predicted spell counts and D run lengths; public sanity score | Planned |
| E02 | Threshold control: nearest team only when distance advantage is substantial | Ignore pressure/deflections without controlled possession | Sensitivity to threshold, false hand-overs | Planned |
| E03 | Smart Tagging priors for attacking team and restarts | Stabilise possession and improve D transitions | Time alignment; sanity checks at kickoffs/set pieces | Planned |
| E04 | Three-state HMM / Viterbi or hysteresis with transition penalties | Convert noisy frame evidence into coherent spells | Segment durations, transition location, robust penalty range | Planned |
| E05 | Duration-aware decoding | Avoid too many tiny spells without deleting real short ones | Ablate duration priors vs simple HMM | Conditional |
| E06 | Locally run video event/ball detector | Resolve hard occlusions where tracking fails | Strict data terms compliance; actual gains over E04 | Conditional |

## For every run

Capture:
- Git commit; command and dependencies; data bundle version/path (never the data itself)
- Configuration, random seed if relevant, full-match number of output rows and class distribution
- Predicted number of possessions and durations in seconds
- Official validator result and public macroF1/spellF1 as a weak smoke test
- Qualitative error type and decision to keep/discard the change

## Decision criteria

Prefer interpretable, repeatable changes that improve plausibility across **multiple parts of the match** without relying on private labels. Avoid selecting a model solely because it wins on the 17 known possessions. Keep a frozen, reproducible fallback at all times.

## Baseline commands

~~~bash
python scripts/check_official.py
python src/predict.py --output submissions/draft_v0.csv --window 50
python evaluation/validate_submission.py submissions/draft_v0.csv
~~~

The generation command requires the organiser-provided tactical ATD bundle. The public checks do not.
