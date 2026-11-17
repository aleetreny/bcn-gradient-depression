# Technical report template — BCN Gradient Depression

> **Do not submit this unfinished template.** Replace every TODO after the final reproducible pipeline is frozen. Export to PDF if desired, or submit Markdown as allowed.

## Team and submission identification

- Registered team name: **TODO** (must match the official registration)
- Members: **TODO**
- Public repository URL: **TODO** (make public before submitting)
- Exact commit SHA: **TODO**
- Final CSV filename and SHA-256 checksum: **TODO**

## 1. Executive summary

**TODO:** One paragraph: exact task, principal method and its distinctive contribution. Do not claim private full-match performance without results.

## 2. Approach and rationale

**TODO:** Define input signals, geometry, temporal modelling, emission scores and transition/episode rules. Explain why this was chosen over the raw nearest-player and uniform smoothing baseline.

## 3. Assets used and discarded

| Asset | Used? | How or why not? |
| --- | --- | --- |
| Tactical ATD | TODO | TODO |
| Smart Tagging | TODO | TODO |
| Tactical homography | TODO | TODO |
| Tactical video | TODO | TODO |
| Broadcast assets | TODO | TODO |

## 4. What counts as possession

**TODO:** Controlled possession, loose balls, deflections, duels, D during stoppages, restart ownership, handling missing ball observations, X handling. State how the approach approximates the official definitions.

## 5. Experiments and checks

**TODO:** Reference full-match published baseline (not our own private metric); public 5-minute sanity result, reproducibility, ablations and reasons for choosing the final method. Explain explicitly that 17 public possessions are not enough to establish generalisation.

## 6. Limitations and failure cases

**TODO:** Occlusion, identity switches, crowding, event tagging noise, domain dependence, boundary localisation, second-half restarts; what more time or data could improve.

## 7. Exact reproduction

~~~bash
git clone REPLACE_WITH_PUBLIC_REPO
cd bcn-gradient-depression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
# Put the competition data in ./data or set MTAH_DATA_DIR
python src/predict.py --output submissions/REPLACE_WITH_TEAM.csv
python evaluation/validate_submission.py submissions/REPLACE_WITH_TEAM.csv
~~~

**TODO:** update to the actual final code and pinned dependencies. Confirm that this procedure generates a byte-identical CSV.

## 8. Data-use and provenance

The data were provided for the competition under the organisers' terms and are not redistributed. Original starter and scoring code credit: https://github.com/metrica-sports/more-than-a-hack-2027.
