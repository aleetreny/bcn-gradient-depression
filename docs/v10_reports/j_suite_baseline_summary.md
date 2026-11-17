# Generalization Judge Suite (`J_SUITE`) & Error Atlas Baseline Audit

- **Runtime**: `48.23s` (`n_boot=1000` paired bootstrap over matches & halves)
- **Output JSON**: `experiments/outputs/v9_swarm/j_suite_baseline.json`

## 0. Official Submission Equivalence Check (`BAR-ATH`, `158,857` graded frames)

| Candidate | Official CSV | Graded Frames | Graded Exact Match | Diff Frames |
|---|---|---:|---:|---:|
| `v85_forensic_strict_cv` | `submissions/submission_v85_forensic_strict_cv.csv` | 158,857 | `100.0000%` | `0` |
| `v7_6family_balanced_xdx` | `submissions/submission_v7_6family_balanced_xdx.csv` | 158,857 | `100.0000%` | `0` |
| `v7_6family_local_newton_xdx` | `submissions/submission_v7_6family_local_newton_xdx.csv` | 158,857 | `100.0000%` | `0` |
| `v85_reteam_count` | `_local/v85sc-count/submissions/submission_v85_forensic_strict_cv.csv` | 158,857 | `100.0000%` | `0` |

## 1. Standard Judges & Live Turnover vs Stoppage Boundary Decomposition

| Judge | Description | `v7_balanced` | `v7_local_newton` | `v85_strict_cv` | `Δ(v8.5 - v7_bal)` | `90% CI (Match)` | `90% CI (Half)` | Folds `v8.5 > v7` | Halves `v8.5 > v7` |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| **J1A** | Sportec 7m `degrade_v3 s0` (All) | `0.7506` | `0.7509` | **`0.7588`** | `+0.0082` | `[+0.0041, +0.0130]` | `[+0.0048, +0.0116]` | `6/7` | `12/14` |
| **J1A_LiveOnly** | Sportec 7m `v3 s0` (**GT Dead Fixed** -> pure `H<->A`) | `0.7571` | `0.7567` | **`0.7644`** | `+0.0073` | `[+0.0036, +0.0117]` | `[+0.0043, +0.0106]` | `7/7` | `14/14` |
| **J1A_noD** | Sportec 7m `v3 s0` (`without_dead_margin`) | `0.7771` | `0.7764` | **`0.7837`** | `+0.0066` | `[+0.0029, +0.0104]` | `[+0.0038, +0.0096]` | `6/7` | `12/14` |
| **J1A_StoppageOnly** | Sportec 7m `v3 s0` (**GT Team Fixed** -> pure `Live<->D`) | `0.9643` | `0.9646` | **`0.9644`** | `+0.0001` | `[-0.0007, +0.0008]` | `[-0.0005, +0.0006]` | `3/7` | `6/14` |
| **J1B_Std** | Sportec 7m `degrade_v6 s0` (Standard decoder) | `0.7436` | `0.7403` | **`0.7451`** | `+0.0015` | `[-0.0026, +0.0050]` | `[-0.0013, +0.0040]` | `4/7` | `8/14` |
| **J1B_Gen** | Sportec 7m `degrade_v6 s0` (Generalist decoder) | `0.6897` | `0.6903` | **`0.7043`** | `+0.0146` | `[+0.0074, +0.0218]` | `[+0.0096, +0.0199]` | `7/7` | `14/14` |
| **J2_Raw** | Metrica Sample 1-3 Raw (has `NaN=D` artifact) | `0.8354` | `0.8376` | **`0.8389`** | `+0.0035` | `[+0.0005, +0.0065]` | `[+0.0006, +0.0063]` | `2/3` | `4/6` |
| **J2_noD** | Metrica Sample 1-3 `without_dead_margin` | `0.8737` | `0.8750` | **`0.8756`** | `+0.0019` | `[-0.0016, +0.0053]` | `[-0.0012, +0.0049]` | `2/3` | `4/6` |
| **J2_LiveOnly** | Metrica Sample 1-3 (**GT Dead Fixed** -> pure `H<->A`) | `0.8005` | `0.8021` | **`0.8011`** | `+0.0007` | `[-0.0026, +0.0039]` | `[-0.0024, +0.0037]` | `2/3` | `4/6` |
| **J2_StoppageOnly** | Metrica Sample 1-3 (**GT Team Fixed** -> pure `Live<->D`) | `0.9251` | `0.9252` | **`0.9243`** | `-0.0008` | `[-0.0014, -0.0002]` | `[-0.0014, -0.0003]` | `0/3` | `0/6` |

### `v85_forensic_strict_cv` Detailed Metrics Across All Judges

| Judge | `spell_f1_mean` | `macro_f1` | `F1@0.3` | `F1@0.5` | `F1@0.7` | `P@0.5` | `R@0.5` | `poss/min` (GT) | `% < 2.0s` (GT) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **J1A** | `0.7588` | `0.9393` | `0.8281` | `0.7653` | `0.6749` | `0.8382` | `0.7057` | `3.84` (`4.58`) | `22.1%` (`29.6%`) |
| **J1A_LiveOnly** | `0.7644` | `0.9499` | `0.8285` | `0.7707` | `0.6868` | `0.8356` | `0.7163` | `3.91` (`4.58`) | `23.2%` (`29.6%`) |
| **J1A_noD** | `0.7837` | `0.6227` | `0.8455` | `0.7872` | `0.7102` | `0.8513` | `0.7336` | `6.83` (`8.00`) | `24.6%` (`30.0%`) |
| **J1A_StoppageOnly** | `0.9644` | `0.9830` | `0.9761` | `0.9648` | `0.9514` | `0.9694` | `0.9604` | `4.53` (`4.58`) | `30.0%` (`29.6%`) |
| **J1B_Std** | `0.7451` | `0.9369` | `0.8187` | `0.7537` | `0.6567` | `0.8391` | `0.6853` | `3.72` (`4.58`) | `20.9%` (`29.6%`) |
| **J1B_Gen** | `0.7043` | `0.9275` | `0.7920` | `0.7132` | `0.5979` | `0.7858` | `0.6553` | `3.80` (`4.58`) | `21.8%` (`29.6%`) |
| **J2_Raw** | `0.8389` | `0.9236` | `0.8891` | `0.8512` | `0.7662` | `0.9079` | `0.8032` | `3.49` (`3.94`) | `20.6%` (`25.1%`) |
| **J2_noD** | `0.8756` | `0.6207` | `0.9121` | `0.8854` | `0.8229` | `0.9200` | `0.8546` | `5.32` (`5.73`) | `21.3%` (`23.7%`) |
| **J2_LiveOnly** | `0.8011` | `0.9475` | `0.8329` | `0.8075` | `0.7578` | `0.8128` | `0.8037` | `3.90` (`3.94`) | `27.8%` (`25.1%`) |
| **J2_StoppageOnly** | `0.9243` | `0.9405` | `0.9505` | `0.9267` | `0.8934` | `0.9351` | `0.9188` | `3.87` (`3.94`) | `26.5%` (`25.1%`) |

## 2. Oracle Headroom Decomposition on `v85_forensic_strict_cv` (Where `v8.5` Loses Points)

| Oracle Intervention | `J1A` (`v3 s0`) `spell_f1` | `J1A` Headroom Gain | `J2-Raw` `spell_f1` | `J2-Raw` Headroom Gain |
|---|---:|---:|---:|---:|
| **Baseline (`v8.5`)** | `0.7588` | `+0.0000` | `0.8389` | `+0.0000` |
| `perfect D` | `0.7644` | **`+0.0056`** | `0.8011` | **`-0.0378`** |
| `perfect team (alive frames)` | `0.9644` | **`+0.2056`** | `0.9243` | **`+0.0854`** |
| `perfect change timing (±3 s)` | `0.8049` | **`+0.0462`** | `0.8761` | **`+0.0371`** |
| `no spurious possessions` | `0.7829` | **`+0.0241`** | `0.8537` | **`+0.0148`** |
| `short possessions recovered` | `0.8572` | **`+0.0984`** | `0.9078` | **`+0.0689`** |
| `short possessions recovered (alive frames only)` | `0.8486` | **`+0.0898`** | `0.8755` | **`+0.0366`** |

## 3. Possession Duration Recall Breakdown (`IoU >= 0.5` & `IoU >= 0.3`)

### 3A. `J1A` (7 Sportec Matches, `2,803` GT Live Possessions)

| Duration Bucket | GT Spells | `v7_bal` Recall@0.5 | `v8.5` Recall@0.5 | `v8.5` Misses@0.5 | Share of All `v8.5` Misses | `v8.5` Recall@0.3 | `v8.5` Recall@0.7 | Mean IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **short (<2.0s)** | 919 | `45.3%` | **`45.7%`** | 499 | `54.5%` | `50.6%` | `39.6%` | `0.438` |
| **medium (2.0-5.0s)** | 946 | `68.1%` | **`68.8%`** | 295 | `32.2%` | `79.3%` | `58.2%` | `0.661` |
| **long (>5.0s)** | 1196 | `89.6%` | **`89.8%`** | 122 | `13.3%` | `95.7%` | `81.4%` | `0.849` |

#### Fine-Grained Duration Buckets on `J1A` (Separating Post-Processor `<34f` Floor vs Decodable `34-49f`)

| Fine Duration Bucket | GT Spells | `v8.5` Recall@0.5 | `v8.5` Misses@0.5 | Share of All Misses | Mean IoU |
|---|---:|---:|---:|---:|---:|
| `ultra_short (<1.36s / <34f)` | 566 | **`35.5%`** | 365 | `39.9%` | `0.344` |
| `short_decodable (1.36-2.0s / 34-49f)` | 353 | **`62.0%`** | 134 | `14.6%` | `0.590` |
| `medium (2.0-5.0s / 50-125f)` | 946 | **`68.8%`** | 295 | `32.2%` | `0.661` |
| `long (5.0-10.0s)` | 562 | **`84.2%`** | 89 | `9.7%` | `0.799` |
| `very_long (>10.0s)` | 634 | **`94.8%`** | 33 | `3.6%` | `0.893` |

### 3B. `J2-LiveOnly` & `J2-Raw` (3 Metrica Matches, `701` GT Live Possessions)

| Duration Bucket | GT Spells | `J2-Raw` (`v7_bal`) Rec@0.5 | `J2-Raw` (`v8.5`) Rec@0.5 | `J2-LiveOnly` (`v8.5`) Rec@0.5 | `J2-LiveOnly` Misses@0.5 | Share of `J2-LiveOnly` Misses |
|---|---:|---:|---:|---:|---:|---:|
| **short (<2.0s)** | 279 | `51.2%` | `50.9%` | **`55.2%`** | 125 | `57.6%` |
| **medium (2.0-5.0s)** | 287 | `84.7%` | `85.4%` | **`83.3%`** | 48 | `22.1%` |
| **long (>5.0s)** | 549 | `93.1%` | `93.1%` | **`92.0%`** | 44 | `20.3%` |

## 4. Event-Play Error Atlas (`compute_error_atlas`)

### 4A. `J1A` DFL Event Categories (7 Sportec Matches, `v8.5` vs `v7_balanced`)

| DFL Event Kind | GT Spells | `% < 2s` | `v7_bal` Miss@0.5 | `v8.5` Miss@0.5 | `v8.5` Missed Spells | Share of All `v8.5` Misses | Median Start Err (f) | P90 |Start Err| (f) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OtherBallAction` | 660 | `37.3%` | `34.2%` | **`33.8%`** | 223 | `24.3%` | `0.0` | `69.0` |
| `none` | 639 | `25.2%` | `29.4%` | **`29.4%`** | 188 | `20.5%` | `0.0` | `71.2` |
| `Play` | 400 | `25.0%` | `32.5%` | **`32.5%`** | 130 | `14.2%` | `0.0` | `136.6` |
| `TacklingGame` | 377 | `36.1%` | `36.6%` | **`34.5%`** | 130 | `14.2%` | `0.0` | `74.9` |
| `Play_high` | 366 | `29.5%` | `29.5%` | **`29.0%`** | 106 | `11.6%` | `0.0` | `65.4` |
| `ThrowIn` | 188 | `18.6%` | `17.5%` | **`18.6%`** | 35 | `3.8%` | `0.0` | `16.4` |
| `Delete` | 135 | `35.6%` | `32.6%` | **`32.6%`** | 44 | `4.8%` | `-1.0` | `82.9` |
| `FreeKick` | 117 | `13.7%` | `10.3%` | **`10.3%`** | 12 | `1.3%` | `0.0` | `12.0` |
| `GoalKick` | 65 | `0.0%` | `16.9%` | **`15.4%`** | 10 | `1.1%` | `0.0` | `42.2` |
| `ShotAtGoal` | 39 | `53.8%` | `35.9%` | **`30.8%`** | 12 | `1.3%` | `0.0` | `41.0` |
| `Foul` | 25 | `96.0%` | `28.0%` | **`32.0%`** | 8 | `0.9%` | `0.5` | `13.7` |
| `CornerKick` | 16 | `75.0%` | `25.0%` | **`18.8%`** | 3 | `0.3%` | `0.0` | `10.5` |
| `KickOff` | 10 | `0.0%` | `30.0%` | **`30.0%`** | 3 | `0.3%` | `0.0` | `1.0` |
| `PenaltyNotAwarded` | 6 | `50.0%` | `50.0%` | **`66.7%`** | 4 | `0.4%` | `-1.0` | `96.4` |
| `Offside` | 4 | `75.0%` | `25.0%` | **`25.0%`** | 1 | `0.1%` | `1.0` | `2.0` |
| `Penalty` | 3 | `100.0%` | `66.7%` | **`66.7%`** | 2 | `0.2%` | `0.0` | `0.0` |
| `ChanceWithoutShot` | 3 | `33.3%` | `66.7%` | **`66.7%`** | 2 | `0.2%` | `19.0` | `34.2` |
| `SpectacularPlay` | 2 | `50.0%` | `100.0%` | **`100.0%`** | 2 | `0.2%` | `-220.0` | `336.0` |
| `PossessionLossBeforeGoal` | 2 | `50.0%` | `50.0%` | **`50.0%`** | 1 | `0.1%` | `-1.0` | `2.8` |
| `RefereeBall` | 1 | `0.0%` | `0.0%` | **`0.0%`** | 0 | `0.0%` | `0.0` | `0.0` |
| `SitterPrevented` | 1 | `0.0%` | `0.0%` | **`0.0%`** | 0 | `0.0%` | `1.0` | `1.0` |
| `BallDeflection` | 1 | `0.0%` | `0.0%` | **`0.0%`** | 0 | `0.0%` | `-2.0` | `2.0` |
| `FairPlay` | 1 | `0.0%` | `0.0%` | **`0.0%`** | 0 | `0.0%` | `-2.0` | `2.0` |

### 4B. `J2-Raw` & `J2-LiveOnly` Metrica Event Categories (3 Metrica Matches)

| Metrica Event Macro Group | GT Spells | `% < 2s` | `J2-Raw` (`v7_bal`) Miss@0.5 | `J2-Raw` (`v8.5`) Miss@0.5 | `J2-LiveOnly` (`v8.5`) Miss@0.5 | `J2-LiveOnly` Missed | Share of `J2-LiveOnly` Misses | Median Start Err (f) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Recovery** | 597 | `25.5%` | `16.9%` | `16.9%` | **`15.9%`** | 95 | `43.8%` | `2.0` |
| **Set Piece (All)** | 227 | `16.3%` | `7.9%` | `7.5%` | **`12.3%`** | 28 | `12.9%` | `0.0` |
| **Challenge-Tackle** | 97 | `28.9%` | `33.0%` | `33.0%` | **`29.9%`** | 29 | `13.4%` | `4.0` |
| **Challenge-Aerial** | 68 | `29.4%` | `13.2%` | `13.2%` | **`14.7%`** | 10 | `4.6%` | `2.0` |
| **Challenge-Ground** | 64 | `31.2%` | `35.9%` | `35.9%` | **`34.4%`** | 22 | `10.1%` | `2.5` |
| **Ball Lost** | 45 | `40.0%` | `62.2%` | `62.2%` | **`57.8%`** | 26 | `12.0%` | `0.0` |
| **Other (PASS)** | 8 | `0.0%` | `25.0%` | `25.0%` | **`25.0%`** | 2 | `0.9%` | `0.0` |
| **Other (none)** | 5 | `20.0%` | `60.0%` | `60.0%` | **`60.0%`** | 3 | `1.4%` | `71.0` |
| **Other (SHOT)** | 3 | `66.7%` | `33.3%` | `33.3%` | **`33.3%`** | 1 | `0.5%` | `13.0` |
| **Other (BALL OUT)** | 1 | `100.0%` | `100.0%` | `100.0%` | **`100.0%`** | 1 | `0.5%` | `-446.0` |

### 4C. Stoppage Start Lag (`Live -> D` transitions on `J1A`, `25 fps`)

| Stoppage Cause | GT Stoppages | Median Lag (f) | Mean |Lag| (f) | Late `> 1.0s` (`> +25f`) | Early `< -1.0s` | Missed Stoppage |
|---|---:|---:|---:|---:|---:|---:|
| `OtherBallAction` | 186 | `+0.0f` | `7.8f` | **`4.8%`** | `5.9%` | `1.6%` |
| `Foul` | 99 | `+0.0f` | `3.0f` | **`2.0%`** | `3.0%` | `2.0%` |
| `none` | 96 | `+0.0f` | `13.4f` | **`15.6%`** | `8.3%` | `9.4%` |
| `TacklingGame` | 93 | `+0.0f` | `4.5f` | **`4.3%`** | `2.1%` | `2.1%` |
| `Play_high` | 57 | `+0.0f` | `5.3f` | **`5.3%`** | `1.8%` | `3.5%` |
| `Play` | 50 | `+0.0f` | `7.2f` | **`12.0%`** | `0.0%` | `6.0%` |
| `ShotAtGoal` | 48 | `+0.0f` | `5.7f` | **`2.1%`** | `4.2%` | `2.1%` |
| `Delete` | 42 | `+0.0f` | `4.4f` | **`4.8%`** | `0.0%` | `0.0%` |
| `Offside` | 10 | `+0.0f` | `3.8f` | **`10.0%`** | `0.0%` | `10.0%` |
| `FinalWhistle` | 4 | `+0.0f` | `2.3f` | **`25.0%`** | `0.0%` | `25.0%` |
| `FreeKick` | 4 | `+0.0f` | `1.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `FairPlay` | 3 | `+0.0f` | `0.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `Penalty` | 3 | `+0.0f` | `0.0f` | **`66.7%`** | `0.0%` | `66.7%` |
| `PenaltyNotAwarded` | 3 | `+0.0f` | `26.0f` | **`33.3%`** | `0.0%` | `0.0%` |
| `ThrowIn` | 3 | `-3.0f` | `3.0f` | **`33.3%`** | `0.0%` | `33.3%` |
| `BallDeflection` | 1 | `+0.0f` | `0.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `ChanceWithoutShot` | 1 | `+0.0f` | `0.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `CornerKick` | 1 | `+0.0f` | `0.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `GoalDisallowed` | 1 | `+5.0f` | `5.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `KickOff` | 1 | `+141.0f` | `141.0f` | **`100.0%`** | `0.0%` | `0.0%` |
| `Nutmeg` | 1 | `+68.0f` | `68.0f` | **`100.0%`** | `0.0%` | `0.0%` |
| `PlayerNotSentOff` | 1 | `+0.0f` | `0.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `SitterPrevented` | 1 | `+0.0f` | `0.0f` | **`0.0%`** | `0.0%` | `0.0%` |
| `VideoAssistantAction` | 1 | `N/A` | `N/A` | **`100.0%`** | `0.0%` | `100.0%` |

### 4D. Spurious Predicted Possessions & Domino Merges (`v8.5` vs `v7_balanced`)

| Judge / Model | Total Pred Spells | Spurious (`IoU < 0.5`) | Spurious (`IoU < 0.3`) | Domino Merges (`1 pred -> >=2 GT`) | Swallowed GT Spells |
|---|---:|---:|---:|---:|---:|
| `J1A` (`v7_balanced`) | 2575 | 443 (`17.2%`) | 254 (`9.9%`) | 215 | 453 |
| `J1A` (`v85_strict_cv`) | 2563 | 418 (`16.3%`) | 233 (`9.1%`) | 207 | 437 |
| `J2-LiveOnly` (`v7_balanced`) | 1114 | 213 (`19.1%`) | 183 (`16.4%`) | 53 | 109 |
| `J2-LiveOnly` (`v85_strict_cv`) | 1106 | 208 (`18.8%`) | 179 (`16.2%`) | 51 | 106 |

## 5. Full-Match Unsupervised Physical & Multi-Modal Consistency on `BAR-ATH` (`158,857` Graded Frames)

| Metric / Audit Check | `v7_6family_balanced` | `v7_6family_newton` | `v85_forensic_strict_cv` | `v85_reteam_count` (`D3`) |
|---|---:|---:|---:|---:|
| Graded `poss/min` (`158,857f`) | `2.38` | `2.41` | **`2.28`** | `2.29` |
| Private Unseen `poss/min` (`151,357f`) | `2.37` | `2.40` | **`2.27`** | `2.28` |
| Graded Live Spells Count | `252` | `255` | **`241`** | `242` |
| Graded `% < 2.0s` | `11.51%` | `11.76%` | **`7.88%`** | `8.68%` |
| Graded `% Dead (D)` | `48.35%` | `48.47%` | **`48.72%`** | `48.76%` |
| Graded `% Home of Live` (`H / (H+A)`) | `60.83%` | `60.92%` | **`61.07%`** | `61.48%` |
| Smart Tagging `in_phase` Live Team Agree (`75,771f`) | `80.09%` | `79.82%` | **`79.98%`** | `80.08%` |
| Smart Tagging `in_phase` Pred Dead Rate (`84,782f`) | `41.66%` | `41.80%` | **`42.07%`** | `42.11%` |
| Smart Tagging `SET PIECES` Pre-Kick `D` Agree | `86.85%` | `87.13%` | **`88.34%`** | `88.36%` |
| Smart Tagging `SET PIECES` Post-Kick Team Agree (When Alive) | `89.78%` | `89.78%` | **`89.78%`** | `82.47%` |
| Physical Kick Restart Median Lag (f) | `1.0f` | `1.0f` | **`1.0f`** | `1.0f` |
| Agreement vs `v85_strict_cv` (All `158,857f`) | `99.37%` | `99.63%` | **`100.00%`** | `99.55%` |
| Agreement vs `v85_strict_cv` (`spell_f1_mean`) | `0.9647` | `0.9694` | **`1.0000`** | `0.9814` |
| Agreement vs `D3` Re-Teamed `v8.5` (All `158,857f`) | `99.04%` | `99.19%` | **`99.55%`** | `100.00%` |
| Agreement vs `D3` Re-Teamed `v8.5` (`spell_f1_mean`) | `0.9474` | `0.9505` | **`0.9814`** | `1.0000` |
| Changed Within `±1s` of D3 Re-Team Stretch | `26.0%` | `26.5%` | **`0.0%`** | `86.2%` |
| Exact `H <-> A` Equivariance (`frame_mismatches`) | `0` (`True`) | `0` (`True`) | **`0` (`True`)** | `0` (`True`) |
