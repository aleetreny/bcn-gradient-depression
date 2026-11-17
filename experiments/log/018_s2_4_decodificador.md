# 018 · S2.4 Decodificador semi‑Markov  (2026-10-23, commit a423260)

Comando: `.venv311/bin/python scripts/s2_4_decoder.py --trials 120`        Entrada: probabilidades fuera de fold de S2.3 (`local_data/oof/s2_3_d3_degrade_v2/`)        Semillas: Optuna TPE = índice del fold; estudio final 99
Código: `src/decode/viterbi_hsmm.py` (Viterbi con cadenas de subestados, numba, 0,04 s por partido) · tests: `tests/test_decoder.py` (4 verdes)

## Qué hace

Elige la secuencia H/A/D que maximiza la suma de log‑probabilidades por frame menos un **coste por cada cambio** (H↔A: λ_HA; juego→D: λ_LD; D→juego: λ_DL), con **duración mínima** por clase (d_min de equipo, d_min de D). Cada parte se decodifica por separado. 5 parámetros, ajustados en Sportec (no en los públicos).

**Protocolo (LOMO anidado):** para cada partido de prueba, Optuna (120 pruebas) maximiza el spellF1 medio en los otros 6. Los parámetros son el **centro de la meseta** (mediana del 10 % mejor de las pruebas), no la mejor prueba suelta.

## Resultado

| Fold | spellF1 entrenamiento (mejor) | **prueba, centro de meseta** | prueba, mejor prueba | prueba, suavizado w 12 (S2.3) | macroF1 | posesiones pred. / verdad | λ_HA | λ_LD | λ_DL | d_min equipo | d_min D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | 0,677 | **0,630** | 0,630 | 0,605 | 0,920 | 378 / 511 | 2,19 | 29,5 | 34,3 | 4 | 57 |
| J03WN1 | 0,671 | **0,663** | 0,664 | 0,613 | 0,902 | 401 / 500 | 2,20 | 36,5 | 12,9 | 2 | 58 |
| J03WOH | 0,662 | **0,702** | 0,702 | 0,676 | 0,920 | 349 / 402 | 2,17 | 35,8 | 1,1 | 10 | 120 |
| J03WOY | 0,677 | **0,622** | 0,623 | 0,584 | 0,900 | 383 / 456 | 2,20 | 23,7 | 7,9 | 6 | 98 |
| J03WPY | 0,666 | **0,682** | 0,682 | 0,636 | 0,931 | 318 / 421 | 2,79 | 26,2 | 0,6 | 14 | 106 |
| J03WQQ | 0,664 | **0,700** | 0,699 | 0,672 | 0,925 | 384 / 401 | 2,11 | 1,0 | 28,1 | 8 | 146 |
| J03WR9 | 0,670 | **0,663** | 0,665 | 0,616 | 0,922 | 319 / 370 | 2,14 | 30,4 | 1,0 | 8 | 66 |
| **media** | | **0,666 ± 0,029** | | 0,629 | 0,917 | | | | | | |

**7/7 folds mejoran** respecto al suavizado de S2.3 (+0,037 de media). La mejor prueba suelta y el centro de meseta dan lo mismo en prueba.

**Parámetros finales** (estudio con los 7 partidos, centro de meseta): λ_HA 2,16 · λ_LD 28,6 · λ_DL 3,85 · d_min equipo 10 frames (0,4 s) · d_min D 106 frames (4,2 s). spellF1 medio en los 7: 0,669.

**Robustez ±20 %** en cada parámetro: spellF1 entre 0,666 y 0,669 → **meseta**. En el gráfico (`plateau.png`): λ_HA tiene un óptimo claro en torno a 2; d_min de equipo es plano hasta ~20 frames; d_min de D, entre 60 y 145; λ_LD, plano a partir de ~10; λ_DL, entre 2 y 10.

**BAR‑ATH** (probabilidades del modelo S2.3 entrenado con los 7 partidos, más la restricción dura de D en la 2.ª parte antes del saque, frames 110.808–113.740; `submissions/s2_4_gbm_hsmm.csv`, fuera de git):

| | baseline | GBM + suavizado (S2.3) | **GBM + decodificador (S2.4)** |
|---|---|---|---|
| spellF1 públicos [IC 5–95 %] | 0,326 [0,214–0,447] | 0,400 [0,296–0,496] | **0,519 [0,368–0,660]** |
| macroF1 públicos | 0,621 | 0,857 | **0,865** |
| F1 vivo/parado públicos | 0,899 | 0,963 | **0,968** |
| posesiones en los 5 min (verdad 17) | 26 | 33 | **20** |
| plausibilidad: pos./min · % D · mediana · % < 2 s | 5,61 · 32 % · 4,6 s · 31 % | 3,80 · 47 % · 4,1 s · 30 % | **3,02 · 48 % · 5,2 s · 19 %** |

## Hallazgos

1. **El decodificador vale +0,037 en LOMO (7/7) y es robusto:** está en una meseta, no en un pico. En los públicos también mejora (0,400 → 0,519; 20 posesiones frente a 17 reales).
2. **Lo identificable son λ_HA (~2) y que D dure varios segundos.** λ_LD y λ_DL cambian mucho de un fold a otro sin efecto: la duración mínima de D (~4 s) hace el trabajo. Simplificación posible para el futuro: un único coste para entrar y salir de D.
3. **El decodificador tiende a unir posesiones:** en Sportec predice menos posesiones que la verdad (318–401 frente a 370–511). Sigue faltando detectar cambios, sobre todo los cortos, como anticipaba S1.7.
4. **d_min de D ≈ 4 s impide predecir D cortos.** Eso choca con la idea M6 (D corto como separador). M6 deberá probarse como post‑proceso aparte.
5. En el BAR‑ATH completo: 3,0 posesiones/min (públicos 3,4; Metrica Sample 3,9; Sportec 4,6) y 48 % de D. Plausible, quizá algo corto en posesiones; se revisará en S2.5.

## Decisión

**Seguir.** El GBM + decodificador es el **candidato 2** y el mejor hasta ahora. Siguiente: **S2.5**, generar v1 sobre el BAR‑ATH, validarla, panel de plausibilidad frente a Sportec y revisión visual de 3 ventanas de 2 minutos al azar, y congelarla como red de seguridad (tag `v1`).
