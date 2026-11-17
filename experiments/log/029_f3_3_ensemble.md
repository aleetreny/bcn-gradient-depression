# 029 · F3.3 Ensemble GBM + TCN con señal de cambio — LOMO anidado  (2026-10-24, commit 8aef6ff)

Comando: `.venv311/bin/python scripts/s3_3_ensemble_nested.py --trials 150`        Config: `configs/degrade_v3.json`        Semillas: TCN interiores 300 + fold

## Protocolo

Se completa el anidado para la TCN: para cada partido exterior T, una TCN entrenada con los 5 partidos que no son ni T ni J predice cada J interior (42 entrenamientos). Las probabilidades del GBM (S2.4b), la cabeza M3 (F3.1) y la TCN exterior (F3.2) ya estaban guardadas. Para cada variante, el decodificador (con β) se ajusta con Optuna sobre las predicciones interiores y T se puntúa una sola vez.
- **A:** probabilidades = media(GBM, TCN); cambio = M3.
- **B:** probabilidades = media(GBM, TCN); cambio = media(M3, cabeza de cambio de la TCN).

## Resultado

| Fold | v1.1 | v1.2 (M3) | A | **B** | posesiones B / verdad |
|---|---|---|---|---|---|
| J03WMX | 0,615 | 0,633 | 0,649 | **0,656** | 361 / 511 |
| J03WN1 | 0,631 | 0,640 | 0,677 | **0,665** | 388 / 500 |
| J03WOH | 0,699 | 0,710 | 0,774 | **0,772** | 328 / 402 |
| J03WOY | 0,647 | 0,653 | 0,695 | **0,680** | 379 / 456 |
| J03WPY | 0,701 | 0,713 | 0,711 | **0,724** | 340 / 421 |
| J03WQQ | 0,704 | 0,704 | 0,734 | **0,739** | 388 / 401 |
| J03WR9 | 0,680 | 0,697 | 0,727 | **0,733** | 333 / 370 |
| **media** | 0,668 | 0,679 | 0,710 ± 0,038 (6/7 > v1.2) | **0,710 ± 0,040 (7/7 > v1.2)** | |

**+0,031 frente a la v1.2 y +0,042 frente a la v1.1.** Parámetros típicos del decodificador: λ_HA 0,7–2,3, β 3,1–3,9, d_min de equipo 10–11 frames.

## Hallazgos

1. **El ensemble GBM + TCN es la mayor mejora desde el decodificador** (+0,031, 7/7 en la variante B). Cumple con holgura la regla de aceptación.
2. Las dos señales de cambio (solo M3, o media de M3 y TCN) empatan en media (0,710). B mejora en todos los folds; se elige B.
3. Sigue faltando segmentar: predice el 70–97 % de las posesiones reales.

## Decisión

**Aceptar el ensemble (variante B) → candidato v2.** Construir v2 para el BAR‑ATH (TCN entrenada en CPU, de forma determinista) y hacer el control público.

## Candidato v2 sobre el BAR‑ATH (commit b8dc471, tag `v2`)

`scripts/make_submission.py --name v2 --decoder-config configs/decoder_v2.json --transition --tcn --proba mean --change mean`:
- GBM determinista + cabeza M3 + TCN entrenada en CPU con algoritmos deterministas;
- decodificador final ajustado con las 7 predicciones exteriores (λ_HA 1,60 · β 5,91 · λ_LD 1,38 · λ_DL 31,5 · d_min 9 / 126);
- **reproducible:** dos construcciones de 58 min cada una, SHA‑256 `0d626fa4d58e823cf50756a3623e90bdeab994baa87256a0ef2ba788b69e8da5`.

| | v1.1 | v1.2 | **v2** |
|---|---|---|---|
| LOMO anidado | 0,668 | 0,679 | **0,710** |
| spellF1 públicos [IC] | 0,495 [0,358–0,611] | 0,533 [0,364–0,691] | 0,506 [0,329–0,682] |
| macroF1 · F1 de D públicos | 0,867 · 0,865 | 0,870 · 0,871 | 0,869 · 0,868 |
| posesiones en los 5 min (verdad 17) | 21 | 16 | **17** |
| BAR‑ATH: pos./min · % D · mediana · % < 2 s | 2,76 · 48,9 % · 6,4 s · 13 % | 2,58 · 49,0 % · 6,9 s · 11 % | 2,60 · 49,6 % · 6,4 s · 15 % |

**Control público superado:** queda dentro del ruido de la v1.2 (con 17 posesiones el IC es muy ancho). **v2 es el mejor candidato.**

## Cribado de variantes de la TCN (descartadas)

Ensemble como en B y con los parámetros de B de cada fold (`experiments/outputs/028/<variante>`):

| Variante | ensemble B | folds mejores que la base | TCN sola (decodificada) | log‑loss TCN |
|---|---|---|---|---|
| **base** (25 épocas, dilatación 128, peso de cambio 0,5) | **0,7098** | — | 0,661 | 0,321 |
| 50 épocas | 0,7031 | 2/7 | 0,647 | 0,433 (sobreajusta) |
| dilatación 256 (~41 s de contexto) | 0,7081 | 2/7 | 0,665 | 0,334 |
| peso de cambio 1,0 | 0,7037 | 3/7 | 0,657 | 0,315 |
| promedio de TCN base + d256 | 0,7117 | 6/7 | — | — |
| promedio de 2–4 TCN | 0,707–0,712 | ≤ 3/7 | — | — |

Ninguna variante supera la regla de +0,01. El promedio de varias TCN suma como mucho +0,002. **Se mantiene la TCN base.**
