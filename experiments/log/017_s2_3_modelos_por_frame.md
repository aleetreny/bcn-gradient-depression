# 017 · S2.3 Modelos por frame (dos cabezas LightGBM)  (2026-10-23, commit 563da50)

Comando: `.venv311/bin/python scripts/s2_3_frame_models.py --config configs/degrade_v2.json`        Config: `configs/degrade_v2.json` (degradación v3)        Semillas: degradación de entrenamiento 1 y 2 + variante con menos ocultación del balón quieto (0,5); prueba con la semilla 0; modelo semilla = fold
Código: `src/models/gbm.py` (`TwoHeadGBM`: P(vivo) y P(H|vivo), calibración isotónica con predicciones fuera de fold agrupadas por partido)

## Resultado

**Primera ejecución descartada por un atajo.** El AUC vivo/parado salía 1,000 porque Sportec congela el balón en los parones (ver informe 015, degradación v3). Esa ejecución se conserva en `experiments/outputs/017/v1_atajo/` (LOMO 0,678 inflado) y **no cuenta**. Todo lo siguiente es con la degradación v3.

**LOMO en Sportec degradado (7 folds)** — sin decodificador: media móvil de las probabilidades ± w frames y argmax.

| Fold | AUC vivo | acierto equipo (verdad en juego) | acierto por frame | baseline (misma degradación) | w 0 | **w 12** | w 25 | w 50 | w 100 | limpio (w 50) |
|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | 0,998 | 0,904 | 0,924 | 0,288 | 0,434 | **0,605** | 0,571 | 0,502 | 0,418 | 0,543 |
| J03WN1 | 0,997 | 0,896 | 0,919 | 0,283 | 0,392 | **0,613** | 0,593 | 0,526 | 0,384 | 0,571 |
| J03WOH | 0,997 | 0,918 | 0,931 | 0,306 | 0,468 | **0,676** | 0,672 | 0,595 | 0,438 | 0,625 |
| J03WOY | 0,996 | 0,888 | 0,913 | 0,275 | 0,398 | **0,584** | 0,574 | 0,534 | 0,425 | 0,602 |
| J03WPY | 0,999 | 0,920 | 0,942 | 0,317 | 0,429 | **0,636** | 0,643 | 0,588 | 0,493 | 0,634 |
| J03WQQ | 0,997 | 0,915 | 0,929 | 0,330 | 0,428 | **0,672** | 0,671 | 0,620 | 0,507 | 0,661 |
| J03WR9 | 0,998 | 0,927 | 0,929 | 0,296 | 0,416 | **0,616** | 0,612 | 0,590 | 0,470 | 0,638 |
| **media** | 0,997 | 0,910 | 0,927 | **0,299** | 0,424 | **0,629** | 0,619 | 0,565 | 0,448 | 0,611 |

spellF1 LOMO **0,629 ± 0,032** (w 12), **7/7 folds** mejor que el baseline (+0,33 de media). macroF1 0,91–0,93. Posesiones predichas con w 25: 343 de media, frente a 437 en la verdad.

**Calibración:** P(H|vivo) casi sobre la diagonal; P(vivo) bien en los extremos y ruidosa en la zona media, donde hay pocos frames (`reliability.png`).

**Features más importantes:**
- **vivo:** velocidad del balón promediada en 0,8 s y 2 s (48 % de la ganancia), fracción de jugadores casi parados en 20 s, velocidad media de A en 8 s, balón fuera del campo, salto entre la última y la próxima posición del balón;
- **equipo:** **último toque r 1 m (39 %)**, último toque r 2 m promediado en 2 s, frames desde el último toque de cada equipo y anchura de H.

**Modelo con los 7 partidos → BAR‑ATH** (w 12; `submissions/s2_3_gbm_w12.csv`, fuera de git):

| | baseline | **GBM S2.3** |
|---|---|---|
| spellF1 públicos [IC 5–95 %] | 0,326 [0,214–0,447] | **0,400 [0,296–0,496]** |
| macroF1 públicos | 0,621 | **0,857** |
| F1 vivo/parado públicos | 0,899 | **0,963** |
| AUC P(vivo) públicos | — | 0,953 |
| acierto de equipo (verdad en juego) públicos | 0,899 | 0,900 (AUC 0,975) |
| posesiones en los 5 min (verdad 17) | 26 | 33 |
| plausibilidad BAR‑ATH: pos./min · % D · mediana · % < 2 s | 5,61 · 32,3 % · 4,6 s · 31 % | **3,80 · 47,3 % · 4,1 s · 30 %** |

## Hallazgos

1. **Primer sistema claramente mejor que el baseline** en el juez principal: LOMO degradado 0,629 frente a 0,299 (7/7 folds). En los públicos también mejora (macroF1 0,62 → 0,86; vivo/parado 0,90 → 0,96). El spellF1 público (0,400) cae dentro del ruido del baseline.
2. **El LOMO sobreestima la detección de D:** AUC vivo 0,997 en Sportec degradado frente a **0,953 en el BAR‑ATH real**. Sportec degradado sigue siendo algo más fácil para D. En cambio, la cabeza de equipo se transfiere casi igual (0,91 frente a 0,90). **Toda mejora de D se comprobará también con el AUC y el F1 vivo/parado públicos** (nuevas columnas del tablero).
3. **Esta S2.3 encontró un atajo grave en los datos de Sportec** (balón congelado en los parones) gracias a que un AUC perfecto se trató como alarma y no como éxito. Corregido en la degradación v3.
4. **El último toque de S1.2 es la feature central del equipo** (39 % de la ganancia), lo que confirma la decisión M1.
5. **Sin decodificador, el sistema parpadea:** 33 posesiones en los 5 min públicos frente a 17, y en Sportec cae de w 12 a w 100 en cuanto se suaviza mucho. Es justo el hueco que debe llenar el decodificador (S2.4). En el BAR‑ATH completo predice 47 % de D, como Sportec (45 %), y 3,8 posesiones/min: plausible.
6. La ventana w 12 se eligió por LOMO entre 5 valores; w 12 y w 25 casi empatan (0,629 y 0,619).

## Decisión

**Seguir → S2.4** (decodificador semi‑Markov ajustado a spellF1 con Optuna) sobre estas probabilidades. El GBM S2.3 entra en el tablero como **candidato 1**.
