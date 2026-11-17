# 035 · F3.12 Apilador temporal sobre GBM, TCN y M3 — descartado  (2026-10-24, commit 0848f2c)

Comando: `.venv311/bin/python scripts/s3_12_stacker.py`        Salida: `experiments/outputs/037/`

## Qué se probó

En vez de la media simple de GBM y TCN (v3), un **GBM de segundo nivel** que mira las seis señales base:
- P(vivo) y P(H|vivo) del GBM y de la TCN;
- cambio de M3 y cabeza de cambio de la TCN.

De cada señal usa su valor desplazado ±0,25/0,5/1/2 s, medias centradas de 1 y 3 s y la diferencia adelante − atrás de 0,5 y 2 s (78 features). Predice H/A/D y una señal de cambio propia.

**Anidado completo** con las predicciones interiores de F3.7, sin reentrenar los modelos base:
- para cada T, el apilado interior de J se entrena con las predicciones interiores de los otros cinco;
- el decodificador se ajusta sobre esos seis;
- T se puntúa una vez.

## Resultado (LOMO anidado, spellF1)

| Fold | ens. cambio medio (v3) | apilador + su cambio | apilador sin cambio |
|---|---|---|---|
| J03WMX | **0,660** | 0,655 | 0,654 |
| J03WN1 | 0,689 | **0,701** | 0,689 |
| J03WOH | 0,751 | **0,761** | 0,750 |
| J03WOY | **0,701** | 0,688 | 0,677 |
| J03WPY | 0,724 | **0,740** | 0,725 |
| J03WQQ | **0,743** | 0,718 | 0,722 |
| J03WR9 | 0,738 | **0,752** | 0,740 |
| **media** | 0,7152 | **0,7163** (4/7) | 0,7079 |

## Hallazgos

1. **Empate con la media simple:** +0,001 de media, 4/7 folds, con oscilaciones de ±0,02 por fold. Aprender la combinación no extrae más que promediar.
2. El número de posesiones predichas apenas cambia (331–391 frente a 370–511 reales): **la infra‑segmentación no se arregla combinando mejor las mismas señales**. Falta información nueva sobre los cambios, no una mejor mezcla.
3. Sin la señal de cambio, el apilador cae a 0,708: la cabeza de cambio sigue siendo útil para el decodificador.

## Decisión

**Descartar.** Más piezas sin ganancia. La media simple se queda.
