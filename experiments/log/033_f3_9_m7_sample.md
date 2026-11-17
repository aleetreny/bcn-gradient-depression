# 033 · F3.9 M7: Metrica Sample como datos de entrenamiento de la cabeza de equipo — descartado  (2026-10-24, commit f518352)

Comando: `.venv311/bin/python scripts/s3_9_sample_training.py`        Código: `TwoHeadGBM.fit(..., team_only=...)`

## Qué se probó

Los 3 partidos de Metrica Sample (el mismo proveedor que el BAR‑ATH; etiquetas derivadas de eventos, S0.4) se añaden al entrenamiento del GBM v1ctl **solo para la cabeza de equipo**. En el Sample el balón falta exactamente en los parones, así que la cabeza vivo/parado se entrena solo con Sportec para no aprender ese atajo del proveedor. Es un cribado con el decodificador sin reajustar.

## Resultado

| Juez | sin Sample | con Sample (solo equipo) |
|---|---|---|
| Sportec, LOMO exterior (GBM decodificado) | 0,6870 | 0,6885 (+0,002, 5/7) |
| Metrica Sample, dejando un partido fuera (1 / 2 / 3) | 0,799 (0,812 / 0,749 / 0,837) | 0,801 (0,820 / 0,761 / 0,824) |
| públicos | 0,584 [0,443–0,714] | 0,507 [0,371–0,634] |

## Hallazgos

1. **No aporta:** +0,002 en Sportec y en el Sample, y −0,08 en los públicos (dentro del ruido, pero en contra).
2. Lo más probable es que las etiquetas del Sample, **derivadas por nuestras reglas** a partir de eventos, sean demasiado ruidosas o no sigan la semántica de la bandera de Metrica. Con 3 partidos no compensan.

## Decisión

**Descartar M7.** El Sample sigue siendo útil como **juez** de transferencia, no como datos de entrenamiento.
