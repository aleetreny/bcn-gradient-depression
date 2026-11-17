# 005 · S0.5 Evaluador  (2026-10-22, commit 785a034)

Comando: `.venv311/bin/python scripts/report_s0_5_evaluator.py`        Config: n/a        Semilla: 0 (bootstrap oficial)
Código: `src/eval/` (`metrics.py`, `errors.py`, `plots.py`, `registry.py`) · tests: `tests/test_eval.py` (5 verdes)

## Resultado

**(a) Reproducción oficial.** `src.eval` no reimplementa nada: llama a `evaluation/possession_metrics.py`. Las 4 líneas de `scoreboard.csv`, reconstruidas con la propia función del runner, son **idénticas como texto en las 54 columnas**.

| Baseline | spellF1 | IC 5–95 % (bootstrap 1000) | macroF1 | posesiones (verdad 17) |
|---|---|---|---|---|
| smooth125 | 0,4242 | 0,267 – 0,594 | 0,5812 | 16 |
| smooth50 | **0,3256** | 0,214 – 0,447 | 0,6207 | 26 |
| smooth25 | 0,2868 | 0,196 – 0,385 | 0,6200 | 36 |
| raw | 0,0378 | 0,019 – 0,060 | 0,5972 | 131 |

Los intervalos de smooth125, smooth50 y smooth25 se solapan: con 17 posesiones públicas no se puede ordenar entre ellos, como ya advierte el plan.

**(b) LOMO.** `lomo(matches, fit_predict)` entrena con 6 partidos y puntúa el 7.º con las métricas oficiales. Las partes van seguidas en Sportec, así que se inserta un frame X entre ellas (test: un tramo que cruza el descanso cuenta como 2 posesiones, igual que en el BAR‑ATH). Prueba de cordura con un **oráculo retrasado k frames** (la verdad desplazada):

| Retraso | spellF1 LOMO | std | macroF1 | peor / mejor fold |
|---|---|---|---|---|
| 0 | 1,000 | 0,000 | 1,000 | 1,00 / 1,00 |
| 6 frames (0,24 s) | 0,906 | 0,019 | 0,975 | 0,88 / 0,93 |
| 12 (0,48 s) | 0,775 | 0,035 | 0,950 | 0,71 / 0,82 |
| 25 (1 s) | **0,557** | 0,043 | 0,900 | 0,49 / 0,61 |
| 50 (2 s) | 0,356 | 0,040 | 0,822 | 0,30 / 0,41 |

**(c) Descomposición de errores.** smooth50 sobre los 5 min públicos:
- **Posesiones:** de las 17 de la verdad empareja 12 (IoU ≥ 0,3). Pierde las 2 cortas (< 2 s) y 3 largas.
- **Posesiones inventadas:** 14 de sus 26 (5 parpadeos de menos de 2 s y 9 largas).
- **Cambios de equipo:** de 12, acierta solo 5 en ±3 s, se le escapan 7 y predice 16 que no existen.
- **Reanudaciones tras D:** el baseline las da 61 frames antes (2,4 s); de 4 se le escapan 2.
- **Tramos D:** se le escapan 3 de los 4 de la verdad e inventa 2.

En el oráculo 1 s tarde (Sportec J03WMX), los 408 cambios salen a +25 exactos (el detector funciona) y **pierde 180 de las 191 posesiones cortas**.

**(d)** Línea de tiempo: `experiments/outputs/005/timeline_public.png` (verdad, smooth50, smooth125 y balón visible). **(e)** `src/eval/registry.py` añade filas a `experiments/registry.csv`.

Tablas: `baselines_reproduced.csv`, `lomo_lagged_oracle.csv`, `errors_smooth50_public.csv`, `errors_lag25_sportec_J03WMX.csv`, `summary.json`.

## Hallazgos

1. **Puerta superada:** los scores oficiales se reproducen al carácter.
2. **En partidos completos, llegar tarde cuesta más de lo estimado:** 1 s de retraso uniforme baja el spellF1 de 1,0 a 0,56 (−44 %) y 0,5 s lo baja a 0,78 (−22 %). El doc 04 estimaba un −22–35 % por 1 s con los 5 min públicos. La razón es que en un partido entero hay muchas posesiones cortas (un 30 % por debajo de 2 s en Sportec), y **1 s de retraso destruye casi todas las posesiones de menos de 2 s** (180 de 191). Clavar el momento del cambio es la palanca principal.
3. El macroF1 apenas lo ve: con 1 s de retraso sigue en 0,90. Confirma que hay que decidir por spellF1.
4. El baseline adelanta las reanudaciones unos 2,4 s y no ve la mayoría de los tramos D. En la línea de tiempo, el D largo de los segundos ~635–656 tiene el balón visible y el baseline lo predice como juego. Encaja con el hallazgo de S0.4: en el BAR‑ATH hay D con balón visible.

## Decisión

Seguir → S0.6 Baseline reimplementado (idéntico frame a frame y primera fila LOMO de Sportec).
