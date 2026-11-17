# 006 · S0.6 Baseline reimplementado  (2026-10-22, commit 638c74a)

Comando: `.venv311/bin/python scripts/report_s0_6_baseline.py`        Config: ventana ±50 (y 0/25/125)        Semilla: n/a
Código: `src/models/baseline.py` · tests: `tests/test_baseline.py` (2 verdes)

## Resultado

**BAR‑ATH:** las 4 variantes, escritas con el escritor oficial, son **byte a byte idénticas** a `evaluation/results/baseline_{raw,smooth25,smooth50,smooth125}.csv` (0 frames distintos de 199.319). Público smooth50: spellF1 0,3256 [0,214 – 0,447], macroF1 0,6207.

**Sportec limpio, LOMO (7 folds) — fila de referencia:**

| Ventana | spellF1 LOMO | std | macroF1 | pos./min pred. (verdad 4,58) | acierto de equipo (verdad en juego) | F1 vivo/parado |
|---|---|---|---|---|---|---|
| raw | 0,098 | 0,011 | 0,364 | 21,6 | 0,774 | 0,710 |
| smooth25 | 0,296 | 0,019 | 0,386 | 6,04 | 0,821 | 0,710 |
| **smooth50** | **0,308** | 0,026 | 0,392 | 3,97 | 0,835 | 0,710 |
| smooth125 | 0,218 | 0,021 | 0,387 | 2,30 | 0,825 | 0,710 |

smooth50 por fold: J03WMX 0,307 · J03WN1 0,284 · J03WOH 0,301 · J03WOY 0,272 · J03WPY 0,317 · J03WQQ 0,361 · J03WR9 0,315.

**Metrica Sample (etiquetas derivadas, media de 3 partidos):** raw 0,105 · smooth25 0,380 · **smooth50 0,423** · smooth125 0,302.

**Descomposición de errores de smooth50, Sportec agregado** (3.061 posesiones de verdad):
- **Posesiones:** empareja 1.277 (IoU ≥ 0,3). Se le escapan 855 cortas y 929 largas.
- **Posesiones inventadas:** 298 parpadeos y 1.088 largas.
- **Cambios de equipo (2.351 en la verdad):** acierta 1.323 en ±3 s; se le escapan 1.028 y predice 1.309 que no existen.
- **Desfase:** de los acertados, 722 llegan más de 0,5 s antes y 252 más de 0,5 s tarde.
- **D:** los 710 tramos D de la verdad (≥ 1 s) se escapan todos. En Sportec el balón siempre se ve, así que el baseline nunca predice D.

**Desfase de los cambios de equipo frente a la bandera de Sportec** (frames; negativo = antes):

| Predicción | n | p10 | p25 | mediana | p75 | p90 |
|---|---|---|---|---|---|---|
| más cercano sin suavizar | 2.280 | −30 | −15 | **−5** | 0 | 15 |
| smooth50 | 1.323 | −55 | −37 | **−16** | 4 | 30 |

Tablas: `lomo_sportec_baseline.csv`, `lomo_sportec_smooth50_per_fold.csv`, `metrica_sample_baseline.csv`, `errors_smooth50_sportec.csv`, `summary.json`.

## Hallazgos

1. **Puerta superada:** idéntico frame a frame y byte a byte.
2. **Referencia LOMO Sportec limpio: spellF1 0,308 ± 0,026** con smooth50, la mejor ventana también aquí. Es parecida a los 0,3256 públicos y por encima de los 0,2433 publicados para el partido completo del BAR‑ATH (el docstring del starter).
3. **Sin D, el techo es bajo:** con un 45 % de D en Sportec, el baseline no acierta ni un tramo D. Un detector de D (Fase 2) debería dar un salto grande.
4. **El jugador más cercano cambia antes que la bandera** de Sportec (mediana −0,2 s) y el suavizado lo adelanta más (−0,64 s). Pista directa para S1.2 y S1.5: la bandera parece cambiar al **controlar** el balón, no al acercarse a él. En S1.2 hay que medir qué regla imita mejor la bandera de Metrica.
5. En Metrica Sample el baseline puntúa más (0,423) porque allí "sin balón" coincide con D (S0.4). No es comparable con nuestro caso.
6. Detalle técnico: la regla `models/` del `.gitignore` ignoraba también `src/models/`. Ahora se aplica solo a la raíz (`/models/`).

## Decisión

Seguir. **Fase 0 terminada** (S0.1–S0.6 con sus puertas superadas). Siguiente: Fase 1, S1.1 Anatomía de los cambios reales, tras revisión.
