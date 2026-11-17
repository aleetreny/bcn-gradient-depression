# 027 · F3.1 M3: cabeza de transición + coste de cambio dependiente del tiempo  (2026-10-24, commit 2059831; candidato v1.2)

Comandos: `scripts/s3_1_transition_head.py --trials 150` (LOMO anidado), `scripts/s3_1b_decoder_final.py --save configs/decoder_m3.json --change trans` y `scripts/make_submission.py --name v1_2_m3 --decoder-config configs/decoder_m3.json --transition` (dos veces)
Config: `configs/degrade_v3.json`, `configs/decoder_m3.json`        Código: `src/models/transition.py`, `src/decode/viterbi_hsmm.py` (β_HA)

## Qué es

- **Cabeza de transición:** un LightGBM que estima P(cambio de equipo a ±6 frames). Usa las 160 features v1 más 60 diferencias "lo que viene menos lo que hubo" (0,4, 1 y 2 s) de las señales de equipo; 3,5 % de positivos.
- **Decodificador:** el coste del cambio H↔A en t pasa a ser max(0, λ_HA − β·logit P_cambio(t)). Con β = 0 es idéntico al anterior (test).
- **Evaluación:** con el mismo LOMO anidado que la v1.1. Las probabilidades de equipo y vivo/parado se reutilizan; la cabeza de transición se entrena en el mismo esquema interior y exterior.

## Resultado (LOMO anidado, `degrade_v3`; emparejado con la v1.1)

| Fold | **M3** | v1.1 | AUC de la cabeza | AP (tasa base) | cambios perdidos M3 / v1.1 | cortas perdidas M3 / v1.1 | posesiones pred. / verdad | λ_HA | β_HA | d_min equipo / D |
|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | **0,633** | 0,615 | 0,940 | 0,51 (0,035) | 180 / 181 | 122 / 124 | 348 / 511 | 0,60 | 1,84 | 15 / 9 |
| J03WN1 | **0,640** | 0,631 | 0,955 | 0,54 (0,035) | 151 / 166 | 100 / 115 | 389 / 500 | 0,58 | 2,37 | 9 / 142 |
| J03WOH | **0,710** | 0,699 | 0,964 | 0,62 (0,028) | 106 / 112 | 63 / 69 | 312 / 402 | 0,87 | 3,14 | 15 / 106 |
| J03WOY | **0,653** | 0,647 | 0,952 | 0,55 (0,031) | 122 / 125 | 76 / 79 | 378 / 456 | 0,75 | 1,43 | 15 / 40 |
| J03WPY | **0,713** | 0,701 | 0,966 | 0,58 (0,028) | 108 / 106 | 73 / 76 | 339 / 421 | 0,72 | 3,99 | 16 / 48 |
| J03WQQ | 0,704 | 0,704 | 0,945 | 0,47 (0,027) | 89 / 91 | 52 / 53 | 363 / 401 | 0,74 | 1,96 | 11 / 131 |
| J03WR9 | **0,697** | 0,680 | 0,968 | 0,59 (0,024) | 94 / 95 | 54 / 58 | 312 / 370 | 0,55 | 5,31 | 17 / 11 |
| **media** | **0,679 ± 0,033** | 0,668 | 0,956 | 0,55 | | | | | 2,37 (mediana) | |

**+0,011 de media y 7/7 folds** (en J03WQQ la mejora es solo de 0,0002). La regla de aceptación pedía ≥ +0,01 y ≥ 5/7: **cumple, por poco.** La feature más importante de la cabeza es el cambio del último toque en 0,4 s (25 % de la ganancia), seguida de los frames desde y hasta el último toque de cada equipo.

**Candidato v1.2 sobre el BAR‑ATH** (decodificador final: λ_HA 0,58 · β 2,41 · λ_LD 38,6 · λ_DL 9,3 · d_min 17 / 9; reproducible, SHA‑256 `98989a3ff16cdd1660902a92d6d5c3dc3619f1ef3b2130b757a31d106d8b14bb`):

| | v1.1 | **v1.2** |
|---|---|---|
| spellF1 públicos [IC] | 0,495 [0,358–0,611] | **0,533 [0,364–0,691]** |
| macroF1 · F1 de D públicos | 0,867 · 0,865 | 0,870 · 0,871 |
| posesiones en los 5 min (verdad 17) | 21 | 16 |
| BAR‑ATH: pos./min · % D · mediana · % < 2 s | 2,76 · 48,9 % · 6,4 s · 13 % | **2,58** · 49,0 % · 6,9 s · 11 % |

## Hallazgos

1. **M3 se acepta:** +0,011 en el LOMO anidado (7/7) y los públicos no empeoran (0,495 → 0,533, dentro del ruido). **Candidato v1.2.**
2. **La cabeza de transición es buena** (AUC 0,96, AP 0,55 con un 3 % de positivos), pero **se traduce en poco**: recupera solo unos cuantos cambios y robos cortos por partido. El decodificador sigue uniendo posesiones (en Sportec, el 68–90 % de las reales).
3. **Alerta de plausibilidad:** en el BAR‑ATH completo cada versión predice menos posesiones (v1 3,1 → v1.1 2,8 → v1.2 **2,6** por minuto), por debajo de todas las referencias (3,4 / 3,9 / 4,6). En el partido completo, **la infra‑segmentación es probablemente la mayor fuente de error que queda**: cada posesión no predicha es un falso negativo en spellF1.

## Decisión

**Aceptar M3 → v1.2.** Siguiente: atacar la infra‑segmentación directamente (el decodificador y el umbral de cambio) además de la TCN (F3.2), y medir el número de posesiones como objetivo explícito, no solo el spellF1.
