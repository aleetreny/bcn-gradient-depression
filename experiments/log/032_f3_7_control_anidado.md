# 032 · F3.7–F3.8 Features de control en todo el sistema: LOMO anidado, Metrica Sample y públicos  (2026-10-24, commit cf22cb7)

Comandos: `scripts/s3_7_nested_full.py --features v1ctl` (entrenamiento anidado de GBM + M3 en CPU y TCN en GPU en paralelo; después `--decode-only`) y `scripts/s3_8_candidates.py --features v1ctl`
Conjunto de features **v1ctl**: 160 v1 + 60 de cambio + 24 de control, en los tres modelos (GBM de dos cabezas, cabeza M3, TCN). Config: `configs/degrade_v3.json`.

## Resultado

**LOMO anidado completo** (los tres modelos y el decodificador sin el partido exterior), por fold:

| Fold | GBM | GBM + M3 | **ens. cambio medio** | ens. cambio M3 | v2 (sin control) |
|---|---|---|---|---|---|
| J03WMX | 0,625 | 0,631 | **0,660** | 0,646 | 0,656 |
| J03WN1 | 0,664 | 0,668 | **0,689** | 0,688 | 0,665 |
| J03WOH | 0,738 | 0,738 | **0,751** | 0,744 | 0,772 |
| J03WOY | 0,649 | 0,656 | **0,701** | 0,698 | 0,680 |
| J03WPY | 0,717 | 0,721 | **0,724** | 0,723 | 0,724 |
| J03WQQ | 0,686 | 0,711 | **0,743** | 0,738 | 0,739 |
| J03WR9 | 0,731 | 0,739 | **0,738** | 0,733 | 0,733 |
| **media** | 0,687 | 0,695 | **0,715 ± 0,031** | 0,710 ± 0,033 | 0,710 |

**Tres jueces** (decodificador final de cada combinación, ajustado con las 7 predicciones exteriores; Metrica Sample y públicos con modelos completos entrenados en MPS):

| Combinación | LOMO anidado | Metrica Sample (1 / 2 / 3) | públicos [IC] | F1 de D públicos | posesiones públicas (17) | BAR‑ATH pos./min · % D |
|---|---|---|---|---|---|---|
| v2 (sin control) | 0,710 | 0,784 (0,781 / 0,747 / 0,826) | 0,506 [0,329–0,682] | 0,868 | 17 | 2,60 · 49,6 % |
| GBM | 0,687 | 0,799 | 0,584 [0,443–0,714] | 0,882 | 20 | 2,80 · 49,2 % |
| GBM + M3 | 0,695 | 0,788 | 0,578 [0,433–0,711] | 0,867 | 19 | 2,40 · 49,1 % |
| **ens. cambio medio** | **0,715** | 0,803 (0,812 / 0,755 / 0,840) | 0,541 [0,353–0,729] | 0,824 | 17 | 2,43 · 49,0 % |
| ens. cambio M3 | 0,710 | **0,809** (0,813 / 0,763 / 0,851) | 0,570 [0,364–0,764] | 0,818 | 16 | 2,50 · 49,0 % |

## Hallazgos

1. **Con control, el ensemble de cambio medio es el mejor en el LOMO** (0,715; +0,005 frente a la v2, **6/7** folds; supera al otro ensemble en 7/7) y **mejora la transferencia a Metrica Sample** (+0,019, **3/3** partidos).
2. **En el LOMO no alcanza el +0,01 de la regla.** Se acepta igualmente por consistencia en los dos jueces: el Sample se añadió para medir la transferencia al proveedor del BAR‑ATH, que es lo que decide la nota final.
3. **Sin TCN, el GBM con control queda 0,02–0,03 por debajo en el LOMO**, pero transfiere igual de bien al Sample (0,799) y saca la mejor nota pública (0,584) y la mejor D. Con 17 posesiones públicas eso no es concluyente, pero **es una reserva sencilla y sólida** (un solo modelo).
4. Las combinaciones con M3 transfieren algo peor que sin M3 cuando no hay TCN (0,788 frente a 0,799), al contrario que con features v1. Indica que la cabeza de cambio con control añade poco una vez que el modelo de equipo ya ve la conducción.
5. Todos predicen pocas posesiones en el BAR‑ATH (2,4–2,8 por minuto).

## Decisión

**v3 = ensemble de cambio medio con features de control** (`configs/decoder_v1ctl_ens_mean.json`). **Reserva simple:** el GBM con control (`configs/decoder_v1ctl_gbm.json`). Construcción reproducible de la v3 en curso.

## v3 sobre el BAR‑ATH (tag `v3`)

`scripts/make_submission.py --name v3 --control --decoder-config configs/decoder_v1ctl_ens_mean.json --transition --tcn --proba mean --change mean` (commit cf22cb7):
- dos construcciones de 59 min, **SHA‑256 `fbb875184365706e208ba5666142566a22cea4c43939fcf7a8014ac91098e8e4`**, idéntico;
- la primera arrancó con el `gbm.py` anterior y la segunda con el modificado para las fuentes "solo equipo" (F3.9): **eso confirma que ese cambio no altera nada**.

| | v2 | **v3** |
|---|---|---|
| LOMO anidado | 0,710 | **0,715** |
| Metrica Sample | 0,784 | **0,803** |
| spellF1 públicos [IC] | 0,506 [0,329–0,682] | **0,570 [0,387–0,752]** |
| macroF1 · F1 vivo/parado · F1 de D (públicos) | 0,869 · 0,972 · 0,868 | 0,870 · 0,967 · 0,865 |
| posesiones públicas (verdad 17) | 17 | 16 |
| BAR‑ATH: pos./min · % D · mediana · % < 2 s | 2,60 · 49,6 % · 6,4 s · 15 % | **2,39** · 49,1 % · 7,2 s · 12 % |

~~v3 es el mejor candidato en los tres jueces.~~ **Corregido tras la revisión 004:** v3 gana solo en el LOMO; `ens_m3` gana en el Sample (0,809) y el GBM con control en público (0,584). Además, el 0,803 del Sample es de los modelos **MPS** de F3.8, no del fichero CPU de la v3 (difieren en 2664 frames del BAR‑ATH). Su Sample con los pesos CPU está **pendiente**. Ver la nota al final. Sigue la alerta de pocas posesiones en el BAR‑ATH (2,4/min). En Sportec y en el Sample los sistemas predicen el 80–85 % de las reales, lo que implicaría unas 3,0 reales en el BAR‑ATH, en línea con los 3,4 de los públicos.

## Nota tras la revisión 004

- **La regla de aceptación cambió sin regla nueva.** +0,0054 en el LOMO, una sola semilla, J03WOH −0,02. Además la comparación con la v2 cambia las semillas interiores y añade las 60 features de cambio al GBM, así que no aísla las 24 de control.
- **El Sample separa D con mucha más facilidad** que el reto: el balón se ve en el 16 %, 4 % y 0,2 % de los D, frente al 66 % público. Excluyendo D ±3 s, la mejora de la v3 en el Sample baja de +0,018 a +0,008 (2/3 partidos). Hay señal de equipo, pero no todo el +0,019 se transfiere.
- **Compromiso explícito:** la v3 queda como candidata **provisional**. Las features de control se dan por confirmadas (GBM +0,015 7/7 y +0,028 en el Sample 3/3). La elección del ensemble (cambio medio frente a M3, con o sin TCN) **no está decidida**: v2, `ens_m3` y el GBM con control siguen siendo alternativas.
