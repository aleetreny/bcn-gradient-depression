# 030 · F3.4 M6 (D corto separador), desplazamiento de cambios y F3.5 transferencia a Metrica Sample  (2026-10-24)

## F3.4a Desplazamiento global de los cambios de equipo (hipótesis de S1.2)

Se adelantaron δ frames todos los cambios H↔A predichos por la v2 (positivo = antes).

| δ | −25 | −12 | −6 | **0** | +6 | +12 | +18 | +25 | +37 |
|---|---|---|---|---|---|---|---|---|---|
| públicos | 0,475 | 0,482 | 0,494 | **0,506** | 0,506 | 0,506 | 0,518 | 0,518 | 0,494 |
| Sportec (anidado) | 0,555 | 0,646 | 0,694 | **0,710** | 0,696 | 0,655 | 0,609 | 0,559 | 0,495 |

En Sportec el óptimo es claramente 0. En los públicos, adelantar 0,7–1 s da +0,012 con solo 12 cambios: ruido. **No se aplica**; la hipótesis de S1.2 sigue sin demostrar.

## F3.4b M6: D corto como separador (`src/decode/postprocess.py`, `scripts/s3_4_short_dead.py`)

Donde la señal de cambio tiene un pico (≥ umbral) pero el decodificador mantiene el mismo equipo en ±12 frames, se inserta un D de L frames. El umbral (0,3–0,8) y L (2–20) se eligen sobre las predicciones interiores de cada fold.

**Resultado:** en los 7 folds, **cualquier combinación empeora las predicciones interiores** (−0,003 a −0,008), así que nunca se aplica (Δ = 0 en exterior). **Descartado.** Donde la señal tiene un pico sin cambio decodificado suele haber un duelo, no dos posesiones: partir una posesión larga correcta rompe su emparejamiento con la verdad.

## Oráculos sobre la v2 (qué queda)

| Arreglo | Sportec (anidado) | Δ | públicos | Δ |
|---|---|---|---|---|
| v2 | 0,710 | — | 0,506 | — |
| posesiones cortas recuperadas | 0,825 | **+0,115** | 0,663 | **+0,157** |
| … solo en frames en juego | 0,814 | +0,104 | 0,663 | +0,157 |
| momento perfecto | 0,765 | +0,055 | 0,577 | +0,071 |
| sin posesiones inventadas | 0,745 | +0,035 | 0,586 | +0,080 |
| D perfecto | 0,722 | +0,012 | 0,533 | +0,027 |

## F3.5 Transferencia a Metrica Sample (`scripts/s3_5_sample_transfer.py`)

Se entrenan los modelos completos una vez (7 partidos de Sportec con `degrade_v3`; guardados en `local_data/models/full_degrade_v3/`) y se evalúa cada candidato con su propio decodificador en los 3 partidos de Metrica Sample. Es el mismo proveedor que el BAR‑ATH, con unas 1.100 posesiones. *Reservas:* las etiquetas son derivadas de eventos por nuestras reglas, y en el Sample el balón falta cuando está parado, así que D es más fácil.

| Sistema | spellF1 medio | partido 1 / 2 / 3 | macroF1 | acierto de equipo | F1 de D | posesiones pred. / verdad |
|---|---|---|---|---|---|---|
| baseline | 0,423 | 0,444 / 0,420 / 0,403 | 0,802 | 0,826 | 0,813 | 497 / 372 |
| v1.1 | 0,770 | 0,782 / 0,730 / 0,798 | 0,909 | 0,869 | 0,907 | 300 / 372 |
| v1.2 (M3) | 0,782 | 0,791 / 0,748 / 0,808 | 0,910 | 0,870 | 0,907 | 311 / 372 |
| **v2** | **0,784** | 0,781 / 0,747 / 0,826 | 0,912 | 0,874 | 0,913 | 317 / 372 |

## Hallazgos

1. **El orden de los candidatos se mantiene en un segundo dominio** (v2 ≥ v1.2 > v1.1 ≫ baseline): las mejoras no son solo un efecto de Sportec.
2. **M3 se transfiere** (+0,012 en el Sample, +0,011 en el LOMO). **El ensemble con la TCN, apenas:** +0,031 en el LOMO de Sportec frente a +0,002 en el Sample (+0,018 en el partido 3, −0,010 en el 1). La TCN parece más ligada al dominio de Sportec. Se mantiene en la v2 porque no empeora, pero su peso real en el BAR‑ATH es incierto.
3. **La infra‑segmentación es común a los tres dominios** (se predice el 80–85 % de las posesiones) y las posesiones cortas siguen siendo el mayor margen (+0,10–0,16). Hay que mejorar la **certeza del modelo** en los cambios: con la incertidumbre actual, el decodificador une posesiones a propósito, porque eso maximiza spellF1.
4. Desplazar los cambios y M6 se descartan.

## Decisión

Siguiente: **features de control más finas** para la cabeza de equipo y de cambio:
- quién se mueve con el balón (conducción);
- a quién va dirigido un pase (receptor probable, según la trayectoria del balón).

La idea es dar certeza en los duelos y en los robos cortos. Se cribará con un LOMO exterior del GBM antes de cualquier anidado. Metrica Sample queda como **segundo juez de transferencia** en el tablero.
