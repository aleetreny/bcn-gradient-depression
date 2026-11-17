# 047 · F4.8 Autoentrenamiento en el proveedor del reto  (2026-10-24, commit 97cf6e6)

Comando: `.venv311/bin/python scripts/s4_8_self_training.py`        Salida: `experiments/outputs/046/`

## Qué se probó

Adaptar el modelo al tracking de Metrica **sin respuestas**. Los 3 partidos de Metrica Sample hacen de BAR‑ATH: sus etiquetas solo puntúan.
1. El GBM v1ctl, entrenado en los 7 Sportec, se decodifica en los partidos.
2. Se toman como pseudo‑etiquetas los frames con P(equipo predicho | vivo) ≥ τ, lejos de cualquier cambio predicho.
3. Se reentrena con Sportec más esos partidos y se decodifica otra vez.

## Resultado (spellF1 en Metrica Sample)

| variante | Sample | sin D ±3 s | % frames pseudo‑etiquetados | acierto de las pseudo‑etiquetas |
|---|---|---|---|---|
| sin adaptar | 0,7995 | 0,8414 | — | — |
| solo equipo, τ 0,9, 1 ronda | 0,7972 | 0,8405 | 44 % | 0,992 |
| solo equipo, τ 0,9, 2 rondas | 0,7937 | 0,8388 | 46 % | 0,990 |
| solo equipo, τ 0,95 | 0,7994 | 0,8448 | 39 % | 0,994 |
| solo equipo, τ 0,9, peso ×3 | 0,7949 | 0,8394 | 44 % | 0,992 |
| solo equipo, τ 0,8, margen 50 | 0,7937 | 0,8428 | 39 % | 0,991 |
| **equipo + parado, τ 0,9, 1 ronda** | **0,8109** | **0,8499** | 80 % | 0,929 |
| equipo + parado, 2 rondas | 0,8066 | 0,8467 | 82 % | 0,928 |

Variante equipo + parado, por partido: 0,812 → **0,826**, 0,749 → **0,762**, 0,837 → **0,844** (3/3). Sin D: +0,007, +0,012, +0,007 (3/3).

## Hallazgos

1. **Adaptar solo el equipo no sirve:** las pseudo‑etiquetas seguras (99 % de acierto) son justo las fáciles y no enseñan nada nuevo.
2. **Adaptar también el balón parado sí mejora** (+0,011, 3/3) y la mejora se mantiene sin los bordes de D (+0,008, 3/3). El modelo aprende cómo se ve una parada en este tracking concreto.
   - *Reserva:* en el Sample el balón desaparece en las paradas, una pista que el BAR‑ATH no tiene igual. Parte de la ganancia puede ser de esa peculiaridad.
3. Una segunda ronda ya empeora: basta con una.

## Decisión

**Prometedor, pendiente de un segundo juez.** Autoentrenamiento transductivo sobre cada partido de Sportec escondido (`scripts/s4_12_self_training_lomo.py`), incluido al principio de la cadena nocturna. Si también mejora allí, se añade a la construcción como una ronda de adaptación sobre el propio BAR‑ATH completo, sin etiquetas.
