# 021 · S2.6 Descomposición de errores de la v1  (2026-10-23, commit 46a6d33)

Comando: `.venv311/bin/python scripts/s2_6_v1_errors.py`        Entrada: v1 fuera de fold en Sportec (probabilidades de S2.3 + `configs/decoder_v1.json`) y `submissions/v1.csv` en los públicos

## Resultado

**Oráculos sobre la v1** (un tipo de error arreglado con la verdad cada vez; intervenciones, no techos):

| Arreglo | Sportec | Δ | folds que mejoran | públicos | Δ |
|---|---|---|---|---|---|
| v1 | 0,669 | — | — | 0,519 | — |
| equipo perfecto en frames en juego *(paraguas)* | 0,950 | +0,281 | 7/7 | 0,944 | +0,426 |
| **posesiones cortas recuperadas** | 0,797 | **+0,129** | 7/7 | 0,615 | +0,096 |
| posesiones cortas recuperadas, solo en frames en juego | 0,788 | **+0,119** | 7/7 | 0,615 | +0,096 |
| **momento de los cambios perfecto (±3 s)** | 0,733 | **+0,065** | 7/7 | 0,627 | +0,108 |
| sin posesiones inventadas | 0,712 | +0,044 | 7/7 | 0,643 | +0,124 |
| **D perfecto** | 0,683 | **+0,014** | 7/7 | 0,533 | +0,014 |
| D perfecto + momento perfecto | 0,741 | +0,073 | 7/7 | 0,644 | +0,126 |

**Errores de la v1 en Sportec** (7 partidos, 3.061 posesiones reales):
- **Posesiones:** empareja 2.199 (IoU ≥ 0,3). Se escapan 533 cortas y 329 largas.
- **Posesiones inventadas:** 388 de las 2.544 predichas (210 parpadeos y 178 largas).
- **Cambios de equipo:** de 2.351 reales acierta 1.528 en ±3 s, se le escapan **823 (35 %)** y añade 305 que no existen.
- **Desfase de los acertados:** p10 −15, p25 −3, **mediana 0**, p75 +3 y **p90 +32** frames. Más de 0,5 s tarde: 282; más de 0,5 s pronto: 170.
- **D:** se escapan 91 de 697 reanudaciones; de 710 tramos D reales (≥ 1 s), 32 salen como juego, y de los 709 predichos, 42 son inventados.

**Errores de la v1 en los públicos:**
- **Posesiones:** empareja 13 de 17; se escapan las 2 cortas.
- **Cambios de equipo:** acierta 7 de 12, que llegan **+13 frames tarde** de mediana (p10 −25, p90 +41), y se escapan 5.
- **Reanudaciones:** 4/4, ligeramente pronto (mediana −9 frames).
- **D:** 4/4 sin errores.

Tablas: `experiments/outputs/021/oracles_v1.csv`, `errors_v1_sportec.csv`, `errors_v1_public.csv`.

## Hallazgos

1. *(Matizado abajo.)* **D ya no es el cuello de botella:** D perfecto solo daría +0,014, en Sportec y en los públicos (4/4 tramos D acertados). *Matiz:* el LOMO sobreestima D (S2.3), y los apagones del BAR‑ATH completo no se pueden comprobar (S2.5).
2. **Todo lo que queda está en el equipo:** se escapan el 35 % de los cambios reales y la mayoría de las posesiones cortas. Recuperar solo las cortas vale +0,12–0,13 en Sportec (7/7) y +0,10 en los públicos.
3. **Cuando acierta un cambio, lo clava** (mediana 0 frames en Sportec), pero hay una cola de retrasos (p90 +32 frames). En los públicos llega tarde (+13 frames de mediana, n = 7), lo que da algo de apoyo al desplazamiento temporal candidato de S1.2, con muy pocos casos.
4. Las posesiones inventadas pesan poco en Sportec (+0,04) y más en los públicos (+0,12, ruidoso con 17 posesiones).

## Decisión — orden de la Fase 3 (actualizado con la v1; *convertido en provisional tras la revisión 003, ver abajo*)

1. **Detectar los cambios de equipo y las posesiones cortas:**
   - cabeza de transición, P(cambio en t ± 3 frames), con coste del decodificador dependiente del tiempo (**M3**);
   - features de toque más finas y modelos temporales (TCN o U‑Net, **M9**);
   - D corto como separador (**M6**).
2. **Momento de los cambios:** M3, más el desplazamiento temporal candidato de S1.2, comprobado en los públicos con la regla de los ≤ 3 parámetros.
3. **Posesiones inventadas:** costes del decodificador dependientes del tiempo (M3).
4. **D:** solo el caso de los apagones (prior del decodificador o vídeo).
5. **Balón que falta** (M2b/M2c/M13): después, si el diagnóstico lo pide.

**Fase 2 terminada:** v1 congelada como red de seguridad. Siguiente: Fase 3, tras revisión.

## Corrección tras la revisión de Codex 003 (2026-10-23)

Revisión: [codex_review_003](../reviews/codex_review_003.md), puntos 3 y 4.

- **D no está resuelto.** "4/4 sin errores" solo significa que ningún tramo D sale mayoritariamente como juego. En los públicos la v1 **pierde 372 de 1.469 frames de D (25 %, unos 15 s)**: **F1 de la clase D 0,847** (precisión 0,979, recall 0,747). El 0,968 del tablero es el `alive_dead_f1` oficial, que toma **juego** como clase positiva. Desde ahora el tablero muestra también el F1 de D. Comparación: baseline 0,323 y GBM + suavizado 0,829.
- **El balón no queda relegado.** S2.6 no mide ninguna intervención de recuperación del balón sobre la v1. En S1.7 el balón valía +0,10 a +0,19 según el modelo.
- **El diagnóstico usa un decodificador ajustado con los mismos 7 partidos** (base 0,669, frente a 0,666 por fold). Codex lo repitió con los parámetros de cada fold y obtuvo cortas +0,122, momento +0,065 y D +0,015: el orden de magnitud se mantiene. S2.6 se repetirá con las predicciones por fold del LOMO anidado (`--nested`).

**Prioridades revisadas (provisionales):**
1. Cambios de equipo y posesiones cortas: primera apuesta, bien motivada.
2. D (fronteras, no solo apagones) y recuperación del balón: **abiertas**, a medir con el simulador corregido (degradación v4) y con el modelo actual.
3. El momento puro de los cambios, junto con el desplazamiento temporal candidato.

Nada se descarta todavía.
