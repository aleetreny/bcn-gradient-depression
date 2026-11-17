# 044 · F4.5 Simulador v5: balón dibujado a rectas — descartado  (2026-10-24, commit 2279124)

Comando: `.venv311/bin/python scripts/s4_5_interp_screen.py`        Salida: `experiments/outputs/043/`
Código: `src/data/ball_interp.py`, componente `interp` de `src/data/degrade.py`, `src/features/ballq.py`, `configs/degrade_v5.json`

## Qué se probó

Siguiendo el log 042, el simulador v5 reproduce en Sportec el balón del ATD dibujado a rectas:
- se alternan tramos reales y tramos rectos, con las longitudes medidas en el partido completo sin etiquetas;
- una recta solo se acepta mientras no se aleje más de 12 m del balón real; el límite se calibró sin etiquetas.

Resultado del calibrado: rectas 0,80 frente a 0,75 en el ATD; balón lento sin nadie a < 3 m 0,27 frente a 0,24; golpeo fantasma 0,17 frente a 0,28. La variante `v1ctlq` añade 7 pistas de calidad del balón: si es recta, la longitud y posición dentro de la recta, y el tiempo desde y hasta un tramo real.

## Resultado (GBM, decodificador sin reajustar)

| | v3 + v1ctl (actual) | v5 + v1ctl | v5 + v1ctlq |
|---|---|---|---|
| Metrica Sample (1 / 2 / 3) | **0,7995** (0,812 / 0,749 / 0,837) | 0,769 (0,797 / 0,708 / 0,801) | 0,764 (0,788 / 0,707 / 0,796) |
| Sample sin D ±3 s | **0,841** | 0,804 | 0,798 |
| LOMO exterior en Sportec* | 0,687 | 0,585 | 0,589 |

\* No es comparable: en v5 el partido de prueba también lleva el balón a rectas, que es más difícil.

## Hallazgos

1. **Entrenar con un balón a rectas empeora la transferencia** al proveedor del reto, en los 3 partidos de Metrica (−0,03 de media, −0,04 sin D). Las pistas de calidad del balón no lo compensan.
2. **Por qué** (*hipótesis*, coherente con los datos): el balón de Metrica Sample es un 90 % rectas y, aun así, el modelo actual transfiere bien (0,80), y allí casi no hay golpeos fantasma (1 %). Las uniones de las rectas reales caen **en los toques**, porque el tracker une detecciones donde el balón cambia de dirección, así que conservan la información de quién toca. Nuestras uniones aleatorias la destruyen.
3. Por tanto, **lo que separa al ATD no es la recta en sí, sino la incoherencia balón–jugador** (golpeos fantasma: 25–28 % en el ATD frente al 1 % del Sample). Eso apunta a balón mal situado o a jugadores sin trackear junto al balón, no a la interpolación.

## Decisión

**Descartar v5** (`degrade_v5.json` queda en el repo como experimento). Si se retoma, el siguiente paso lógico sería simplificar la trayectoria del balón conservando los cambios de dirección, o simular la ausencia de jugadores junto al balón. Ninguno de los dos tiene ahora prioridad sobre las ideas fuera de la caja.
