# 016 · S2.2 Features v1  (2026-10-23, commit bddd2da; reejecutado con la degradación v2, ee08b7b)

Comando: `.venv311/bin/python scripts/s2_2_features.py`        Config: `configs/degrade_v1.json` (degradación v2, semilla 0)        Semilla: 0
Código: `src/features/v1.py` · tests: `tests/test_features_v1.py` (3 verdes: **invariantes al orden de los slots**, **no leen las etiquetas**, máscaras del balón correctas)

## Resultado

**160 features** por frame (unos 8 s por partido, cacheadas en `local_data/cache/features_v1/`). No usan el tercer canal del balón, ni la vibración, ni Smart Tagging.

| Grupo | n | % NaN en BAR‑ATH | % NaN en Sportec limpio | % NaN en Sportec degradado | Diferencia mediana con el ATD (KS), limpio → degradado |
|---|---|---|---|---|---|
| balón (posición, visible, frames desde/hasta verlo, última/próxima posición, fuera del campo, velocidad) | 19 | 15,2 | 0 | 16,7 | 0,107 → **0,068** |
| balón‑equipos (distancias, jugadores a 2/5/10 m) | 12 | 30,6 | 0 | 33,4 | 0,058 → **0,029** |
| reglas M1 (último toque r 1/2/3 m, zona de control, más cercano, frames desde/hasta el último toque) | 11 | 0 | 0 | 0 | 0,083 → 0,083 |
| colectivas (centroides, dispersión, líneas, anchura, velocidades, área, portero) | 34 | 3,6 | 0 | 3,4 | 0,066 → **0,059** |
| contexto temporal (medias móviles ±0,4…8 s y ±20/40/80 s; diferencias hacia delante y hacia atrás) | 83 | 9,0 | 0 | 10,1 | 0,088 → **0,054** |
| estructura (minutos desde el inicio de la parte) | 1 | 0 | 0 | 0 | 0,093 |

La degradación acerca al ATD **128 de 160 features**; solo 4 se alejan más de 0,02 (distancia del portero al balón y jugadores de A a 2–5 m del balón).

**Las que peor se transfieren** (KS entre ATD y Sportec degradado > 0,2): posición del portero (H 0,45, A 0,27), salto entre la última y la próxima posición conocida del balón (0,32), dispersión en x promediada en 20–80 s (0,23–0,31) y visibles promediados en 80 s.
**Las que mejor:** jugadores a < 2/10 m del balón, jugadores en las áreas, posición y del balón, balón fuera del campo y diferencia de distancias a la última posición del balón (KS ≤ 0,02).

Tablas: `experiments/outputs/016/nan_by_group.csv`, `nan_by_feature.csv`, `domain_shift_ks.csv`.

## Hallazgos

1. **Puerta superada:** tests verdes (invariancia al orden de los slots, sin fugas de etiqueta, máscaras), y los NaN del Sportec degradado cuadran con el BAR‑ATH en todos los grupos.
2. **S2.2 encontró un fallo de la degradación v1:** el portero se mezclaba con los jugadores de campo, y la posición del portero, la dispersión y el hueco entre centroides se alejaban del ATD después de degradar (KS hasta 0,76). Corregido en la degradación v2 (ver informe 015). Es exactamente para lo que sirve esta tabla.
3. **Lo que sigue sin transferirse es estilo de juego:** el portero del Barça juega más adelantado y la anchura de los equipos difiere. Conviene vigilar si el modelo se apoya en esas features (importancias en S2.3) y, si lo hace, probar a quitarlas.
4. El salto entre la última y la próxima posición conocida del balón se transfiere mal (0,32): los huecos simulados no reproducen del todo dónde reaparece el balón.

## Decisión

Seguir → **S2.3** con las 160 features. Si en S2.3 las features de estilo (portero, dispersión larga) aparecen entre las más importantes, se probará una versión sin ellas.
