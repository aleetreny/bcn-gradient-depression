# 003 · S0.3 Lector Sportec  (2026-10-21, commit 3e19049)

Comando: `.venv311/bin/python scripts/report_s0_3_sportec.py`        Config: n/a        Semilla: 0 (orden de slots por partido)
Código: `src/data/sportec.py`, `src/data/common.py` (clase `Match` común) · tests: `tests/test_sportec.py` (4 verdes, XML sintético)

## Resultado

Los 7 partidos se cargan en ~15 s en paralelo (unos 3 s por partido, regex sobre el XML), con cache `.npz`. Cada frame tiene etiqueta (0 sin etiquetar).

| Partido | min | % D | posesiones | pos./min | media (s) | mediana (s) | % < 2 s | tramos D | D media (s) | % cambios directos H↔A | slots |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | 97,3 | 41,5 | 511 | 5,25 | 6,7 | 3,0 | 37,4 | 103 | 23,5 | 80,0 | 32 |
| J03WN1 | 94,4 | 51,7 | 500 | 5,30 | 5,5 | 3,1 | 34,2 | 107 | 27,4 | 79,0 | 29 |
| J03WOH | 91,5 | 47,6 | 402 | 4,39 | 7,2 | 3,9 | 27,4 | 102 | 25,6 | 75,1 | 32 |
| J03WOY | 95,0 | 46,5 | 456 | 4,80 | 6,7 | 3,6 | 28,1 | 113 | 23,5 | 75,6 | 32 |
| J03WPY | 97,5 | 44,3 | 421 | 4,32 | 7,7 | 3,8 | 29,5 | 97 | 26,7 | 77,1 | 31 |
| J03WQQ | 94,9 | 40,7 | 401 | 4,23 | 8,4 | 4,7 | 25,2 | 99 | 23,4 | 75,5 | 31 |
| J03WR9 | 97,9 | 41,9 | 370 | 3,78 | 9,2 | 4,2 | 25,4 | 92 | 26,7 | 75,3 | 31 |
| **Sportec (7)** | 668 | **44,9** | **3.061** | **4,58** | 7,2 | **3,7** | **30,0** | 713 | ~25 | ~77 | |
| **BAR‑ATH públicos** | 5 | **19,6** | **17** | **3,40** | 14,2 | **7,6** | **11,8** | 4 | 14,7 | 75,0 | |

Balón visible: 100 % en los 7 (Sportec es tracking limpio). Posesión = racha máxima de H o de A dentro de una parte, igual que el evaluador.

**Primeros 5 min de cada partido de Sportec** (mismo tramo que los públicos): D entre 22 % y 68 % (media 35 %), de 22 a 38 posesiones (media 30), mediana de 1,8 a 7,0 s.

**Comprobaciones:**
- **kloppy 3.18** (`sportec.load_tracking`, 1 de cada 25 frames): en la versión inicial solo se contrastó J03WMX (5.840 frames, etiquetas y x del balón). **Ampliado tras la revisión externa** a los **7 partidos** (40.113 frames): etiquetas H/A/D idénticas al 100 %, balón x/y y **875.129 posiciones de jugador** con una diferencia máxima de 3,7e‑6 m, y la misma visibilidad de jugadores frame a frame. El script falla si algo no cuadra.
- **Dirección:** el signo de ataque se deduce por parte (x media del equipo local frente al visitante). Tras normalizar, el portero local queda en x ≈ 15–23 m y el visitante en x ≈ 86–95 m en los 7 partidos.
- **Velocidad:** se recalcula desde las posiciones (m/s). El atributo `S` de DFL es esa misma velocidad en km/h (ratio 3,60 en los 7 partidos). Así el formato coincide con el ATD, cuyo canal resultó estar en m/s (S0.2).

Gráfico: `experiments/outputs/003/spell_durations.png` (distribución acumulada de la duración de las posesiones). Tabla: `per_match.csv`; resumen: `summary.json`.

## Hallazgos

1. *(Hipótesis; ver la revisión en HALLAZGOS.)* **Las posesiones de Sportec son más cortas y más numerosas** que las de la verdad pública: mediana de 3,7 s frente a 7,6 s, un 30 % por debajo de 2 s frente al 12 %, y 4,6 posesiones/min frente a 3,4. En el gráfico, la curva pública queda a la derecha de los 7 partidos. Con solo 17 posesiones públicas no es concluyente, pero va en la dirección del riesgo que anticipaba el plan: la bandera de Sportec reacciona antes que la de Metrica. **Consecuencia:** no copiar las duraciones de Sportec en el decodificador. S1.5 lo medirá con más cuidado.
2. **Sportec marca D el 45 % del tiempo** (41–52 %), lo normal en fútbol (el balón está en juego un 55–60 %). Los 5 min públicos tienen solo un 19,6 % de D, pero los primeros 5 min de Sportec van del 22 % al 68 %: **un tramo de 5 min no es representativo del % de D del partido**. Es probable que la verdad completa del BAR‑ATH tenga bastante más D que el 20 % público. Importa para calibrar el decodificador y el umbral de D (S1.5, S1.7).
3. Los tramos D de Sportec duran de media unos 25 s (públicos: 14,7 s, con 4 tramos). El % de cambios directos H↔A (sin D entre medias) coincide: ~77 % frente al 75 %.
4. Detalle técnico: en la 2.ª parte de DFL los suplentes solo tienen frames mientras juegan, así que un slot por persona con NaN fuera del campo reproduce bien la estructura. Los slots se barajan con semilla fija para que el índice no signifique nada.

## Decisión

Seguir → S0.4 Lector Metrica Sample. Para S1.5 queda medir si la diferencia de duraciones es semántica (cuándo cambia la bandera) o solo variación entre partidos, y comparar el % de D con Metrica Sample.
