# 011 · S1.5 Semántica Sportec frente a Metrica  (2026-10-23, commit e4f4373)

Comando: `.venv311/bin/python scripts/s1_5_flag_semantics.py`        Config: ventanas de 5 min cada 30 s; eventos DFL alineados con un desfase estimado por partido        Semilla: n/a

## Resultado

### 1. ¿Es raro el tramo público? (ventanas de 5 min dentro de las partes)

| | ventanas | posesiones p10/p50/p90 | % ventanas con ≤ 17 posesiones | mediana de posesión p10/p50/p90 | % ventanas con mediana ≤ 7,56 s | % D p10/p50/p90 | % ventanas con D ≤ 19,6 % |
|---|---|---|---|---|---|---|---|
| Sportec | 1.205 | 15 / 23 / 31 | 20 % | 2,4 / 3,9 / 6,5 s | 94 % | 23 / 46 / 65 | **5 %** |
| Metrica Sample (derivadas) | 510 | 14 / 19 / 27 | 37 % | 3,1 / 4,8 / 9,5 s | 77 % | 16 / 36 / 52 | 15 % |

**Solo ventanas con poco D (≤ 30 %):**

| | ventanas | % con ≤ 17 posesiones | % con mediana ≥ 7,56 s | correlación entre % D y mediana de posesión |
|---|---|---|---|---|
| Sportec | 273 | 21 % | **14 %** | −0,35 |
| Metrica Sample | 178 | 29 % | **35 %** | −0,33 |

Primeros 5 min de cada parte en Sportec: de 22 a 38 posesiones y del 20 % al 68 % de D.

### 2. ¿Cuándo cambia la bandera de Sportec respecto a los eventos?

Las horas de los eventos DFL **no están alineadas al frame**. El desfase que minimiza la distancia pasador–balón en los pases cambia mucho de un partido a otro: −16, −44, +16, +32, +4, +12 y +8 frames. Incluso corregido, la distancia mediana se queda en 1,8–3,1 m.

Con ese desfase corregido, en los 2.351 cambios de equipo (eventos en ±3 s):

| Evento | n | p25 | mediana | p75 |
|---|---|---|---|---|
| evento con balón del **equipo nuevo más cercano** al cambio − cambio de bandera *(antes rotulado "primer evento", por error)* | 1.934 | −11 | **+5** | +25 |
| **primer** evento del equipo nuevo en la ventana ±3 s − cambio de bandera *(añadido tras la revisión)* | 1.934 | — | **−6** | +20 |
| último pase del **equipo antiguo** − cambio de bandera | 1.060 | −54 | **−35** | −12 |

El 32 % de los cambios tiene un evento del equipo nuevo a ±0,5 s. Solo el 4,5 % tiene el último pase del equipo antiguo en el medio segundo previo.

### 3. Bordes de D frente al balón

| | n | resultado |
|---|---|---|
| Sportec: inicios de D con balón fuera del campo a ±3 s | 710 | 68 %. El balón sale **4 frames antes** del D (p25 −6, p75 −2) |
| Sportec: el balón empieza a moverse (> 3 m/s) − fin de D | 695 | **−1 frame** (p25 y p75 −1): la bandera vuelve a "juego" en el instante del saque |
| Públicos BAR‑ATH: inicios de D | 4 | solo 1 con el balón fuera (−2 frames); en 3 el balón sigue visible y dentro del campo |
| Públicos BAR‑ATH: fines de D, movimiento del balón − reanudación | 4 | −9, −75, −11 y −31 frames: el balón ya se mueve antes de que la verdad pase a juego |

Gráfico: `experiments/outputs/011/flag_semantics.png`. Tablas: `sliding_windows.csv`, `sportec_changes_vs_events.csv`, `summary.json`.

## Hallazgos

1. *(Matizado abajo.)* **Se retira la afirmación de S0.3 de que la bandera de Sportec sea "más rápida"**, pero sin demostrar lo contrario. 17 posesiones en 5 min es normal (percentil 20 en Sportec, 37 en el Sample). La mediana de 7,6 s parecía rara (percentil 94 en Sportec), pero **las ventanas con pocos parones tienen posesiones más largas** (correlación −0,35). Entre las ventanas con poco D, una de cada 7 en Sportec y una de cada 3 en el Sample tiene posesiones tan largas. **El tramo público es un tramo tranquilo**, no una prueba de semántica distinta.
2. **El 19,6 % de D público sí es raro:** percentil 5 en Sportec y 15 en el Sample. Lo esperable para el partido completo es bastante más D (medianas del 36–46 %). **No hay que calibrar la proporción de D con los 5 min públicos.**
3. *(Corregido abajo: se medía el evento más cercano, no el primero.)* **La bandera de Sportec cambia cuando el equipo nuevo toma el balón, no en el pase fallado:** el primer evento del equipo nuevo está a +5 frames de mediana y el último pase del antiguo a −35 (1,4 s antes). Coincide con S1.2 (último toque, 92 %). *Matiz:* los eventos DFL tienen desfases de hasta ±1,8 s según el partido, así que esto es orientativo. **Nunca usar las horas de los eventos DFL para medir tiempos finos.**
4. **Los bordes de D de Sportec son físicos y precisos:** el D empieza 4 frames después de que el balón salga, y el juego vuelve en el frame del saque (−1). Un 32 % de los D empieza sin balón fuera del campo: faltas, fueras de juego, etc.
5. **En el BAR‑ATH la verdad parece reanudar el juego algo después de que el balón se mueva** (3 casos medibles: 9, 11 y 31 frames; el cuarto ya se movía). *Hipótesis débil (n = 3).* En el ATD el balón también se mueve cuando lo colocan para el saque. Si se confirma, el decodificador necesitará un pequeño desplazamiento en los fines de D, igual que el de los cambios de equipo (S1.2).

## Decisión

**Seguir.** Sportec sirve para (a) **los bordes de D**, que son físicos y precisos (con la reserva de que Metrica podría reanudar algo más tarde), y (b) **el momento del cambio de equipo**, que sigue al control como en Metrica según lo poco que se ve.
- ~~**No se recalibran las duraciones** del decodificador por proveedor: la diferencia aparente era el tramo.~~ *(Corregido abajo: las dos explicaciones siguen abiertas.)*
- **Sí se fija que la proporción de D no se calibra con los públicos.**

Siguiente: S1.6.

## Corrección tras la revisión de Codex 002 (2026-10-23)

Revisión: [codex_review_002](../reviews/codex_review_002.md), puntos 2 y 6.

- **Primero frente a más cercano:** el script tomaba el evento del equipo nuevo **más próximo** al cambio (`argmin |Δ|`) y lo llamaba "primero". Ya está renombrado (`new_nearest_action`) y se añade el primero de la ventana (`new_first_action_in_window`). El más cercano está a +5 frames y el primero a −6. Ninguno de los dos prueba qué evento pertenece a la posesión. Con los desfases de los eventos DFL (±1,8 s), **lo único que se sostiene** es que la bandera queda cerca de eventos del equipo nuevo y lejos (1,4 s) del último pase del antiguo. No es una prueba de "primera toma de control".
- **Proveedores:** los percentiles muestran que el tramo público es **compatible** con la variación entre tramos de Sportec. **No descartan** una diferencia entre proveedores: son partidos distintos, las etiquetas del Sample son derivadas, las ventanas se solapan (no son 1.205 observaciones independientes) y el corte D ≤ 30 % no es causal. **Las dos explicaciones siguen abiertas.** Las duraciones del decodificador se decidirán por transferencia (LOMO y comprobación por partido), sin suponer ni igualdad ni diferencia.
- **D:** igual que no se calibra la proporción de D con los públicos, tampoco se impone al BAR‑ATH el 36–46 % de otras fuentes.
