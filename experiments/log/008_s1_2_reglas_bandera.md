# 008 · S1.2 ¿Qué regla imita la bandera?  (2026-10-23, commit 815f6a6)

Comando: `.venv311/bin/python scripts/s1_2_flag_rules.py`        Config: parrilla en el script (77 combinaciones)        Semilla: n/a
Código: `src/features/possession_rules.py` · tests: `tests/test_possession_rules.py` (2 verdes)

## Resultado

Tres reglas, con un máximo de 3 parámetros cada una. El equipo se arrastra durante los huecos de balón y ninguna regla dice nada de D.
- **Más cercano + histéresis (k):** el equipo solo cambia si el nuevo más cercano dura k frames con el balón visible.
- **Último toque (r, θ, δ):** hay toque si el balón gira más de θ° o cambia de velocidad más de δ m/s (en ±3 frames) con el más cercano a menos de r m.
- **Zona de control (r, m, exclusiva):** un jugador del equipo está a menos de r m durante m frames; si es exclusiva, sin rivales a menos de r m.

Métricas sobre frames con verdad H o A y balón visible:
- **acuerdo:** % de frames en que la regla da el equipo de la verdad;
- **desfase:** cambios de equipo de la verdad emparejados (±3 s) con cambios de la regla al mismo equipo (> 0 = la regla llega tarde);
- **extra:** cambios de la regla en juego que no corresponden a ningún cambio real.

**Mejor ajuste de cada regla en los 5 min públicos (12 cambios de equipo):**

| Regla | Parámetros | Acuerdo | Emparejados | \|desfase\| mediano | desfase mediano | extra por cambio | Sportec con los mismos parámetros: acuerdo / desfase |
|---|---|---|---|---|---|---|---|
| Zona de control | r 2 m, m 3, exclusiva | **0,849** | 11/12 | 22 | +18 | 1,75 | 0,839 / −3 |
| Más cercano + histéresis | k 25 | 0,840 | 8/12 | 32,5 | +20 | 1,50 | 0,851 / +8 |
| Último toque | r 3 m, θ 20°, δ 3 | 0,839 | 9/12 | **16** | +15 | **0,67** | 0,897 / −2 |

**Mejor ajuste de cada regla en Sportec (7 partidos, 2.351 cambios):**

| Regla | Parámetros | Acuerdo | Emparejados | \|desfase\| mediano | desfase mediano | extra por cambio | Público con los mismos parámetros: acuerdo / \|desfase\| |
|---|---|---|---|---|---|---|---|
| **Último toque** | r 1 m, θ 20°, δ 3 | **0,922** | 1.974 | **2** | −2 | 0,45 | 0,726 / 36 |
| Zona de control | r 1 m, m 3, exclusiva | 0,877 | 1.692 | 4 | 0 | 0,40 | 0,813 / 20 |
| Más cercano + histéresis | k 25 | 0,851 | 1.398 | 21 | +8 | 0,36 | 0,840 / 32,5 |

**Sensibilidad del último toque al radio** (θ 20°, δ 3) — público: r 1 m 0,726 (3 cambios emparejados) · 1,5 m 0,760 (3) · 2 m 0,825 (7) · 3 m 0,839 (9). En Sportec: 0,922 → 0,897.

**Desfases en público, uno por cambio (frames):**
- último toque r 3: −19, −10, 8, 11, 15, 16, 34, 47, 48 (3 perdidos)
- zona de control: −38 … +55 (1 perdido, 21 cambios extra)
- más cercano sin histéresis: −74 … +37 (0 perdidos, **110 cambios extra**)

**Control de un posible desfase temporal balón–jugadores en el ATD:** desplazando la pista del balón ±20 frames, la distancia al jugador más cercano es mínima en −4 frames tanto en el ATD como en Sportec (mismo perfil). Es física normal (el jugador queda donde estaba el balón), no un desfase del tracker. **Descarta un desfase temporal**, no un sesgo espacial constante.

Gráfico: `experiments/outputs/008/timeline_public_rules.png`. Tablas: `grid.csv`, `best_on_public.csv`, `best_on_sportec.csv`.

## Hallazgos

1. **La bandera de Sportec se parece mucho al último toque:** 92 % de acuerdo y llega puntual (mediana −2 frames). Encaja con que las etiquetas de DFL salgan de eventos de toque.
2. **En el BAR‑ATH público las tres reglas empatan** (84–85 %). Con 12 cambios no se puede elegir: diferencias de 1 punto son ruido.
3. ~~**Contra la bandera de Metrica, todas las reglas llegan tarde**~~ *(corregido abajo: depende del filtro elegido)* (+15 a +20 frames de mediana, unos 0,6–0,8 s), mientras que en Sportec llegan a tiempo. *Hipótesis* (solo 9–11 cambios): la bandera de Metrica cambia antes de que el ATD muestre el toque, por ejemplo al interceptar o al salir el pase. Importa mucho, porque en S0.5 se vio que 0,5 s de retraso cuesta un 22 % de spellF1.
4. **En el ATD los toques necesitan un radio de 2–3 m** (en Sportec, 1 m). No es ruido de posición (el ATD vibra igual o menos que Sportec) ni un desfase temporal (descartado). Lo más probable es que el balón del ATD esté suavizado o interpolado y el giro se detecte cuando el jugador ya se ha separado. Lo confirmará la degradación (S2.1).
5. **El último toque es la regla que menos cambios inventa** (0,67 por cambio real en público, frente a 1,50–1,75 de las otras y 9 del más cercano sin histéresis). Como inventar posesiones cuesta en spellF1, es la mejor candidata a feature central.

## Decisión

**Modificar y seguir.** Usaremos las tres reglas como features (M1):
- el **último toque** como feature central, en una versión de radio múltiple (1, 2 y 3 m) para que sirva en los dos dominios;
- la **zona de control** como segunda feature;
- el **más cercano** solo como contexto.

Para el decodificador (S2.4) se considera, como **candidato** (ver corrección), un parámetro de **desplazamiento temporal global**, que se comprobará en los 5 min públicos dentro de la regla de ≤ 3 parámetros. Su motivo es el retraso de +15/+20 frames frente a la bandera de Metrica. Siguiente: **S1.3** (anatomía de los huecos del balón en el partido completo).

## Corrección tras la revisión de Codex 002 (2026-10-23)

Revisión: [codex_review_002](../reviews/codex_review_002.md), puntos 3, 4 y 7.

- **Código corregido** (`src/features/possession_rules.py`): las velocidades del balón ya no se calculan a través del descanso (Sportec junta las dos partes) y el estado de las reglas se reinicia en cada parte. La zona de control no exclusiva ya no favorece siempre a A: si los dos equipos cumplen, gana el más cercano, y en empate no decide. Hay tests nuevos de simetría y de dos partes.
- **Reejecución:** los mejores ajustes de Sportec no cambian (último toque 0,922, zona de control 0,877). En público, el mejor ajuste de la zona de control pasa a ser **no exclusivo, r 2 m, m 25** (acuerdo 0,852), pero solo empareja **6 de 12** cambios, con mediana **−0,5 frames**.
- **Desfases en público, uno por cambio** (ahora se guardan en `public_offsets_and_shift_profile.json`):

| Regla | desfases (frames) | emparejados | mediana | extra |
|---|---|---|---|---|
| más cercano sin histéresis | −74 … +37 | 12/12 | **−8,5** | 110 |
| zona de control (no exclusiva, r 2, m 25) | −75, −66, −3, 2, 55, 63 | 6/12 | −0,5 | 6 |
| más cercano, k 25 | −51 … +48 | 8/12 | +20 | 18 |
| último toque r 3 | −19 … +48 | 9/12 | +15 | 8 |

- **El hallazgo 3 queda sin demostrar:** el "todas llegan tarde" depende de los filtros elegidos (la histéresis retrasa por construcción) y de qué cambios sobreviven al emparejamiento. Además, se eligió y se evaluó en los mismos 5 minutos. **No hay un desfase común de Metrica establecido.** El desplazamiento temporal global queda como **candidato** del decodificador, no como necesidad. Si se usa, contará dentro de los ≤ 3 parámetros calibrados en público y se registrará junto a esa elección.
- La comprobación balón–jugadores ya está en el script (`shift_profile`). Descarta un desfase temporal, no un sesgo espacial constante.
