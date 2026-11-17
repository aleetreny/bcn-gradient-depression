# 015 · S2.1 Degradación calibrada de Sportec  (2026-10-23, commit 144db8d)

Comandos: `.venv311/bin/python scripts/s2_1_degradation.py --trials 60` (iteraciones de calibración) y `.venv311/bin/python scripts/s2_1_degradation.py --config configs/degrade_v1.json` (evaluación final)
Config: `configs/degrade_v1.json`        Semillas de degradación: 0, 1, 2
Código: `src/data/degrade.py`, `src/features/frame_features.py` (features v0) · tests: `tests/test_degrade.py` (3 verdes: determinista, **no lee las etiquetas**, slots de equipo fijo)

## Qué hace la degradación

A partir de Sportec limpio, **sin mirar las etiquetas**, en este orden:
1. **Suaviza** las posiciones de los jugadores (media móvil de 9 frames) y recalcula sus velocidades.
2. **Hace desaparecer jugadores**: ausencias sueltas y episodios en los que 2–5 jugadores de un equipo desaparecen a la vez.
3. **Rompe identidades** y reparte los fragmentos en un banco de slots de equipo fijo (24 H y 22 A, como el ATD).
4. **Crea duplicados**: el slot viejo sigue un rato junto al jugador.
5. **Abre huecos largos de balón**, con un riesgo que depende de la velocidad del balón, de los jugadores cerca y de un indicador de parón calculado con el tracking. Entre huecos queda un tramo visible mínimo.
6. **Oculta el balón cuando está quieto** más de 1 s, con probabilidad 0,89.
7. **Añade apagones** de unos 3 por partido, de 35 a 65 s.

El tercer canal del balón no se simula: **las features v1 no lo usan**.

**Calibración** con Optuna en 2 partidos (J03WMX, J03WQQ) contra las estadísticas del ATD medidas con el mismo código. Primero los jugadores (distribución de visibles por equipo y fragmentos nuevos por minuto) y después el balón (visible en juego en movimiento y en tramos lentos, huecos por partido, mediana del hueco y % de balón quieto).

| Iteración | qué cambió | resultado |
|---|---|---|
| 0 (prueba) | valores por defecto | asimetría H/A por un banco de slots compartido (**fallo corregido**: bancos por equipo) y huecos demasiado largos (sorteo uniforme en intervalos anchos; **corregido**: escala logarítmica) |
| 1 | tasas de hueco por contexto, tramo visible mínimo, regla del **balón quieto** | balón bien en juego en movimiento; duplicados bajos (3,4 %) |
| 2 | más peso a los duplicados, al balón quieto y a la visibilidad en tramos lentos | jugadores clavados; el balón empeora |
| **v1 final** | **jugadores de la iteración 2 + balón de la iteración 1** (calibraciones independientes) | tabla siguiente |

Las salidas de las iteraciones 1 y 2 están en `experiments/outputs/015/iter1/` y `iter2/`.

## Resultado (v1 final; 7 partidos × 3 semillas)

| Estadística | ATD BAR‑ATH | Sportec limpio | **Sportec degradado v1** |
|---|---|---|---|
| balón visible (frames evaluados) | 0,593 | 1,000 | 0,565 |
| balón visible, juego en movimiento | 0,779 | 1,000 | **0,766** |
| balón visible, tramos lentos | 0,262 | 1,000 | **0,169** (más bajo) |
| huecos por 90 min | 127 | 0 | 118 |
| mediana / p75 del hueco | 10,7 / 21,1 s | — | 12,1 / 26,1 s |
| balón quieto (< 0,5 m/s) entre los frames visibles | 5,0 % | 46,3 % | 12,5 % |
| visibles por equipo ≤8 / 9 / 10 / 11 / ≥12 (%) | 5,5 / 5,6 / 22,4 / 59,8 / 6,7 | 0 / 0 / 9,3 / 90,7 / 0 | **5,4 / 4,8 / 22,9 / 60,3 / 6,6** |
| fragmentos nuevos por minuto | 11,2 | 0,0 | **11,1** |
| frames con < 12 slots | 3,6 % | 0 % | 2,4 % |
| vibración de jugadores | 0,0014 m | 0,0100 m | 0,0016 m |

**Validación adversaria:** LightGBM intenta distinguir frames del BAR‑ATH de frames de Sportec; un frame de cada 10, CV de 5 folds agrupada por bloques de 1 min.

| Features | Sportec limpio | Sportec degradado v1 | Lo que más delata (degradado) |
|---|---|---|---|
| todas + vibración | 1,000 | 0,998 | vibración (la forma del suavizado del ATD es distinta) |
| todas sin vibración | 0,998 | 0,950 | dispersión y centroides de los equipos |
| **calidad de tracking** (balón visible y huecos, visibles, distancias, velocidad del balón) | **0,927** | **0,745** | visibles (total, H y A), velocidad del balón, frames desde que se vio el balón |
| estilo de juego (centroides, dispersión, velocidades) | 0,991 | 0,903 | dispersión y centroides en x |

**Nueva referencia — baseline smooth50 en LOMO Sportec degradado v1:** spellF1 **0,300** (semillas 0,303 / 0,301 / 0,295; desviación entre folds ~0,020); macroF1 0,74; **5,2 posesiones/min**; **36 % de D** predicho.

## Hallazgos

1. **La degradación imita bien a los jugadores** (visibles por equipo, fragmentos y duplicados prácticamente iguales al ATD) **y razonablemente al balón** (visibilidad en juego en movimiento, número y longitud de huecos). Lo que peor sale: balón visible en tramos lentos (0,17 frente a 0,26) y balón quieto (12 % frente a 5 %).
2. **Puerta orientativa no superada del todo:** el AUC con features de calidad baja de 0,93 a 0,745, sin llegar a 0,7. Lo que aún delata es el número de visibles **a lo largo del tiempo** (la distribución por frame ya cuadra, así que debe de ser su estructura temporal) y la velocidad del balón.
3. **El estilo de juego delata siempre (AUC 0,90)** y no es arreglable degradando: son partidos de otra liga y otro estilo. Es un cambio de dominio real que el modelo tendrá que tolerar. Conviene que las features dependan poco de centroides absolutos.
4. **La vibración de los jugadores distingue los dominios casi perfectamente**, aunque las medianas se parezcan: el ATD suaviza de otra manera. **No se usará ninguna feature de vibración**, y las velocidades se usarán suavizadas.
5. **Evidencia a favor:** el baseline se comporta en Sportec degradado (5,2 posesiones/min, 36 % D) casi como en el BAR‑ATH completo (5,6 y 32 %). En Sportec limpio daba 4,0 y 0 %. Es decir, la degradación reproduce cómo reacciona un sistema real a nuestro tracking.
6. **Riesgo a vigilar:** en tramos lentos el balón falta más que en el ATD (0,17 frente a 0,26 visible), así que en Sportec degradado "sin balón → D" es una pista algo más fuerte que en la realidad. Un modelo podría apoyarse demasiado en ella.

## Decisión

**Seguir con `degrade_v1`**, con dos salvaguardas para no depender de una sola degradación (riesgo del plan):
- entrenar con **varias semillas y variantes** (iteraciones 1 y 2 y una con menos ocultación del balón quieto) como aumento de datos;
- informar cada modelo también en **LOMO limpio**.

Si un modelo cambia mucho de una variante a otra, se revisará la degradación. Siguiente: **S2.2 Features v1**.

## Corrección: degradación v2 (detectada en S2.2, commit ee08b7b)

S2.2 mostró que en v1 los fragmentos del portero acababan en slots de jugador de campo. Así el portero contaba en la dispersión y en los centroides, y el slot marcado como portero lo ocupaban otros jugadores: la posición del portero se alejaba del ATD **más** después de degradar que antes. Ahora cada equipo tiene un **slot reservado al portero**, como en el ATD (`degrade.VERSION = 2`; las claves de caché llevan la versión). Reevaluada con la misma config: las estadísticas, el AUC (0,746 con features de calidad) y el baseline (0,300) **no cambian**, porque ni las features v0 del AUC ni el baseline usan el portero. Fragmentos nuevos por minuto: 11,0.

## Corrección: degradación v3 + `configs/degrade_v2.json` (detectada en S2.3, commit 563da50)

S2.3 dio un AUC vivo/parado de **1,000** en los 7 folds. La causa: **Sportec congela el balón en los parones** (velocidad exactamente 0 en el 99,8 % de los frames de D con balón visible, y solo en el 0,2 % de los de juego). En el ATD eso no ocurre nunca: en los D públicos el balón se mueve con mediana 2,7 m/s porque los jugadores lo recogen y lo colocan. Tras la degradación v2 aún quedaba visible un 14 % de esos frames congelados: un atajo perfecto que solo existe en Sportec.

Arreglo (`degrade.VERSION = 3`): en cada tramo de balón quieto, el balón visible **sigue al jugador más cercano** con un pequeño desplazamiento y ruido. Se **recalibraron solo los parámetros del balón** (jugadores de v1 fijos) → `configs/degrade_v2.json`:

| Estadística | ATD | degradado v1 | **degradado v2 (v3 del algoritmo)** |
|---|---|---|---|
| balón visible, juego en movimiento | 0,779 | 0,766 | **0,769** |
| balón visible, tramos lentos | 0,262 | 0,169 | **0,227** |
| balón quieto entre los frames visibles | 5,0 % | 12,5 % | **3,9 %** |
| huecos por 90 min | 127 | 118 | 88 (peor) |
| mediana del hueco | 10,7 s | 12,1 s | 16,8 s (peor) |
| visibles por equipo y fragmentos | = ATD | = ATD | = ATD |
| AUC adversario con features de calidad | — | 0,746 | 0,753 |
| baseline LOMO degradado | — | 0,300 | **0,298** |

Mejora lo que importaba para no dejar atajos (balón quieto y visibilidad en parones), a cambio de huecos algo más largos y menos numerosos. **Desde aquí la referencia es `degrade_v2.json`.**

## Degradación v4 del algoritmo + `configs/degrade_v3.json` (revisión 003, commit del informe 022)

El balón de parón de la v3 iba pegado a un jugador (revisión 003). Nuevo modelo de balón de parón calibrado contra los D públicos: ver [022](022_s2_1b_balon_parado.md). Evaluación con `degrade_v3.json` (`experiments/outputs/015_v3/`):
- balón visible en juego en movimiento 0,771 y en tramos lentos 0,224;
- balón quieto 3,6 %;
- huecos 88 por 90 min (mediana 16,6 s);
- jugadores iguales al ATD;
- **AUC adversario con features de calidad 0,728** (antes 0,753);
- baseline LOMO 0,290.

**Referencia desde aquí: `degrade_v3.json`.**
