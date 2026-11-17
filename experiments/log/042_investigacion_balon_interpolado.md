# 042 · Investigación: dónde perdemos y por qué. El balón del ATD está interpolado en línea recta  (2026-10-24)

Análisis sin entrenar nada. Todas las medidas del BAR‑ATH completo usan solo el tracking: los frames "en juego" son los que la v4 no llama D. Los públicos se usan solo para **entender** qué significan las etiquetas, como permite el reto; no se ajusta nada con ellos.

## 1. Dónde perdemos puntos (v4, LOMO anidado en Sportec)

| | IoU 0,3 | 0,5 | 0,7 |
|---|---|---|---|
| precisión | 0,885 | 0,802 | 0,685 |
| recall | 0,726 | 0,658 | 0,563 |

- **Lo que falla es el recall:** se escapa el 35 % de las posesiones reales.
- **Las cortas casi no se ven:** recall@0,5 del 24 % por debajo de 1 s y del 50 % entre 1 y 2 s, frente al 85–96 % por encima de 8 s.
- Las que se escapan tienen el balón visible el 76 % del tiempo, frente al 94 % de las acertadas.
- **El momento de las posesiones acertadas es casi perfecto:** inicio y fin a ±0,04 s (rango intercuartílico).

Oráculos sobre la v4:

| arreglo | ganancia |
|---|---|
| equipo perfecto en frames vivos | **+0,236** |
| posesiones cortas recuperadas | +0,119 |
| momento perfecto (±3 s) | +0,049 |
| sin posesiones inventadas | +0,032 |
| D perfecto | +0,012 |

## 2. En el BAR‑ATH los errores ocurren con el balón visible

En los públicos (comprobación), el balón se ve el 93,5 % del tiempo de juego, y **el 95 % de los frames con el equipo mal están con el balón visible**. Acierto de equipo con balón visible: 0,92.

Tipos de error en los 5 minutos:
- cambios de equipo tardíos;
- bordes de D: un saque de puerta con el portero a 0,3 m del balón durante 4 s, y un fuera de banda detectado 6,5 s tarde;
- un tramo de 8 s en el que el ATD pone a los jugadores del Athletic más cerca, pero la verdad da el balón al Barça.

**Retraso en los cambios:**

| dónde | mediana (estimado − real) |
|---|---|
| públicos | +0,3 a +0,8 s (solo 8 cambios emparejados, de −1 a +2 s) |
| Metrica Sample (846 cambios) | +0,12 s |
| Sportec | 0,00 s |

**No es lo bastante firme para actuar.**

## 3. La geometría balón–jugador del ATD no se parece a la del simulador

| | balón lento sin nadie a < 3 m | "golpeo" sin nadie a < 4 m | jugadores visibles a < 10 m |
|---|---|---|---|
| BAR‑ATH (ATD) | 0,11–0,24* | **0,25–0,28** | 3,36 |
| Sportec degradado (nuestro simulador) | 0,05 | 0,05 | 3,60 |
| Sportec limpio | 0,01 | 0,03 | 3,77 |
| Metrica Sample | 0,02 | 0,01 | 3,65 |

\* según la máscara de juego (indicador de parón o v4).

**Desfasar el balón respecto a los jugadores apenas lo arregla.** El mejor desfase son jugadores +0,2–0,4 s, y el golpeo fantasma solo baja de 0,25 a 0,21. No es un problema de sincronización.

## 4. La causa: el balón del ATD está dibujado a trozos rectos

Un frame es "lineal" si la segunda diferencia de la posición del balón es < 2 cm, es decir, si va a velocidad constante en línea recta.

| | % del balón en movimiento dentro de tramos rectos ≥ 0,4 s |
|---|---|
| **BAR‑ATH (ATD)** | **72 %** (74 % de todos los frames visibles en juego) |
| Sportec | 1–2 % |
| Metrica Sample 1 | 90 % |

- **Forma:** los tramos rectos duran una mediana de 17 frames (p90 108, p99 309) y alternan con tramos curvos muy regulares (mediana 15 frames, p75 20, p90 25). El balón recto va más despacio (5,2 m/s frente a 8,2) y la proporción no depende de cuántos jugadores haya cerca (0,73–0,77).
- **Por qué hay golpeos fantasma:** cerca de las uniones de los tramos rectos, el 47 % de los "golpeos" no tiene jugador a < 4 m, frente al 18 % en el resto. Con el balón lento dentro de un tramo recto no hay nadie a < 3 m el 32 % del tiempo, frente al 8 % fuera.

**Interpretación** (*hipótesis*): el ATD solo detecta de verdad el balón en ráfagas cortas y une las ráfagas con rectas. Del 59 % de frames con balón "visible", solo una cuarta parte es observación real. Nuestro simulador entrena con el balón de física real de Sportec, así que **el modelo aprendió a fiarse de la proximidad balón–jugador en tramos donde, en el BAR‑ATH, el balón es una línea inventada**. Esto encaja con:
- que los errores de equipo del BAR‑ATH estén con el balón visible;
- la caída de 0,71 (Sportec) a 0,61 (públicos).

## 5. Qué ha hecho otra gente

- **PathCRF (KDD 2026)** y **Ball Radar (KDD 2023)** infieren el poseedor solo con los jugadores: 92,4 % y 86 % de acierto de equipo sin balón. Su lección útil para nosotros: no depender del balón cuando no es fiable.
- PathCRF modela la posesión **a nivel de jugador**: quién la tiene, o hacia quién viaja el balón, con transiciones prohibidas.
- Los trabajos recientes de posesión desde tracking usan casi siempre "el más cercano dentro de un radio". Con un balón interpolado, esa regla falla justo donde lo hemos medido.

## Decisión propuesta

**Simulador v5:** reproducir en Sportec la interpolación del ATD. Se alternan ráfagas curvas y tramos rectos con las longitudes medidas aquí (sin etiquetas), y los tramos rectos se rellenan con la recta entre sus extremos. Se añaden features que digan al modelo cuándo el balón es una recta (y desde cuándo).

Cribado barato con el GBM: LOMO exterior y Sample, cuyo balón también es recto en un 90 %. Más un control de realismo: los golpeos fantasma del simulador deberían acercarse al 25 % del ATD. Si mejora, una corrida anidada completa.
