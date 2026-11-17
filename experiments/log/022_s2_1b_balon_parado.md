# 022 · S2.1b Auditoría y calibración del balón de parón (revisión 003, punto 2)  (2026-10-23, commit 1cfaf35)

Comando: `.venv311/bin/python scripts/s2_1b_dead_ball.py`        Config resultante: `configs/degrade_v3.json` (= degrade_v2 + `dead_ball_model: "atd"`)
**Las etiquetas se usan solo para auditar el simulador.** La degradación nunca las lee (tests `tests/test_degrade.py`).

## Problema

La degradación v3 (`degrade_v2.json`) sustituía el balón congelado de Sportec por un balón **pegado a un jugador**. Durante los parones: distancia mediana al jugador más cercano de 0,68 m y un jugador a ≤ 1,24 m en el 95 % de los frames, frente a 2,9 m y 32 % en el ATD. Otra huella artificial de D.

## Qué se hizo

Nuevo modelo de balón de parón (algoritmo v4, `dead_ball_model: "atd"`):
- cada parón del balón original se convierte en tramos de 1–4 s en los que el balón **lo tiene uno de los 3 jugadores más cercanos** (a 0,3–1,5 m, siguiéndolo) o **rueda o vuela** en línea recta hasta un punto a 3–15 m, con una velocidad lognormal;
- el último segundo vuelve al punto del saque.

Se calibraron 3 constantes globales con una parrilla de 27 combinaciones en 2 partidos, contra los D públicos del ATD. Resultado: probabilidad de rodar 0,7, velocidad mediana 3,5 m/s y distancia máxima de posesión 1,5 m.

## Resultado

Balón visible durante los D (media de los 7 partidos, semilla 0):

| Fuente | balón visible en D | distancia mediana al jugador | % a ≤ 1,24 m | velocidad mediana / p75 |
|---|---|---|---|---|
| **ATD públicos (verdad)** | 66 % | **2,89 m** | **32 %** | **2,69 / 5,65 m/s** |
| ATD partido completo (indicador de parón) | 26 % | 1,98 m | 36 % | 1,85 / 4,64 m/s |
| Sportec degradado v2 (balón pegado) | 19 % | 0,68 m | 95 % | 1,20 / 1,93 m/s |
| **Sportec degradado v3 (balón de parón ATD)** | 18 % | **2,89 m** | **29 %** | 2,00 / 3,78 m/s |

**Capacidad de distinguir D de juego** (más alto = más fácil):

| | velocidad del balón sola | sonda de 6 features del balón (CV) | sonda, solo frames con balón visible |
|---|---|---|---|
| ATD públicos | 0,686 | 0,747 | 0,702 |
| Sportec v2 | 0,889 | 0,935 | — |
| Sportec v3 | 0,789 | 0,925 | 0,942 |

**Transferencia** (entrenar en Sportec degradado y medir el AUC de D en los públicos del ATD):

| | sonda de 6 features del balón | detector de D con las 160 features |
|---|---|---|
| Sportec v2 | 0,731 | 0,950 |
| **Sportec v3** | **0,770** | **0,976** |

Feature a feature, el AUC de D frente a juego con balón visible ya es comparable entre Sportec v3 y el ATD:

| Feature | Sportec v3 | ATD |
|---|---|---|
| velocidad del balón | 0,79 | 0,69 |
| velocidades medias en 0,8–2 s | 0,77–0,78 | 0,70–0,76 |
| aceleración | 0,59 | 0,70 |
| distancia al jugador de A | 0,56 | 0,80 |
| velocidad media de los jugadores | 0,91 | 0,79 |

## Hallazgos

1. **La huella del balón pegado al jugador desaparece:** distancia y proporción a ≤ 1,24 m iguales al ATD. La velocidad del balón de parón sigue algo baja (2,0 frente a 2,7 m/s). El mejor ajuste quedó en el borde de la parrilla.
2. **Lo aprendido en Sportec v3 se transfiere mejor al ATD real:** el detector de D con todas las features pasa de 0,950 a 0,976 de AUC en los públicos, frente a 0,997 dentro de Sportec. La distancia de transferencia se reduce aproximadamente a la mitad.
3. **La sonda dentro de Sportec sigue alta (0,92–0,94) frente a 0,70–0,75 en los públicos**, pero esa comparación está desequilibrada: unos 700 parones frente a 4 en solo 5 bloques. La prueba justa es la de transferencia (hallazgo 2).
4. **Sigue abierto:**
   - la visibilidad del balón en D (en el ATD, 66 % en los públicos frente a 26 % con el indicador en el partido completo; los públicos son un tramo de tracking muy bueno);
   - la velocidad de los jugadores separa D algo mejor en Sportec que en el ATD (0,91 frente a 0,79), un efecto del dominio y no del balón simulado.

## Decisión

**Adoptar `configs/degrade_v3.json`** como degradación de referencia. Evaluación de la degradación con esta config (S2.1): AUC adversario con features de calidad 0,728 (antes 0,753) y baseline LOMO 0,290.
