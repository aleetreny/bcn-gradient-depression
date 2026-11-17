# 041 · Regla nueva sobre los públicos y calibración del simulador sin etiquetas  (2026-10-24)

Comando: `.venv311/bin/python scripts/s2_1b_dead_ball.py --out experiments/outputs/042`        Config: `configs/degrade_v3.json`

## Qué ha cambiado en el reto

El repositorio oficial se actualizó el 22 y 23 de octubre (CHALLENGE, DATA, EVALUATION, FAQ, RULES):
- **Los 5 minutos públicos no son datos de entrenamiento ni un conjunto de ajuste.** Solo sirven para comprobar que la cadena es correcta: frames alineados, fichero bien formado, etiquetas que significan lo que creemos. Nunca para elegir entre dos versiones.
- **Ninguna etiqueta a mano sobre el partido del reto**, ni para entrenar ni para validar. Sí se permiten etiquetas a mano sobre otros partidos.
- Los datos externos tienen que ser de **otros partidos**.

## Qué nos afectaba

**Solo una pieza:** la calibración del balón parado del simulador (S2.1b, log 022). Elegía 3 constantes minimizando la distancia a las estadísticas del balón en los D públicos (hallazgo 1 de la revisión 004).

Todo lo demás encaja con la regla:
- los parámetros de degradación (S1.3/S1.4) salen del tracking completo **sin etiquetas**;
- ningún modelo se entrenó con los públicos (log 039);
- los decodificadores se ajustan en Sportec;
- las elecciones de la Fase 4 se hicieron con el LOMO y el Sample.

En la regla de la Fase 4, los públicos solo bloquean un fichero roto, lo cual es "comprobar que la cadena es correcta".

## Calibración nueva, sin ninguna etiqueta

- **Objetivo:** el balón visible en los frames del partido completo que el indicador de parón marca como lentos. El indicador solo mira el movimiento de los jugadores y exige ≥ 12 visibles.
- **Rejilla:** las mismas estadísticas en los frames lentos, según el mismo indicador, de los **7** partidos de Sportec degradados. Antes se usaban los D reales de dos partidos.

Las etiquetas solo aparecen en la auditoría, que informa y no elige.

| dead_roll_prob | dead_speed_median | dead_hold_max_m | pérdida |
|---|---|---|---|
| **0,7** | **3,5** | **1,5** | **0,012** |
| 0,7 | 2,5 | 1,5 | 0,028 |
| 0,5 | 3,5 | 2,5 | 0,058 |

## Hallazgos

1. **Sale exactamente la misma configuración** que con los públicos (0,7 / 3,5 / 1,5), y con un margen más claro (0,012 frente a 0,028 de la siguiente). `configs/degrade_v3.json` no cambia, salvo su procedencia.
2. Por tanto, **la v4, la reserva y todas las cifras siguen siendo válidas** sin reentrenar. Los ficheros conservan su hash, porque el constructor ignora la procedencia.
3. El hallazgo 1 de la revisión 004 queda cerrado: la cadena completa ya no lee etiquetas públicas en ningún ajuste. El anidado del LOMO tampoco depende ya de los D de J03WMX/J03WQQ.

## Decisión

Calibración sin etiquetas adoptada. A partir de aquí, los públicos solo se usan para comprobar formato y alineación, y para bloquear un fichero roto.
