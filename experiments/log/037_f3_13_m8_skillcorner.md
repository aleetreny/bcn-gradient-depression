# 037 · F3.13 M8: SkillCorner como datos de entrenamiento — descartado  (2026-10-24, commit dc3f196)

Comando: `.venv311/bin/python scripts/s3_13_skillcorner.py`        Salida: `experiments/outputs/038/`        Lector: `src/data/skillcorner.py`

## Datos

**SkillCorner Open Data:** 20 partidos de la A‑League 2024/25 con tracking de retransmisión a 10 fps, interpolado a 25 fps.
- Solo se usan los jugadores y el balón **detectados por la cámara**: mediana de 12 jugadores visibles; balón visible el 59 % del tiempo.
- Etiquetas de SkillCorner, no de Metrica:
  - vivo = dentro de una fase de juego;
  - equipo = posesión por frame, arrastrada mientras el balón viaja.

Sus estadísticas se parecen mucho a las del reto:

| | SkillCorner (20) | públicos BAR‑ATH | Sportec |
|---|---|---|---|
| posesiones/min | 3,36 | 3,4 | 4,6 |
| % D | 46 % | — | 45 % |
| mediana de posesión | 5,6 s | — | — |

## Resultado (GBM v1ctl; decodificador fijo, cribado)

| | solo Sportec | + SkillCorner, solo cabeza de equipo | + SkillCorner, las dos cabezas |
|---|---|---|---|
| Sportec, LOMO exterior | **0,687** | 0,674 (1/7) | 0,671 (1/7) |
| Metrica Sample | **0,800** | 0,791 | 0,794 |
| Sample sin D ±3 s | **0,841** | 0,837 | 0,838 |
| públicos (desarrollo) | **0,584** | 0,488 | 0,500 |

**Diagnóstico** (acierto de equipo por frame con balón vivo):

| entrenado en → probado en | acierto |
|---|---|
| Sportec → Sportec degradado (referencia) | ~0,91 |
| Sportec → SkillCorner | 0,839 |
| SkillCorner → SkillCorner (otros 10 partidos) | 0,844 |
| SkillCorner → Sportec degradado | 0,824 |

## Hallazgos

1. **SkillCorner empeora en todos los jueces.** Más partidos no compensan su peor calidad.
2. **El problema es la información, no el desacuerdo de etiquetas.** Un modelo entrenado en SkillCorner solo llega a 0,84 de acierto en el propio SkillCorner, lo mismo que el modelo de Sportec. Con 12 jugadores visibles y el balón el 59 % del tiempo, el techo es bajo, y lo que enseña se transfiere peor (0,82 en Sportec frente a 0,91).
3. Por lo mismo, **preentrenar la TCN con él no es prometedor**: aprendería de una señal más pobre que la de Sportec degradado.

## Decisión

**Descartar M8.** El lector queda en el repo (`src/data/skillcorner.py`, con tests) por si sirve como juez de robustez, pero no entra en el entrenamiento.
