# 031 · F3.6 Features de control — cribado  (2026-10-24, commit 8af563a)

Comando: `.venv311/bin/python scripts/s3_6_control_screen.py`        Config: `configs/degrade_v3.json`
Código: `src/features/control.py` (24 features, invariantes al orden de los slots y sin etiquetas)

## Qué se añade

- **Conducción:** para cada jugador, cercanía al balón × alineación de las velocidades × parecido de las velocidades. Se toma el máximo por equipo y la diferencia entre equipos, con medias de 0,4 y 1 s.
- **Receptor probable:** con el balón a más de 4 m/s, el jugador que va por delante más cerca de la trayectoria: su equipo, la distancia perpendicular y el tiempo de llegada.
- **Disputa:** jugadores de cada equipo a menos de 2 m del balón.
- **Dirección del juego:** velocidad del balón y de los equipos en x (hacia la portería de A es positivo), con medias de 1 y 3 s.

En Sportec limpio, la diferencia de conducción promediada en 1 s distingue el equipo con AUC 0,88 por sí sola, y la dirección del balón en 3 s con 0,72.

## Resultado (cribado: LOMO exterior, decodificador sin reajustar)

| Fold | GBM v1 (decodificado) | **GBM + control** | GBM v1 en el ensemble B | GBM + control en el ensemble B | acierto de equipo v1 → control |
|---|---|---|---|---|---|
| J03WMX | 0,615 | **0,640** | 0,656 | 0,667 | 0,903 → 0,910 |
| J03WN1 | 0,631 | **0,655** | 0,665 | 0,666 | 0,887 → 0,891 |
| J03WOH | 0,699 | **0,715** | 0,772 | 0,772 | 0,917 → 0,926 |
| J03WOY | 0,647 | 0,648 | 0,680 | 0,680 | 0,891 → 0,894 |
| J03WPY | 0,701 | 0,702 | 0,724 | 0,723 | 0,917 → 0,922 |
| J03WQQ | 0,704 | 0,711 | 0,739 | 0,736 | 0,917 → 0,919 |
| J03WR9 | 0,680 | **0,713** | 0,733 | 0,734 | 0,923 → 0,928 |
| **media** | 0,668 | **0,683 (+0,015, 7/7)** | 0,7098 | 0,7112 (+0,001, 4/7) | 0,908 → 0,913 |

**Transferencia a Metrica Sample** (GBM completo con control + decodificador de la v1.1): **0,798** frente a 0,770 sin control (+0,028). Mejora los 3 partidos: 0,782 → 0,814 · 0,730 → 0,760 · 0,798 → 0,821.

## Hallazgos

1. **Las features de control mejoran el GBM** (+0,015, 7/7) y **se transfieren muy bien** a Metrica Sample (+0,028, 3/3), justo el tipo de tracking del BAR‑ATH.
2. **Dentro del ensemble con la TCN apenas suman** (+0,001): la TCN ya capta en Sportec buena parte de esa información. Pero la TCN **no se transfiere** al Sample (+0,002, F3.5) y el control sí.
3. **Cambio de estrategia:** para el BAR‑ATH importa más lo que se transfiere que lo que solo luce en el LOMO de Sportec. Se usarán **los dos jueces** (LOMO de Sportec y Metrica Sample) para elegir la combinación final.

## Decisión

**Evaluación anidada completa con las features de control** en GBM, M3 y TCN (F3.7), con 4 combinaciones (GBM; GBM + M3; ensembles con TCN), y transferencia de cada una al Sample.
