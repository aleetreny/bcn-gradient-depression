# 034 · F3.10 Features de control v2 — descartadas  (2026-10-24, commit 9e42cf4)

Comando: `.venv311/bin/python scripts/s3_10_control2_screen.py`        Código: `src/features/control.py` (`control_features_v2`, 12 features)

## Qué se probó

Sobre el conjunto v1ctl:
- **golpeos:** picos de aceleración del balón (> 15 m/s²) con un jugador a < 3 m; equipo del último golpeo, arrastrado dentro de la parte, y frames desde el golpeo;
- **balón en vuelo:** > 6 m/s sin nadie a menos de 2 m;
- **persistencia de la conducción:** frames desde la última conducción de cada equipo;
- **contexto largo:** medias de 4–8 s de la conducción y de la dirección del balón.

Por separado son muy informativas en Sportec limpio: AUC de equipo 0,93 para el último golpeo promediado en 3 s y 0,925 para la conducción promediada en 4 s.

## Resultado (cribado del GBM, decodificador sin reajustar)

| Juez | v1ctl | v1ctl2 |
|---|---|---|
| Sportec, LOMO exterior | 0,687 | 0,684 (3/7 mejores) |
| Metrica Sample (1 / 2 / 3) | 0,799 (0,812 / 0,749 / 0,837) | 0,795 (0,810 / 0,745 / 0,828) |
| públicos | 0,584 | 0,585 [0,459–0,702] |

## Decisión

**Descartar.** Son redundantes con lo que el modelo ya obtiene de las features de control v1 y del último toque. El camino de añadir features de posesión parece agotado con este modelo.
