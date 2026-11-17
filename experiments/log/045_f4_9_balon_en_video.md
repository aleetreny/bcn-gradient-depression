# 045 · F4.9 Detector de balón propio en el vídeo de TV — no supera al ATD  (2026-10-24)

Comando: `.venv_video/bin/python scripts/s4_9_video_ball_probe.py`. Es un entorno aparte con ultralytics 8.4 y OpenCV; el entorno de construcción no se toca.

## Qué se probó

El vídeo de la retransmisión (1080p, 25 fps, 164.487 frames) está alineado con el táctico (log 043). Se pasa un detector genérico (YOLO11, clase "sports ball" de COCO, entrada de 1280 px, en MPS) y se proyecta el pie de la caja al campo con la homografía inversa. Los frames se procesan en local y no se muestran a nadie.

## Resultado (900 frames de 4 tramos largos de juego)

| | detector | ATD (balón de TV de Metrica) |
|---|---|---|
| frames con balón | 98 % | 90 % |
| velocidad | 0,11–0,13 s/frame (~2,5 h el partido alineado) | — |
| distancia detector–ATD | mediana 0,8–1,0 m | — |
| cuando discrepan más de 3 m (242 frames): jugador a < 1,5 m del balón | **14 %** | **47 %** |
| frames sin balón ATD y detector con confianza > 0,3 (43): jugador a < 1,5 m | 7 % | — |

## Hallazgos

1. **Casi siempre ve "un balón"**, pero cuando discrepa con el ATD suele ser el detector el que se equivoca: falsos positivos, o balones en el aire mal proyectados al suelo, porque la homografía solo vale para el plano del campo.
2. Donde el ATD no tiene balón, lo que detecta casi nunca tiene un jugador cerca. **No rellena los huecos de forma útil.**
3. La coincidencia con el ATD es mayor en sus tramos rectos (0,9 m) que en los curvos (1,7 m). Los tramos curvos parecen ser balones en vuelo o detecciones que bailan.

## Decisión

**No usar el vídeo con un detector genérico.** Superar al balón de Metrica exigiría un detector entrenado con balones de fútbol en retransmisión y un modelo de altura del balón: un proyecto de varios días con resultado incierto. Queda anotado como posible línea futura.
