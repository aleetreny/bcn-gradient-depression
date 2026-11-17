# 043 · F4.6 Tracking de TV alineado y F4.7 modelo a nivel de jugador  (2026-10-24, commits 2279124, bc1e997)

## F4.6 Alinear el tracking de la retransmisión

Comando: `.venv311/bin/python scripts/s4_6_broadcast_align.py`        Salida: `experiments/outputs/044/`, `local_data/broadcast_alignment.npz`

La organización da un segundo tracking del partido, hecho desde la cámara de TV (164.488 frames). Tiene cortes y repeticiones y no viene con tabla de correspondencia. Se alinea por tramos continuos de homografía válida: un desfase constante por tramo, el que minimiza la distancia mediana de cada jugador de TV al jugador táctico más cercano. Sin etiquetas.

**Resultado:**
- 97 de 109 tramos alineados y 8 repeticiones detectadas;
- error mediano entre jugadores de **0,28 m**;
- desfase estable de ~4015 frames, que deriva de 4020 a 4014;
- cubre el **64 %** de los frames evaluados del partido.

**¿Aporta balón?** Estado del balón en los frames alineados (proporción):

| táctico \ TV | recta | falta | real |
|---|---|---|---|
| recta | 0,479 | 0,044 | 0,069 |
| falta | 0,057 | 0,151 | 0,009 |
| real | 0,054 | 0,010 | 0,127 |

Solo en el **5 % de todos los frames evaluados** la TV da una detección real donde el táctico tiene una recta o nada. Cuando el táctico va en recta, la TV suele ir en recta a la vez (48 %), lo que sugiere que Metrica fusiona las dos cámaras en un mismo balón. Cuando los dos son reales, discrepan 0,67 m de mediana.

**Decisión:** poco recorrido como fuente de balón. La alineación queda disponible (por ejemplo, para el vídeo de TV).

## F4.7 Modelo a nivel de jugador ("¿es este jugador el que tiene el balón?")

Comando: `.venv311/bin/python scripts/s4_7_player_model.py`        Salida: `experiments/outputs/045/`        Código: `src/features/players.py`

Por cada jugador visible se calculan:
- velocidad y aceleración;
- **atención**: cuánto corren hacia él los rivales y los compañeros;
- agobio: rivales a 5 y 10 m;
- posición;
- balón: distancia, rango y si el balón es una recta.

Etiqueta en Sportec: el jugador del equipo en posesión más cercano al balón limpio. Un LightGBM da la probabilidad de cada jugador, y por equipo se suma y se normaliza. La fusión con el GBM se hace sumando logits, con un peso elegido en 3 folds interiores sin el partido de prueba.

**Señal real pero débil:** atención de rivales mediana de 2,59 en el poseedor frente a 0,78 en el resto.

| acierto de equipo (frames vivos) | modelo de jugador | GBM v1ctl |
|---|---|---|
| todos | 0,781 | 0,912 |
| balón real | 0,802 | 0,932 |
| balón en recta | 0,794 | 0,930 |
| balón ausente | 0,610 | 0,757 |
| Metrica Sample | 0,832 | — |

**Fusión:** el peso elegido es 0 en los 7 folds, así que el LOMO (0,687) y el Sample (0,7995) no cambian.

**Decisión: descartar.** En esta forma sencilla es mucho más débil que el GBM y no aporta nada que este no tenga. Los trabajos publicados llegan al 92 % con redes de atención entre 22 jugadores con identidad fija y tracking limpio. Nuestro tracking rompe las identidades (46 slots), y llegar a eso sería un proyecto de varios días con resultado incierto.
