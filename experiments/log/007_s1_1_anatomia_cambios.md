# 007 · S1.1 Anatomía de los cambios reales  (2026-10-23, commit b71963d)

Comando: `.venv311/bin/python scripts/s1_1_transition_anatomy.py`        Config: ventana ±3 s        Semilla: n/a
Datos: BAR‑ATH ATD + verdad pública (5 min, 20 transiciones) + Smart Tagging.

## Resultado

20 transiciones: **12 cambios de equipo**, 4 inicios de D y 4 fines de D.

| # | t (s) | tipo | de→a | balón en el cambio | % balón ±0,5 s | más cercano en el cambio | dist. H / A (m) | cambio del más cercano (frames) |
|---|---|---|---|---|---|---|---|---|
| 1 | 6,6 | equipo | A→H | x,y | 100 | A | 3,4 / 2,2 | −74 |
| 2 | 14,7 | equipo | H→A | x,y | 100 | A | 4,8 / 2,5 | −14 |
| 3 | 17,6 | equipo | A→H | x,y | 100 | H | 1,1 / 4,0 | −44 |
| 4 | 59,2 | equipo | H→A | x,y | 100 | A | 11,4 / 7,0 | −18 |
| 5 | 66,6 | inicio D | A→D | nada (último balón −21 frames) | 0 | — | — | n/a |
| 6 | 72,2 | fin D | D→H | x,y | 92 | H | 4,1 / 8,5 | ya era H |
| 7 | 100,8 | equipo | H→A | x,y | 100 | A | 3,6 / 1,8 | −33 |
| 8 | 104,0 | equipo | A→H | x,y | 100 | A | 7,9 / 7,5 | +6 |
| 9 | 146,4 | equipo | H→A | x,y | 100 | A | 3,8 / 1,1 | −3 |
| 10 | 166,4 | equipo | A→H | x,y | 100 | A | 2,1 / 1,3 | +11 |
| 11 | 167,2 | equipo | H→A | x,y | 100 | H | 1,5 / 2,9 | +24 |
| 12 | 183,2 | equipo | A→H | x,y | 100 | A | 3,3 / 2,5 | +37 |
| 13 | 184,6 | equipo | H→A | x,y | 100 | A | 1,7 / 1,2 | +16 |
| 14 | 188,2 | inicio D | A→D | x,y | 100 | H | 6,6 / 18,8 | n/a |
| 15 | 208,6 | fin D | D→A | x,y | 100 | A | 4,8 / 2,3 | ya era A |
| 16 | 244,3 | equipo | A→H | x,y (reaparece −0,6 s) | 100 | H | 3,4 / 5,1 | +30 |
| 17 | 247,8 | inicio D | H→D | x,y | 100 | A | 13,9 / 3,4 | n/a |
| 18 | 266,1 | fin D | D→A | x,y (reaparece −0,5 s) | 100 | A | 6,3 / 2,1 | +48 |
| 19 | 273,7 | inicio D | A→D | x,y (balón fuera del campo 78 frames) | 100 | H | 5,3 / 14,2 | n/a |
| 20 | 288,2 | fin D | D→H | x,y | 100 | H | 3,6 / 16,1 | ya era H |

Desfase = frame en que el equipo más cercano pasa a ser el nuevo equipo − frame del cambio real (negativo = antes). Se mide en los 16 cambios hacia H o A.

Gráfico: `experiments/outputs/007/mini_timelines.png`, con 20 paneles (verdad, equipo más cercano, balón visible y distancia del balón al jugador más cercano de cada equipo). Tabla completa con Smart Tagging: `transitions.csv`.

## Hallazgos

1. **Los 12 cambios de equipo ocurren con el balón visible**: el 100 % de los frames en ±0,5 s. Según la regla de decisión de S1.1, la prioridad es **clavar el momento** (S1.2) más que rellenar el balón. *Matiz:* los 5 min públicos tienen un tracking mejor que la media del partido (S0.2), así que S1.3 debe confirmar que los cambios no caen en huecos en el resto del partido.
2. **El "más cercano" es un mal reloj para el cambio:** en 13 cambios su desfase va de −74 a +48 frames (de −3 s a +2 s), con mediana +6. No es un retraso fijo que se pueda restar: es ruido.
3. **El problema son los duelos.** En la mitad de los cambios (#3, #9–#13) hay jugadores de los dos equipos a 1–3 m del balón durante segundos y el más cercano parpadea. La verdad cambia en algún punto del duelo. En 3 de los 12 cambios, en el instante del cambio el más cercano todavía es del equipo antiguo (#8, #10, #12). Hace falta algo más fino que la distancia: quién **toca** el balón (cambio de dirección o de velocidad) o quién lo controla durante un tiempo. Es justo lo que compara S1.2.
4. **Robos cortos:** las dos posesiones públicas de menos de 2 s (#10→#11, 0,8 s, y #12→#13, 1,3 s) las refleja tarde o no las refleja: el más cercano marca H unos 0,4 s después o tras el final del tramo real. Encaja con el coste que se midió en S0.5.
5. **Inicios de D:** en 3 de 4 el balón sigue visible, pero se **aleja de todos** (> 13 m del equipo que lo tenía: #14, #17, #19) o **sale del campo** (#19). En #5 el balón desaparece 0,8 s antes. Señales útiles: distancia mínima creciente y balón fuera.
6. **Fines de D:** en 3 de 4 el equipo más cercano durante el parón ya es el que saca, porque el lanzador está junto al balón (#6, #15, #20). En 2 casos el balón reaparece justo antes del saque (#6 y #18, a −0,4/−0,5 s). Señal útil para el equipo del reinicio y su momento.
7. **Smart Tagging:** en la ventana pública solo hay un SET PIECES a menos de 10 s de un fin de D (#20: empieza 2,8 s antes del reinicio). Durante el D de #17–#18 hay un SET PIECES que empieza unos 12 s antes del reinicio real. Se medirá en S1.6 sobre todo el partido.

## Decisión

Seguir → **S1.2** (qué regla imita la bandera: más cercano, último toque o zona de control), evaluada solo en frames con balón visible, como dice el plan. En paralelo es necesario **S1.3**, para confirmar si en el partido completo los cambios caen también con el balón visible.
