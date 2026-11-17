# 01 · Qué han conseguido otros trabajos y retos parecidos

Papers, datasets públicos y competiciones relacionados con detectar la posesión a partir de tracking, con cada cifra **comprobada en la fuente original**. Revisado el 21 oct 2026.

## Resumen (lo importante en 1 minuto)

- **Sí hay trabajo previo muy parecido al nuestro**: modelos que deciden qué equipo tiene el balón y cuándo el balón está parado usando solo las posiciones de los jugadores. Ninguno publica resultados con nuestra métrica (spellF1) ni sobre tracking de Metrica, así que sus cifras sirven de orientación, no de promesa.
- **PathCRF (KDD 2026)**: acierta el **equipo** con balón el **92,4 %** de los frames sin ver el balón. Su mejora más grande no viene de la red, sino de **prohibir transiciones imposibles al decodificar la secuencia** (+7,8 puntos en detección de eventos). Publica código y 5 modelos entrenados. Ojo: su cifra famosa de "69,6 %" mide otra cosa (qué jugador pasa a qué jugador).
- **Ball Radar (KDD 2023)**: adivina el equipo con balón el 86 % del tiempo sin ver el balón, y el 97 % viendo solo el 20 % del balón. Nosotros lo vemos el **59 %**, un caso más fácil que los de sus tablas.
- **Lang (2022)**: un modelo de árboles detecta el balón parado con un 92 % de acierto mirando solo cómo se mueven los jugadores. Pero sus fronteras fallan ~1,7 s de media, y eso en nuestra métrica es mucho.
- **Borghesi (2023)**: separar "¿balón vivo o parado?" de "¿qué equipo?" funciona bien (86–89 %).
- **El recurso más valioso es un dataset, no un modelo:** Sportec/DFL (7 partidos, 25 fps, licencia CC‑BY 4.0) trae en cada frame si el balón está vivo y qué equipo lo tiene. Son exactamente nuestras etiquetas H/A/D.
- **Cambiar de proveedor de tracking duele:** PathCRF pierde 25 puntos al pasar de un proveedor a otro. Habrá que adaptar los datos externos a cómo es nuestro tracking.
- **Decisión que sacamos:** primero modelo de árboles + decodificador temporal ajustado a spellF1; las redes (TCN), como retador después.

## Desarrollo (cifras comprobadas, para agentes)

Leyenda: ✅ comprobado en la fuente original · ⚠️ verificado con matiz/corrección · ❓ no verificable (paywall/bloqueo) · ⬇️ verificado pero poco relevante.

### PathCRF (Kim et al., KDD 2026, arXiv 2602.12080)

| Resultado o dato | Estado | Nota |
|---|---|---|
| Detecta eventos sin balón, solo trayectorias de jugadores | ✅ | El balón solo se usa para *construir* etiquetas de entrenamiento. |
| Tabla: Ball TP 47,01 / Non‑CRF 69,24 / Non‑CRF+Viterbi 69,68 / Static Masked 69,54 / Dynamic Masked 69,64 | ✅ | Las cifras son **edge accuracy** (estado = arista entre 26 nodos: 22 jugadores + 4 "fuera"). |
| ¿Qué mide el 69,6 %? | ⚠️ | No es acierto de equipo. Acierto de **equipo** por frame = **92,40 %** (share local 62,13 % real vs 62,64 % predicho). |
| F1 de eventos 48,39 → 65,34 → 73,14 → 73,83 → 75,69 | ✅ | El salto grande lo da **imponer restricciones de transición** (Non‑CRF → Non‑CRF+VCD = +7,8 pts) más que el CRF en sí (+2,5 pts más). Lección: *la decodificación con restricciones es lo que más rinde*. |
| Cinco checkpoints `saved/100…140` | ✅ | Repo con licencia MPL‑2.0, `inference.py`, `tutorial.ipynb`. |
| Caída de 24,72 pts al cambiar de proveedor | ✅ | LOMO Sportec 73,88 → Sportec→K League 49,16. **Matiz:** K League es **GPS (wearables)**, no óptico; los autores lo atribuyen a la tecnología y a convenciones de anotación. |
| (detalle importante) | ⚠️ | PathCRF **excluye los periodos de balón parado** (entrena/infiere solo en juego vivo) y asume **conjunto fijo de 22 jugadores**. Nosotros tenemos 46 slots sin identidad → adaptación no trivial. |
| (detalle importante) | ✅ | A 25 fps sube a 77,21 F1 vs 75,69 a 5 fps, pero entrenar cuesta ×12. |

### Sportec Open DFL (Bassek et al., Scientific Data 2025)

| Resultado o dato | Estado | Nota |
|---|---|---|
| 7 partidos completos, 1.002.644 frames, 25 Hz | ✅ | Temporada 2022/23: 2 de Bundesliga, 5 de 2. Bundesliga (5 de ellos con Fortuna Düsseldorf → poca diversidad de equipos). |
| `BallPossession` 1=local, 2=visitante; `BallStatus` 0=parado, 1=vivo | ✅ | Parado "cuando el árbitro interrumpe". El paper **no dice** si estos dos campos se generan automática o manualmente. |
| CC‑BY 4.0, figshare 10.6084/m9.figshare.28196177 | ✅ | También vía `floodlight` (`IDSSEDataset`) y `kloppy` (`sportec.load_open_tracking_data`, expone `ball_state` y `ball_owning_team_id`). ~2,4 GB. |
| Etiquetas compatibles con las nuestras | ⚠️ | Compatible en forma. Falta comprobar **semántica**: ¿cuándo cambia `BallPossession` (al primer toque o tras control)? Ver experimento E1.2 en [05_plan_para_ganar.md](05_plan_para_ganar.md). |

### SkillCorner Open Data

| Resultado o dato | Estado | Nota |
|---|---|---|
| 10 partidos A‑League 24/25, 10 fps, objeto `possession` (`player_id`, `group`) | ✅ | Tracking de **retransmisión** con jugadores extrapolados (`is_detected`). |
| Fases de juego | ✅ | Solo existen con balón vivo → la clase D habría que inferirla de huecos. Etiquetas generadas por SkillCorner, no corregidas. |

### Metrica Sample Data

| Resultado o dato | Estado | Nota |
|---|---|---|
| 3 partidos anónimos, tracking + eventos; el 3.º en EPTS FIFA | ✅ | **Mismo proveedor y mismo formato EPTS** que nuestro ATD. Ball Radar lo usó para entrenar. Etiquetas H/A/D habría que derivarlas de eventos. |

### Borghesi et al. 2023 (TCN, ESWA 223:119780)

| Resultado o dato | Estado | Nota |
|---|---|---|
| 86,2 % posesión (+7,3) y 89,2 % vivo/parado (+33,2) | ✅ (abstract) | Texto completo bloqueado (paywall / servidor PoliTo rechaza). |
| Ternario vs dos binarios; agregación invariante al orden (suma, self‑attention, conv 2D) | ✅ | |
| Mejor arquitectura: dos redes con self‑attention | ❓ | No pude verificarlo en el texto completo. |
| Datos confidenciales, sin código | ✅ | Proveedor Math&Sport. |

### Lang et al. 2022 (Sci. Rep., vivo/parado)

| Resultado o dato | Estado | Nota |
|---|---|---|
| 102 partidos, 47 de test; LR 86,7 / DT 84,6 / RF 89,3 / AdaBoost 92,0 | ✅ | |
| 81 % precisión en interrupciones > 2 s | ✅ | F1 de interrupciones = **79,8**. |
| 77 % inicios / 81 % finales a ≤ 2 s | ✅ | Pero solo **51 % / 58 % a ≤ 1 s**, y error medio **1,7–1,8 s**. Para spellF1 esto es mucho. |
| (detalle importante) | ✅ | **No usa el balón.** Solo pos/vel/acel de 22 jugadores con desfases ±0,4/0,8/2/4/20/40/80 s (1.320 features). Filtro de mediana de 3,64 s. Los autores creen que añadir el balón llevaría a casi 100 %. |

### Ball Radar (Kim et al., KDD 2023)

| Resultado o dato | Estado | Nota |
|---|---|---|
| H‑LSTM 3,69 m / 85,26 %; +RL 3,66 m / 85,85 % | ✅ | 85,85 % = acierto de **equipo** poseedor **sin ver el balón**. |
| Ablación xy → +vel → +vel,speed,acc: 82,04 → 84,90 → 85,85 % | ✅ | Usar derivadas sí ayuda. |
| Imputación 80/90/95/100 % → 97,41/95,04/91,54/83,96 % | ⚠️ | "80 %" = se oculta el 80 % del balón (se ve el 20 %). Nosotros vemos ~59 %, aunque nuestros huecos se concentran en jugadas difíciles. |
| Código sí, checkpoints no | ✅ | Repo `hyunsungkim-ds/ballradar`; datos: Fitogether (GPS, privado) + **Metrica sample** (público). |

### Peral et al. (arXiv 2512.19528, "WACV 2026")

| Resultado o dato | Estado | Nota |
|---|---|---|
| 283 partidos, tracking + recortes de vídeo, CropDrop | ✅ | Kognia + IRI (CSIC‑UPC). Datos propietarios, 6,25 Hz, ventanas de 9,6 s. |
| Tabla BallRadar / TranSPORTmer / ours / CropDrop | ✅ | Estado 79,77 → 85,85 %; poseedor 83,79 → 89,56 %. |
| Excluye secuencias con balón fuera | ✅ | No sirve para D. |
| Venue WACV 2026 | ❓ | arXiv no lo indica. |
| Ocultar trayectorias enteras empeora | ✅ | |

### Kaggle DFL Bundesliga Data Shootout (2022)

| Resultado o dato | Estado | Nota |
|---|---|---|
| Ranking y soluciones 1.º–3.º | ⬇️ | No verificado en detalle: es *action spotting* en vídeo, otra tarea. Las lecciones genéricas (validar por partido, ensembles, post‑proceso de secuencia) son correctas. |

## Qué nos llevamos

- **Usar:** Sportec como datos de entrenamiento y validación (dejando un partido fuera cada vez), estropear Sportec para que se parezca a nuestro tracking, separar vivo/parado de equipo, decodificador temporal con transiciones restringidas, Smart Tagging como señal auxiliar.
- **Tener en cuenta:** PathCRF acierta el equipo al 92 % (no al 70 %); nuestro régimen de balón visible es favorable; el grueso de la mejora de PathCRF viene de las restricciones de transición.
- **Orden de modelos:** árboles (GBM) antes que redes (TCN).
- **Más hallazgos propios:** [04_como_puntua_spellf1_y_otros_hallazgos.md](04_como_puntua_spellf1_y_otros_hallazgos.md).
