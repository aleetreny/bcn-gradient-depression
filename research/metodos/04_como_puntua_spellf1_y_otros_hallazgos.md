# 04 · Cómo puntúa spellF1 y otros hallazgos propios

## Resumen

1. **Lo que hay que imitar es una "bandera" de un sistema de tracking, no a un árbitro.** La verdad oficial se generó *mecánicamente* a partir de dos campos del tracking corregido de Metrica: "balón muerto" y "equipo dueño". No hay suavizado ni criterio humano. Por eso cambia de dueño muy rápido, con posesiones de **0,76 s y 1,3 s**.
2. **El error más caro es llegar tarde a un cambio H↔A**, no equivocarse en los bordes del balón parado. 1 s de retraso global cuesta un 22–35 %. Empezar el balón parado 2 s tarde casi no cuesta nada.
3. **Perderse un robo corto cuesta triple.** Si no ves un "A→H 1 s→A", fusionas las dos posesiones de A y pierdes tres de golpe. Por eso suavizar mucho es malo.
4. **Truco de decodificación:** cuando el modelo sospecha un robo corto pero no está seguro, es mejor escribir un **D corto** que nada o que el equipo contrario. Un D separa posesiones y casi no penaliza si te equivocas. Hay que confirmarlo en Sportec.
5. **Hay competencia seria:** otro equipo está entrenando un detector de balón con el propio vídeo para rellenar los huecos (issue #2 del repo oficial). Esa es probablemente la línea "de máximo techo".
6. **Segunda fuente de balón:** el ATD de la retransmisión ve el balón el 50 % del tiempo con *otras* oclusiones. Si se sincroniza con la táctica, puede rellenar parte de los huecos.

## Desarrollo

### 1. Qué "significa" la verdad (y la hipótesis de trabajo)

CHALLENGE.md: la verdad sale de forma *mecánica* del tracking **corregido**. Un frame es D donde el tracking marca el balón muerto y, si no, el equipo que el tracking nombra como dueño. Nada se suaviza, interpola ni ajusta a mano.

Consecuencias:

- El ATD automático que recibimos es, muy probablemente, el **precursor sin corregir** de ese tracking. Cuando el balón del ATD es visible y bueno, la "regla de dueño" del proveedor se puede emular con geometría (jugador más cercano con histéresis, o toque = balón dentro del radio de control + cambio de dirección o velocidad, a lo Vidal‑Codina).
- **Hipótesis H1 (a validar en local con los 5 min públicos):** en los frames con balón visible, una regla de "último jugador que tocó el balón" con 2–3 parámetros (radio, velocidad mínima del balón, frames de confirmación) reproduce la verdad mucho mejor que el "más cercano + ventana ±50". Si se cumple, el problema se reduce sobre todo a **reconstruir el balón en los huecos** y a **detectar el balón muerto**.
- Ajustar solo 2–3 parámetros físicos con 5 min es aceptable. Ajustar penalizaciones de suavizado con ellos, no (EVALUATION.md demuestra que elige mal).
- Sportec/DFL también es una "bandera de tracking" (`BallPossession`) y no un juicio humano. Su semántica debería parecerse más a la nuestra que unas etiquetas derivadas de eventos. Hay que comprobarlo (E1.1 en [05](05_plan_para_ganar.md)).

### 2. Estructura de la verdad pública (5 min, datos públicos del repo)

| # | Etiqueta | Inicio | Duración |
|---|---|---|---|
| 1 | A | 11185 | 6,6 s |
| 2 | H | 11351 | 8,1 s |
| 3 | A | 11553 | 2,8 s |
| 4 | H | 11624 | 41,6 s |
| 5 | A | 12665 | 7,4 s |
| 6 | D | 12851 | 5,5 s |
| 7 | H | 12989 | 28,6 s |
| 8 | A | 13704 | 3,2 s |
| 9 | H | 13785 | 42,4 s |
| 10 | A | 14846 | 20,0 s |
| 11 | **H** | 15346 | **0,76 s** |
| 12 | A | 15365 | 16,0 s |
| 13 | **H** | 15766 | **1,32 s** |
| 14 | A | 15799 | 3,6 s |
| 15 | D | 15889 | 20,4 s |
| 16 | A | 16400 | 35,7 s |
| 17 | H | 17293 | 3,5 s |
| 18 | D | 17380 | 18,3 s |
| 19 | A | 17838 | 7,6 s |
| 20 | D | 18027 | 14,5 s |
| 21 | H | 18389 | 11,8 s |

- 12 de las 16 fronteras entre posesiones consecutivas son **cambios directos H↔A** (sin D por medio). El cambio de equipo es el evento dominante.
- 6 de 17 posesiones duran < 4 s. Ventanas de ±2 s (baseline smooth50) ya amenazan las de 0,76 s y 1,32 s.
- Ritmo: 17 posesiones en 300 s de juego ≈ **3,4/min** → del orden de **300–400 posesiones** en el partido. Es un orden de magnitud para controlar el recuento predicho, no un objetivo a clavar.

### 3. Sensibilidad de spellF1 (simulación con el evaluador oficial sobre los 5 min públicos)

Script: [`scripts/spellf1_error_costs.py`](../../scripts/spellf1_error_costs.py) (reproducible, solo usa la verdad pública). Partiendo de una predicción perfecta:

| Error introducido | spellF1 |
|---|---|
| Perfecto | 1,000 |
| Todo con 5 frames (0,2 s) de retraso | 0,977 |
| Todo con 12 frames (0,5 s) de retraso | 0,906 |
| Todo con 25 frames (1 s) de retraso | 0,777 (0,654 en el partido completo según EVALUATION.md) |
| Todo con 50 frames (2 s) de retraso | 0,588 |
| D empieza 25 frames tarde | **1,000** |
| D empieza 50 frames tarde | 0,977 |
| No predecir nunca D (rellenar con el equipo anterior) | 0,812 |
| Fusionar posesiones < 1 s / < 2 s / < 3 s / < 4 s | 0,913 / 0,827 / 0,743 / 0,560 |
| 1 / 3 / 6 cambios espurios de 1 s en posesiones largas | 0,933 / 0,820 / 0,687 |
| Robos cortos reales escritos como **D** en vez de con su equipo | 0,938 |
| Robos cortos reales no detectados (fusionados) | 0,827 |
| 1 / 3 falsas alarmas de 1 s escritas como equipo contrario | 0,911 / 0,760 |
| 1 / 3 falsas alarmas de 1 s escritas como **D** | 0,937 / 0,822 |

Lecturas (con 17 posesiones; los valores exactos tienen mucho ruido, pero las *direcciones* son estructurales):

- **El momento exacto de los cambios H↔A es lo que más importa.** Hay que apuntar a errores < 0,3 s en las transiciones con balón visible. Ahí las señales de "toque" valen oro.
- **Las fronteras de D toleran 1–2 s** siempre que el tramo D exista (sin D, las posesiones vecinas se estiran y fallan el IoU). Prioridad para D: *detectarlo*, no clavar el frame.
- **Un falso cambio de 1 s cuesta ~1/N_posesiones × 2**: crea una posesión espuria y parte una real. **Un robo corto perdido cuesta ~3/N**, porque fusiona a sus vecinos. Hay que equilibrar ambos con el decodificador y ajustarlo con spellF1 en Sportec, nunca con accuracy.
- **D como separador barato:** ante una sospecha de robo corto con equipo incierto, un D breve conserva separadas las posesiones vecinas y en una falsa alarma cuesta menos que poner el equipo contrario. Se implementa como un estado más en el decodificador (ver E4.4) y se valida en Sportec. **No** se trata de rellenar el CSV de D sin criterio.
- `X` en frames evaluados: no aporta nada que no pueda hacer el decodificador, y castiga macroF1, que es el desempate. **No usar X dentro de las partes.**

### 4. Estructura gratuita y legítima (publicada por los organizadores)

- 2.ª parte: frames **110.808–113.741 son D** (2.934 frames antes del saque inicial de la 2.ª parte; FAQ.md).
- Inicio de cada parte = saque inicial. Tras un gol, saca el equipo que lo encaja (Smart Tagging marca `Kick Off (After Goal)` ×4 y `Kick Off (Start)` ×2).
- Smart Tagging `SET PIECES` (59) da el **tipo** y el **momento** de los reinicios. Son anclas para cierres de D y para "quién saca". Los saques de banda vienen con `Team = N/A`, así que en ellos hay que inferir el equipo con la geometría (quién está en la banda con el balón).

### 5. Competencia y techo: balón desde el vídeo

Issue #2 del repo oficial (equipo SyntaxEror): entrenan un detector de balón pequeño sobre el **vídeo táctico**. Usan como etiquetas automáticas las posiciones del ATD proyectadas con la homografía y lo aplican solo donde el ATD no tiene balón. Preguntan si los pesos (3,5 MB) pueden ir en el repo y si se acepta una GPU. **Sin respuesta oficial todavía.**

- Es una idea sólida, autosupervisada y 100 % local: hay ~113k frames con balón visible como etiquetas gratis.
- Para nosotros: línea **E6**, "techo alto". Solo merece la pena si (a) la respuesta al issue #2 lo permite y (b) E3 demuestra que el error está concentrado en los huecos de balón. El repo debe poder regenerar el CSV (RULES.md), así que el entrenamiento ha de ser determinista o los pesos publicables.
- Detalle práctico: los huecos son los frames *difíciles* (balón tapado o en el aire). Un detector entrenado con frames fáciles se degrada justo ahí. Hay que medirlo: ocultar en validación frames con balón visible pero en zonas densas.

### 6. Segunda fuente de balón: retransmisión

- ATD de retransmisión: 164.488 frames, balón visible el 50,5 %, sin sincronizar con el táctico y con cortes y repeticiones.
- Sincronización propuesta (E6.2): pasar los jugadores de ambos ATD a coordenadas de campo y, por tramos, buscar el desfase que minimiza la distancia entre las nubes de jugadores (Chamfer o Hungarian). Descartar repeticiones (homografía `valid=0` o desfase incoherente) y propagar el desfase por tramos continuos de cámara.
- Ganancia potencial: frames donde la táctica pierde el balón y la retransmisión, en primer plano, lo ve. Hay que medir el solapamiento en local antes de invertir.

### 7. Reglas y operación

- **Etiquetar a mano para validar:** RULES.md permite "pequeñas cantidades" para entrenar o validar. El issue #8 (cuánto es "pequeño") sigue sin respuesta. Opción prudente: etiquetar *solo* para validación unos 5–10 min extra en local, con la misma convención que la verdad pública. Ojo: la verdad es una bandera de tracking y nuestras etiquetas humanas tendrán otra convención.
- **Gestión de datos:** mantener el bundle y las exportaciones reconstruibles fuera del repositorio público; publicar únicamente el código, la documentación y los resultados permitidos.
