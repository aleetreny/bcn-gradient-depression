# 013 · S1.7 Margen de mejora por tipo de error  (2026-10-23, commit 266b055)

Comando: `.venv311/bin/python scripts/s1_7_oracles.py`        Config: baseline smooth50; oráculos en `src/eval/oracles.py`        Semilla: n/a

## Resultado

Se arregla con la verdad **un solo tipo de error** de la predicción del baseline y se vuelve a puntuar. Escenarios:
- **público** BAR‑ATH (17 posesiones; el IC del baseline es 0,21–0,45, así que es ruidoso);
- **Sportec limpio** (7 partidos; el balón siempre se ve y el baseline nunca predice D);
- **Sportec con los huecos del ATD trasplantados:** la secuencia de visibilidad del balón del BAR‑ATH copiada sobre cada partido. Es lo más parecido a nuestro caso antes de S2.1; con ella el baseline predice un 32 % de D.

| Arreglo | público | Δ | Sportec limpio | Δ | Sportec + huecos ATD | **Δ** | folds que mejoran |
|---|---|---|---|---|---|---|---|
| baseline | 0,326 | — | 0,308 | — | 0,210 | — | — |
| equipo perfecto en frames en juego *(paraguas)* | 0,790 | +0,464 | 0,743 | +0,435 | 0,553 | **+0,344** | 7/7 |
| posesiones cortas (< 2 s) recuperadas *(también repinta D)* | 0,451 | +0,125 | 0,494 | +0,186 | 0,416 | **+0,206** | 7/7 |
| posesiones cortas recuperadas, **solo en frames en juego** *(añadido)* | 0,451 | +0,125 | 0,494 | +0,186 | 0,347 | **+0,138** | 7/7 |
| D perfecto | 0,400 | +0,074 | 0,480 | +0,172 | 0,374 | **+0,164** | 7/7 |
| balón siempre visible (sobre el baseline) | — | — | — | — | 0,308 | **+0,098** | 7/7 |
| balón siempre visible, **bajo equipo perfecto** *(añadido; frente a equipo perfecto con huecos)* | — | — | — | — | 0,743 | **+0,190** | 7/7 |
| momento de los cambios perfecto (±3 s) | 0,381 | +0,056 | 0,398 | +0,090 | 0,290 | **+0,081** | 7/7 |
| sin posesiones inventadas | 0,400 | +0,074 | 0,229 | −0,079 | 0,234 | +0,024 | 6/7 |
| D perfecto + momento perfecto | 0,449 | +0,123 | 0,602 | +0,294 | 0,474 | +0,264 | 7/7 |

Gráfico: `experiments/outputs/013/oracles.png`. Tabla: `oracles.csv`.

## Cómo leerlo (con cautela)

- **"Equipo perfecto" no es un tipo de error, sino un paraguas:** corrige el equipo en todos los frames en juego y con ello arregla también el momento, los robos cortos y los cambios inventados. Es el techo de "acertar quién tiene el balón" (+0,34), no un elemento más de la lista.
- **"Sin posesiones inventadas" no está bien definido:** al quitar una posesión falsa se fusiona con la anterior, lo que alarga tramos que sí emparejaban (en Sportec limpio empeora). No se usa para priorizar.
- ~~**"Balón siempre visible" está probablemente sobreestimado**~~ *(corregido abajo: no es ni techo ni cota)*: los huecos trasplantados caen al azar respecto a los parones de Sportec.
- Los arreglos no se suman exactamente: D + momento da +0,264, frente a +0,245 sumando los dos por separado.

## Hallazgos

1. *(Matizado abajo.)* **La mayor parte de la nota está en acertar quién tiene el balón** (+0,34 en el escenario realista), por delante del D (+0,16) y de tener el balón siempre visible (+0,10 sobre el baseline).
2. *(Corregido abajo: depende de si se repinta D.)* **Dentro de eso, lo que más vale es no perder las posesiones cortas** (+0,21; 7/7 partidos): recuperar solo las de menos de 2 s ya da más que un D perfecto. Confirma el "un robo corto perdido cuesta triple" del doc 04 y el hallazgo de S0.5.
3. **Arreglar solo el momento de los cambios da relativamente poco** (+0,08), porque el baseline ni siquiera detecta muchos cambios: el oráculo solo puede mover los que existen. Esto **matiza S1.1/S1.2**: primero hay que **detectar** bien los cambios (duelos, robos cortos) y después clavar su momento.
4. **D vale lo suyo** (+0,16 en Sportec), aunque en público solo +0,07 porque ese tramo tiene poco D (S1.5). Con un partido completo con 35–45 % de D, en el BAR‑ATH pesará más que en los públicos.
5. ~~Rellenar el balón queda por detrás (≤ +0,10, seguramente menos)~~ *(corregido abajo)*.

## Decisión

~~**Orden de la Fase 3 por ganancia potencial**~~ *(sustituido abajo por un orden provisional)*:
1. **Detectar cambios de equipo y posesiones cortas:** features de toque y control (M1), cabeza de transición (M3) y D corto como separador (M6).
2. **Detector de D:** señales de S1.1/S1.3/S1.6 (hueco largo, jugadores lentos, balón alejándose o fuera, tiempo sin etiquetar) (M4a/M4b).
3. **Momento de los cambios:** decodificador con desplazamiento temporal (S1.2) y M3.
4. **Balón que falta:** M2b/M2c/M13, solo si después de lo anterior sigue mereciendo la pena. M2a ya se descartó en S1.3.

**Fase 1 terminada.** Siguiente: Fase 2 (S2.1, degradación calibrada de Sportec), tras la revisión de Codex.

## Corrección tras la revisión de Codex 002 (2026-10-23)

Revisión: [codex_review_002](../reviews/codex_review_002.md), punto 1, más el detalle menor del registro.

- **Las cifras son intervenciones sobre el baseline, no techos ni tipos de error independientes.** Cada arreglo cambia una predicción concreta y los arreglos se solapan.
- **Posesiones cortas:** el arreglo original modificaba 17.129 frames, de los que 7.745 (45 %) eran D en el baseline, así que también corregía parones falsos. **Si solo se repintan frames que el baseline da en juego, la mejora baja a +0,138**, por debajo del **+0,164** de D perfecto. Además, 198 de las 867 posesiones cortas tratadas siguen sin emparejar después del arreglo (pintar la misma etiqueta dentro de un tramo largo del mismo equipo no crea fronteras). Desde S1.7, `src/eval/oracles.py` incluye la variante `short_spells_alive_only` y un contador de recuperación.
- **Balón:** +0,098 mide el efecto en el baseline de proximidad. Con el arreglo de equipo perfecto, pasar de los huecos al balón completo vale **+0,190**. Por tanto, **+0,10 no es una cota** de lo que puede aportar recuperar el balón a otro modelo. El "56 % de los huecos en parones" de S1.3 usa un indicador, no la verdad.
- **Registro:** la fila 013 ponía 0,2103 para el baseline trasplantado; el valor del script es **0,2098**. Corregido.

**Decisión revisada (orden provisional de la Fase 3):** tres líneas con ganancias **del mismo orden** en el baseline, todas con 7/7 partidos:
- detectar los cambios de equipo y las posesiones cortas (+0,14 a +0,21 según la definición);
- D (+0,16);
- recuperar el balón (+0,10 a +0,19 según el modelo).

El momento puro de los cambios (+0,08) va detrás. **Ninguna línea se descarta ni se relega todavía.** El orden se fijará con efectos marginales y combinaciones en la degradación calibrada de Sportec (S2.1), con modelos y no solo con el baseline. M1 y el detector de D siguen siendo lo primero que se construye, porque son baratos y los necesita cualquier sistema (Fase 2), no porque esté demostrado que valgan más.
