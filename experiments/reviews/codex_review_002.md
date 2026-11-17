# Codex review 002 · Fase 1

Fecha: 2026-10-23. Rama: `Ale`. Commit revisado: `fee550c` (`fee550c` incluye el endurecimiento `5cd5ad2` y S1.1–S1.7). Se hizo `git fetch origin Ale`, se cambió a `Ale` y se ejecutó `git pull --ff-only origin Ale`.

## Resumen en lenguaje sencillo

Los lectores, el evaluador y la reproducción de resultados funcionan. He ejecutado los análisis con los datos originales: los cinco informes de Fase 0 y los siete scripts de Fase 1 terminan correctamente, y pasan las 22 pruebas actuales. El contraste ampliado de Sportec también pasa en los siete partidos.

Sin embargo, **no daría por cerrado el diagnóstico para tomar todas las decisiones de la siguiente fase**. Hay errores de código que las pruebas actuales no cubren y conclusiones más fuertes que la evidencia:

- **El orden «robos cortos > D > momento > balón» es provisional.** Los experimentos corrigen errores que se solapan. La cifra del balón no es un techo de mejora.
- **No se ha demostrado que Sportec y Metrica cambien la posesión igual.** La comparación de ventanas debilita la hipótesis inicial, pero tampoco demuestra su contraria.
- **Smart Tagging como pista suave sigue siendo una decisión razonable.** Hay que corregir una inversión de equipo omitida; eso no convierte sus tiempos en anclas precisas.
- Los datos justifican estudiar los huecos largos del balón. No permiten contar posesiones reales ocultas usando solamente el equipo del jugador más cercano.

No se ha cambiado código ni avanzado a Fase 2. Este informe es el único archivo añadido al proyecto.

## Hallazgos por gravedad

P1: resolver antes de usar el resultado para relegar una línea de trabajo. P2: corregir el cálculo o la interpretación antes de reutilizarlo como evidencia o como feature. No se han encontrado problemas P0.

### 1. [P1] S1.7 no establece un techo del balón ni una clasificación independiente por tipo de error

**Archivos:** `scripts/s1_7_oracles.py:80–81`; `src/eval/oracles.py:102–109`; `experiments/log/013_s1_7_margen_por_error.md:29–46`; `experiments/HALLAZGOS.md:28`.

La fila «ball fully visible» vuelve a ejecutar **el mismo baseline de proximidad**, que con balón visible siempre predice juego en Sportec. Su mejora de **+0,0983** mide el efecto en ese algoritmo y escenario; no limita lo que una trayectoria recuperada puede aportar a otro modelo o al combinarse con otras mejoras. Un contraejemplo con los mismos siete partidos y las mismas máscaras: aplicar `perfect_team` a las predicciones con y sin huecos aumenta la diferencia media a **+0,1897**. Esto no predice una ganancia alcanzable, pero refuta que +0,10 sea una cota general. El argumento de que el 56 % de los huecos son parones tampoco lo demuestra: S1.3 midió un indicador de lentitud, no la verdad de juego/parón del partido completo.

Además, `short_spells` pinta la etiqueta verdadera dentro de las posesiones cortas, incluso donde el baseline decía D. En la reproducción modifica **17.129 frames**, de los que **7.745 (45,2 %) eran D**: también está corrigiendo falsos parones y sus fronteras. Si se mantiene D intacto y se permite únicamente la corrección sobre frames que el baseline ya consideraba en juego, la mejora pasa de **+0,2062 a +0,1377**, por debajo del **+0,1643** de D perfecto. Es una sensibilidad a la definición del experimento, no una nueva clasificación definitiva.

Tampoco todas las posesiones seleccionadas quedan recuperadas: **198 de las 867** siguen sin una predicción del mismo equipo con IoU ≥ 0,3 después de pintarlas. Si ya están embebidas en una predicción larga del mismo equipo, escribir la misma etiqueta dentro no crea sus fronteras.

**Consecuencia:** ni las categorías son reparaciones independientes ni «balón al final» queda demostrado. La cautela de que las ganancias no suman es correcta, pero no resuelve estas diferencias de alcance.

**Recomendación:** conservar las cifras como intervenciones concretas sobre el baseline, eliminar `≤ +0,10` y la palabra «techo» cuando corresponda, describir exactamente qué repara cada intervención y comprobar efectos marginales/combinaciones en la degradación calibrada antes de fijar la prioridad. No es necesario abandonar M1 ni el detector de D para corregir esta interpretación.

### 2. [P2] S1.5 mide el evento más cercano y lo presenta como el primero

**Archivos:** `scripts/s1_5_flag_semantics.py:126–130`; `experiments/log/011_s1_5_semantica_banderas.md:31`; `experiments/log/011_s1_5_semantica_banderas.md:51`; `experiments/HALLAZGOS.md:26`.

`new_act[np.argmin(np.abs(new_act - f))]` selecciona el evento del equipo nuevo **más próximo** a la bandera dentro de ±3 s. No selecciona su primer evento. Entre los **1.934** cambios con evento candidato, ambos criterios difieren en **571** casos. La mediana publicada de **+5 frames** corresponde al más cercano; tomando literalmente el primer evento de esa misma ventana resulta **−6 frames**.

Ejemplo sintético: para un cambio en el frame 100 y eventos del equipo nuevo en 80 y 110, el script devuelve +10 como `new_first_action`, aunque el primero está a −20.

**Consecuencia:** la tabla y la comparación con el último pase del equipo antiguo no miden los conceptos que dicen medir. El +5 no demuestra por sí mismo que la bandera siga a la primera toma de control. La incertidumbre de alineación de los eventos agrava esa interpretación.

**Recomendación:** si se mantiene el cálculo, llamarlo «evento más cercano» y limitar la conclusión a proximidad temporal. Para estudiar primera toma de control se necesita definir la secuencia de posesión/eventos y su emparejamiento; sustituir simplemente `argmin` por el mínimo tampoco prueba que el evento pertenezca a esa posesión.

### 3. [P2] El control no exclusivo favorece siempre a A cuando ambos equipos cumplen el criterio

**Archivo:** `src/features/possession_rules.py:133–135`.

Con `exclusive=False`, el bucle escribe primero HOME y después AWAY sobre el mismo array. Si ambos equipos cumplen el radio y la duración, A sobrescribe H, aunque H esté más cerca. No hay una resolución explícita del duelo.

**Reproducción:** durante cuatro frames, distancias constantes H = 0,1 m y A = 1,9 m, con `r=2`, `m=3`, `exclusive=False`. Resultado: `[-1, -1, A, A]`. Intercambiar las identidades de los equipos no intercambia simétricamente la decisión.

**Impacto:** afecta a las **16 configuraciones no exclusivas** de la parrilla de control y a su futuro uso como feature. Los mejores ajustes publicados son exclusivos, por lo que este fallo no invalida directamente sus filas ganadoras.

**Recomendación:** definir una política de ambigüedad simétrica y comprobar la invariancia al intercambiar H/A.

### 4. [P2] Las señales de toque calculan velocidades a través del descanso

**Archivo:** `src/features/possession_rules.py:82–93`.

`Kinematics.from_match` comprueba la visibilidad en `t`, `t−3` y `t+3`, pero no que pertenezcan a la misma parte. Sportec concatena las dos partes sin el tiempo del descanso: la diferencia entre la última posición de una parte y la primera de la siguiente se interpreta como un desplazamiento de 0,12 s. Las reglas tampoco reciben una separación de partes para reiniciar su estado.

**Reproducción sintética:** balón inmóvil seis frames en x=0 en la primera parte y seis en x=20 en la segunda. Se generan seis muestras falsas con cambio de velocidad **166,7 m/s** y giro de **90°**. En los siete partidos reales aparecen cambios de velocidad artificiales de aproximadamente **49,7–497,7 m/s** junto al descanso; varias muestras tienen verdad H/A y entran en el dominio que evalúa S1.2.

**Impacto:** contamina las features y puede provocar un toque o un estado inicial de equipo ficticio. Su alcance temporal es pequeño respecto al partido: no explica por sí solo el acuerdo global de ~92 %.

**Recomendación:** exigir continuidad de parte y observaciones para las diferencias temporales, y definir explícitamente el reinicio de estado entre partes. Añadir un caso de dos partes a las pruebas.

### 5. [P2] S1.6 omite invertir `DEFENSIVE TRANSITION`

**Archivo:** `scripts/s1_6_smart_tagging.py:121–128`.

La inversión solo se aplica a códigos que empiezan por `DEFENDING`. `DEFENSIVE TRANSITION` no cumple esa condición, aunque el propio informe indica que debe invertirse y sus intervalos son gemelos de `ATTACKING TRANSITION` con el equipo contrario.

**Reproducción real:** las 24 instancias de `Defensive Transition` dan **40,7 %** de acuerdo con el último toque y **32,9 %** con la verdad pública. Con la inversión pasan a **59,3 %** y **67,1 %**, respectivamente, coincidiendo con su fase gemela. El agregado ponderado por instancias/frames pasa de **69,31 % a 69,78 %** frente al toque y de **64,78 % a 65,49 %** frente a la verdad pública.

**Impacto:** una fase lleva el equipo al revés y las tablas agregadas quedan ligeramente sesgadas. No cambia la recomendación de usar Smart Tagging como pista suave. El agregado cuenta varias veces los frames cubiertos por fases solapadas; describe acuerdo sobre observaciones fase-frame, no una predicción única por frame.

**Recomendación:** incluir explícitamente esta fase en el mapa de inversión y comprobar la igualdad de sus etiquetas de posesión con la fase atacante gemela.

### 6. [P2] S1.5 no permite descartar una diferencia semántica entre proveedores

**Archivos:** `experiments/log/011_s1_5_semantica_banderas.md:49–58`; `scripts/s1_5_flag_semantics.py:191–194`; `experiments/HALLAZGOS.md:7`.

Que el número de posesiones público esté en el percentil 20 de Sportec y que el 14 % de sus ventanas con D ≤ 30 % tenga una mediana tan larga demuestra **compatibilidad con variación entre tramos**. No descarta que también exista una diferencia entre proveedores. Son partidos distintos; Metrica Sample tiene etiquetas derivadas por reglas propias; las ventanas de cinco minutos cada treinta segundos se solapan y no equivalen a 1.205 observaciones independientes. El corte de D ≤ 30 % tampoco identifica causalmente que el poco D explique toda la diferencia.

**Consecuencia:** es correcto retirar la afirmación inicial de que Sportec es más rápido. Es excesivo sustituirla por «hipótesis descartada», «la diferencia era el tramo» y cerrar por ese motivo la posibilidad de adaptar duraciones. La formulación «no hay pruebas suficientes» que aparece en otra entrada de HALLAZGOS es más fiel a los datos.

**Recomendación:** mantener ambas explicaciones abiertas. Tratar los percentiles como diagnóstico descriptivo y comprobar sensibilidad/transferencia por partido cuando se evalúe el sistema; tampoco imponer a BAR–ATH el 36–46 % de D observado en otras fuentes.

### 7. [P2] La latencia seleccionada en público no demuestra que haga falta un desplazamiento global

**Archivos:** `scripts/s1_2_flag_rules.py:87–104`; `experiments/log/008_s1_2_reglas_bandera.md:49–60`; `experiments/HALLAZGOS.md:20`.

La parrilla selecciona el mejor ajuste de cada regla usando acuerdo y desfase en los mismos cinco minutos donde después presenta sus resultados. Eso es ajuste exploratorio, no validación independiente. El plan permite calibrar hasta tres parámetros físicos: las 77 combinaciones no son por sí solas una infracción, pero esa autorización no convierte la puntuación de selección en prueba de generalización ni permite ignorar la selección de familia y las decisiones posteriores.

En particular, la histéresis introduce retraso por construcción. Reejecutando `evaluate` en público: más cercano sin histéresis tiene desfase mediano **−8,5 frames** con 12 cambios emparejados; el ajuste elegido `k=25` tiene **+20 frames** con solo 8 emparejados. Parte del patrón «todas llegan tarde» depende de los filtros elegidos y del subconjunto que sobrevive al emparejamiento. No identifica un desfase común de Metrica que deba corregirse restando 0,6–0,8 s a toda transición.

**Recomendación:** mantener el desplazamiento global como candidato, no como necesidad ya demostrada. Congelar y registrar conjuntamente qué parámetros y elecciones se calibran con público; validar las reglas y el desplazamiento fuera de la muestra de selección cuando sea posible, y conservar los cinco minutos como comprobación limitada. Los desfases de eventos emparejados deben leerse junto con cambios perdidos y espurios.

### 8. [P2] «Al menos 36 cambios ocultos» no es un límite inferior de cambios reales

**Archivos:** `scripts/s1_3_ball_gaps.py:108–119`; `experiments/log/009_s1_3_huecos_balon.md:47`; `experiments/HALLAZGOS.md:23`.

El cálculo compara el equipo del jugador más cercano antes y después de un hueco. Que cambie no implica que cambie la posesión real: un pase entre compañeros puede hacer que en uno de sus extremos un rival sea momentáneamente el más cercano. S1.1 ya muestra errores de esa regla incluso cuando el balón se ve. Además, los dos frames observados están fuera del hueco: no localizan el supuesto cambio dentro de él. Que el equipo coincida tampoco descarta varios cambios internos.

**Consecuencia:** el ~43 % es una medida de desacuerdo de proximidad a través de los huecos, no «al menos ~36» posesiones reales ocultas. Tampoco debe sustituirse «lento» por D real ni «no lento» por juego confirmado al extrapolar fuera de la ventana pública. El indicador de lentitud detecta aproximadamente la mitad de los D públicos y no hay etiquetas completas para conocer su comportamiento en los huecos largos.

**Recomendación:** conservar el recuento como cambios de equipo más cercano en los extremos, con su denominador de casos observables, y mantener los nombres «indicador de parón»/«movimiento» en los agregados y en los parámetros de degradación.

## Comprobaciones y límites de esta revisión

Se ejecutó una copia exacta de los archivos versionados en `fee550c`, exportada con `git archive` a `/tmp/bcn-codex-review-002/repo`. Los datos crudos proceden de `<repo>/data/tactical` y `data_external`, enlazados a esa copia temporal. Se construyeron caches nuevas: no se reutilizaron las del análisis original. Se utilizó su entorno existente Python 3.11.16, NumPy 1.26.4, pandas 2.2.3 y kloppy 3.18.0.

| Comprobación | Resultado |
|---|---|
| `python -m pytest -q -p no:cacheprovider tests` | 22 pasan |
| `python -m pyflakes src tests scripts/s1_*.py` | Sin incidencias |
| `python scripts/check_official.py` | Pasa |
| `scripts/report_s0_2_bar_ath.py` a `scripts/report_s0_6_baseline.py`, cada uno con `--out` temporal | Los cinco pasan, incluidos sus nuevos controles |
| `scripts/s1_1_transition_anatomy.py` a `scripts/s1_7_oracles.py`, cada uno con `--out` temporal | Los siete terminan con código 0 |
| Sportec contra kloppy | 7 partidos, 40.113 frames muestreados y 875.129 posiciones de jugador; etiquetas idénticas, visibilidad idéntica y error máximo de posición de jugador ~3,67e−6 m |
| Reproducción de `oracles.csv` | Idéntico byte a byte al output local existente de S1.7 |
| Contraejemplos adicionales | Sesgo A, discontinuidad de partes, primero/más cercano, inversión defensiva y sensibilidad de oráculos reproducidos fuera del repositorio |

Los comandos, salidas y sondas de esta revisión están en `/tmp/bcn-codex-review-002/` (`runs/results.json`, `runs/`, `probes.py`, `oracle_probe.py` y `timing_probe.py`). Son auxiliares locales temporales, no archivos para entregar ni nuevos experimentos del proyecto. Los contraejemplos numéricos usan las mismas funciones y máscaras de S1.7; no son modelos entrenados ni promesas de mejora.

**Endurecimiento anterior:** `5cd5ad2` sí amplía el contraste a los siete partidos, devuelve error cuando fallan los controles añadidos y añade un trabajo de CI para pruebas sintéticas. No se repiten como pendientes los problemas ya resueltos.

**Detalles menores:** `experiments/registry.csv:14` registra 0,210300 para el baseline trasplantado, mientras el script y su CSV guardado dan 0,2098 (media sin redondear 0,20980497). Parece una transcripción: los principales deltas redondeados del informe sí se reproducen. La comprobación balón–jugadores de `experiments/log/008_s1_2_reglas_bandera.md:41` y varias cifras exploratorias no se guardan completas en el script; no he certificado esa supuesta exclusión del desfase. Menor vibración tampoco excluye un sesgo de posición constante.

**Veredicto:** base de lectura y puntuación verificada; diagnóstico útil, con correcciones pendientes antes de fijar como definitivas la equivalencia de proveedores, el desplazamiento temporal global y la clasificación de mejoras de S1.7. No se han modificado código, informes anteriores, parámetros del proyecto ni datos originales.
