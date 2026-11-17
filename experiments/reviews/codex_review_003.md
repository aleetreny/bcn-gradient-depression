# Codex review 003 · v1 y Fase 2

Fecha: 2026-10-23. Rama revisada después de `git pull --ff-only origin Ale`: `6cbdce879241df33dfbe406ce222f6d185c840cc`. El tag anotado `v1` apunta a `cb55e1a5cd4af111158b69260ca368ead786e4ab`; los commits posteriores añaden S2.6 y su documentación, sin cambiar el algoritmo congelado.

## Resumen en lenguaje sencillo

**La v1 funciona y he reproducido su fichero exacto dos veces**, con el SHA-256 publicado. El 0,519 público es real para esos cinco minutos. También se reproducen los cálculos publicados de Sportec. Sin embargo, **el 0,666 no es una evaluación completamente anidada de toda la cadena**, y la simulación todavía hace que los parones sean más fáciles de reconocer que en el ATD.

- **No he encontrado una lectura directa de las etiquetas en la degradación ni en las features.** Cambiar las etiquetas por ceros, barajarlas o eliminarlas deja idénticos los datos degradados. El entrenamiento del GBM sí separa correctamente el partido de prueba y mantiene juntas sus variantes durante la calibración.
- **Sí hay una vía indirecta de información en la selección del decodificador:** para ajustarlo se usan predicciones de otros partidos producidas por GBM que fueron entrenados con el partido exterior de prueba.
- **El arreglo del balón congelado deja otra huella artificial:** durante los parones, el balón simulado se coloca casi siempre pegado a un jugador. Sus distancias, velocidades y visibilidad siguen teniendo una relación con D mucho más fuerte que en los públicos.
- **Detectar cambios y posesiones cortas es una prioridad razonable.** No está demostrado que los únicos problemas pendientes de D sean los apagones ni que recuperar el balón tenga que esperar al final.
- La construcción desde una copia limpia necesita un JSON no versionado que el comando de v1 no genera. Una vez regenerado con S1.3/S1.4, las dos construcciones reproducen el hash esperado.

La v1 puede conservarse como reserva reproducible. Antes de usar su LOMO como juez principal para nuevas mejoras, corregiría el anidado y sometería el detector de D a controles de transferencia más exigentes. No he cambiado código ni implementado Fase 3.

## Hallazgos por gravedad

P1: afecta a la validez de la evaluación o permite un atajo importante en el entorno simulado. P2: corregir antes de reutilizar el resultado como evidencia, configuración o procedimiento reproducible. No se han encontrado problemas P0.

### 1. [P1] El LOMO del decodificador reutiliza OOF globales y no aísla el partido exterior en toda la cadena

**Archivos:** `scripts/s2_4_decoder.py:75–85`; `scripts/s2_3_frame_models.py:70–83`; `src/models/gbm.py:47–70`; `experiments/log/018_s2_4_decodificador.md:10`.

S2.4 carga una sola predicción OOF por partido, generada en S2.3. Para un partido exterior T, Optuna excluye T de su función objetivo, pero las probabilidades de cada partido interior J vienen de un GBM entrenado con todos los partidos salvo J: **ese GBM incluye T y sus etiquetas**, también a través de sus calibradores. Excluir la fila T de Optuna no elimina esta dependencia.

Ejemplo concreto: al evaluar exteriormente `J03WMX`, el ajuste del decodificador usa `J03WN1.npy`. Ese fichero fue producido por un GBM entrenado con `J03WMX`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ` y `J03WR9`.

**Lo que sí está bien:** la predicción base de T fue producida sin entrenar con T; las variantes de un mismo partido se agrupan en la calibración isotónica. No es una mezcla directa de frames de entrenamiento/prueba en S2.3. El problema está en el segundo nivel, donde se seleccionan los parámetros del decodificador.

**Consecuencia:** he reproducido **0,666002 ± 0,029282**, pero ese número no certifica el rendimiento de una cadena completamente anidada. El tamaño y dirección concretos del sesgo no se pueden deducir restando unos puntos al resultado actual.

**Recomendación:** para cada T exterior, generar OOF interiores usando únicamente los otros seis partidos: la predicción de J debe entrenarse y calibrarse sin T ni J. Ajustar el decodificador con esas predicciones, entrenar el modelo exterior con los seis y evaluar T una sola vez. La caché debe identificar tanto el fold exterior como el interior. No he implementado esta corrección ni presentado una nueva cifra como si fuera el LOMO corregido.

### 2. [P1] La degradación v3 sustituye el balón congelado por una señal de D artificialmente ligada a un jugador

**Archivos:** `src/data/degrade.py:118–139`; `src/data/degrade.py:266–280`; `src/features/v1.py:95–115`; `experiments/log/015_s2_1_degradacion.md:79–95`.

El indicador `rest` sigue calculándose sobre el balón original de Sportec, cuya congelación identifica casi perfectamente D. Ahora se usa para generar una trayectoria: se elige el jugador más cercano al principio del tramo y se coloca el balón sobre su trayectoria, con desplazamiento de **0,6 m** y ruido acumulado recortado por coordenada a **±0,45 m**. Se elimina la velocidad exactamente cero, pero se introduce una relación muy particular balón–jugador. Si el jugador elegido permanece visible, la distancia a ese jugador queda acotada aproximadamente por `0,6 + sqrt(2) * 0,45 = 1,24 m`.

**Comprobación con las features reconstruidas desde datos crudos, semilla 0:**

| Señal condicionada a la verdad, usada solo para esta auditoría | Siete partidos Sportec degradados | BAR–ATH público |
|---|---|---|
| Balón visible durante D | 13,8–22,3 % | 66,4 % |
| Distancia mediana del balón al jugador más cercano durante D visible | 0,56–0,74 m | 2,89 m |
| D visible con algún jugador a ≤ 1,24 m | 91,5–97,0 % | 32,0 % |
| Velocidad mediana del balón durante D visible | 1,05–1,32 m/s | 2,69 m/s |
| AUC de la velocidad del balón por sí sola, en muestras con velocidad conocida | 0,867–0,912 | 0,686 |

Un pequeño control con seis features de balón, 100 árboles y validación por partido alcanza AUC **0,940** en este Sportec y **0,750** en los públicos. Es una sonda diagnóstica, no otra estimación del rendimiento de v1. Las features reconstruidas de prueba y del ATD coinciden exactamente con las caches del análisis original, de modo que el contraste no procede de leer versiones distintas de features.

**No es una lectura directa de etiquetas por `degrade`.** Es una transformación condicionada por un artefacto del proveedor que está muy asociado a ellas. La prueba «no mira `labels`» es necesaria, pero no detecta esta clase de atajo. Ajustar las estadísticas marginales del balón y la visibilidad en el indicador de lentitud tampoco garantiza igualar estas relaciones condicionadas a D.

**Consecuencia:** hay evidencia concreta de facilidad artificial residual para D. No demuestra que todo el salto 0,997→0,953 provenga de este mecanismo, ni permite estimar el rendimiento del resto del BAR–ATH a partir de cuatro parones públicos. Sí impide tratar el LOMO actual como una validación convincente de la transferencia de D.

**Recomendación:** contrastar distribuciones y transiciones condicionadas a juego/parón donde haya verdad; estudiar visibilidad, distancia balón–jugador y movimiento relativo, además de velocidad. Evaluar variantes que rompan esta huella y controles por grupos de features. Usar las etiquetas para auditar el simulador, manteniéndolas fuera de la transformación que genera los ejemplos. Hasta entonces, las mejoras de D en Sportec necesitan una reserva explícita en el tablero.

### 3. [P2] S2.6 no demuestra que D esté resuelto salvo apagones ni que el balón deba ir al final

**Archivos:** `experiments/log/021_s2_6_errores_v1.md:31–51`; `src/eval/errors.py:113–116`; `evaluation/possession_metrics.py:208–225`; `experiments/RESULTADOS.md:18–24`.

«4/4 sin errores» significa que ninguno de los cuatro tramos D es predicho **mayoritariamente** como juego y no se inventan otros tramos según ese control. No significa fronteras exactas ni ausencia de errores de D.

En el CSV reproducido hay **1.469 frames verdaderos de D** en público. V1 marca correctamente 1.097, pierde **372 (25,3 %, 14,88 s)** y marca D en otros 23 frames que eran juego. El F1 de la clase D es **0,8474**. El **0,9682** del tablero es la métrica oficial `alive_dead_f1`, con **juego como clase positiva**; no es el F1 de D ni un promedio equilibrado de ambas clases. La métrica oficial está calculada correctamente: el problema es usarla como confirmación suficiente de que D apenas necesita trabajo.

Los oráculos sí muestran que arreglar el equipo y las posesiones cortas cambia mucho el spellF1 de estas predicciones. Es razonable empezar por M3/M1/modelos temporales. Sin embargo:

- D perfecto produce un efecto marginal pequeño **en estas predicciones y muestras**, con las limitaciones de transferencia del punto 2. Eso no demuestra «D: solo apagones» en el partido completo.
- S2.6 **no mide ninguna intervención de recuperación del balón sobre v1**, ni su efecto marginal después de mejorar equipo/D. No hay un experimento nuevo que justifique relegar esa línea respecto a todas las demás.
- Las reparaciones siguen siendo intervenciones que se solapan; su orden no representa una clasificación causal de métodos que todavía no se han construido.

**Recomendación:** sustituir la conclusión tajante por prioridades provisionales. Mantener cambios/posesiones cortas como primera apuesta bien motivada; mantener abiertos D fuera de apagones y recuperación del balón hasta medirlos en un simulador más fiable y con el modelo actual. Añadir precisión/recall/F1 de D y errores de fronteras al control, conservando también la métrica oficial.

### 4. [P2] El diagnóstico de S2.6 vuelve a usar parámetros del decodificador ajustados con los siete partidos

**Archivos:** `scripts/s2_6_v1_errors.py:46–54`; `scripts/s2_4_decoder.py:94–106`; `experiments/log/021_s2_6_errores_v1.md:3–18`.

S2.6 decodifica cada predicción OOF con `configs/decoder_v1.json`. Esa configuración se seleccionó maximizando el score en **los mismos siete partidos**, incluido el que se está puntuando. Las probabilidades son OOF del GBM, pero la predicción completa utilizada para analizar los errores no queda fuera del ajuste del decodificador. Esto explica la base **0,668526** del diagnóstico, distinta del **0,666002** de la evaluación por fold.

He repetido los oráculos usando los parámetros de decodificador registrados para cada fold: base **0,666002**, mejora de cortas solo en juego **+0,121878**, de momento **+0,065154** y de D **+0,014675**. Los órdenes de magnitud se mantienen; esta corrección aislada no destruye el indicio a favor de las posesiones cortas. Tampoco corrige el problema de los OOF de segundo nivel del punto 1.

**Recomendación:** guardar las predicciones completas de prueba de cada fold exterior y usarlas para S2.6 y los paneles de Sportec. Reservar la configuración final, entrenada con todos los partidos, para generar BAR–ATH. Si se conserva el diagnóstico actual, etiquetarlo como análisis con decodificador reajustado sobre los mismos partidos.

### 5. [P2] `make_submission_v1.py` no construye v1 desde una copia limpia con solo los datos crudos y las configs versionadas

**Archivos:** `src/data/degrade.py:39–40`; `src/data/degrade.py:74–75`; `src/data/degrade.py:151–153`; `scripts/make_submission_v1.py:39–47`.

La configuración versionada no contiene los histogramas y tasas empíricas usados por la degradación. `cfg.params` queda vacío y `load_params()` exige `local_data/degradation_params.json`, que no está en Git. El comando de construcción no lo genera ni documenta su preparación en su punto de entrada.

**Reproducción:** exportar `6cbdce8` con `git archive`, enlazar `data/tactical` y `data_external` originales y ejecutar el comando anunciado de v1 sin caches. Termina con **`FileNotFoundError: local_data/degradation_params.json`**. No faltaban los partidos ni las dependencias.

Ejecutando antes `scripts/s1_3_ball_gaps.py` y `scripts/s1_4_slot_anatomy.py` se genera el requisito. Después, con features reconstruidas, las dos construcciones sí dan:

`4491ebe4408a769c64a7625c866a67dc1275302df6eccd90f9a8ffd0f2ddf020`

**Recomendación:** incluir la generación determinista de ese agregado en la construcción, o documentar y verificar explícitamente el requisito y su hash. La repetibilidad está comprobada una vez preparados los agregados; el comando actual no es por sí solo el flujo completo desde crudos que anuncia.

### 6. [P2] Las caches de features no identifican el contenido de los datos ni de la configuración

**Archivos:** `src/features/v1.py:221–230`; `scripts/s2_3_frame_models.py:55–66`; `scripts/make_submission_v1.py:40–47`.

Las claves se forman con el nombre del partido, la versión global de degradación, el **nombre de archivo** de la config y el nombre de la variante. No incluyen el contenido de la config, el JSON de tasas/histogramas, la versión del código de features ni una identidad verificable de los datos. `cached_features` devuelve el parquet existente sin validar nada de ello.

Así, modificar un parámetro manteniendo `degrade_v2.json`, o usar otra config con el mismo nombre en otro directorio, puede dejar las features anteriores. La degradación se calcula con los parámetros nuevos pero el modelo consume las features viejas; S2.3 incluso calcula el baseline sobre la degradación nueva. Dos construcciones seguidas pueden coincidir al byte mientras ignoran el cambio solicitado.

**Contraejemplo ejecutado fuera del repositorio:** con una clave del formato usado por la aplicación, cambiar en +10 m las posiciones del balón y volver a llamar a `cached_features` devuelve la tabla anterior; en el frame sintético 100, `ball_x` sigue en **46,242867** mientras `build_features` sobre los datos nuevos da **56,242867**.

**Límite de este hallazgo:** no he encontrado contaminación por caches antiguas en las cifras actuales. Las features de prueba reconstruidas coinciden con las originales, las siete probabilidades OOF reentrenadas son idénticas elemento a elemento y el CSV coincide con el hash congelado.

**Recomendación:** identificar y validar inputs/configs/código en la caché, y proporcionar una forma de reconstruir sin ella. Esto debe resolverse antes de confiar en nuevas ablaciones cambiando parámetros, aunque la v1 congelada actual se haya reproducido correctamente.

## Qué se ha ejecutado y qué queda sin certificar

Se usó una copia temporal exacta de los archivos versionados en `6cbdce8`, bajo `/tmp/bcn-codex-review-003/repo`. Los datos originales se enlazaron desde `<repo>/`. Se reconstruyeron las caches de lectores y de features en la copia temporal. Entorno: Python 3.11.16, NumPy 1.26.4, pandas 2.2.3, LightGBM 4.7.0, scikit-learn 1.9.1 y numba 0.60.0.

| Comprobación | Resultado |
|---|---|
| `python -m pytest -q -p no:cacheprovider tests` | 36 pruebas pasan |
| `python -m pyflakes src tests scripts/s2_*.py scripts/make_submission_v1.py` | Sin incidencias |
| `scripts/check_official.py` | Pasa |
| Construcción de v1 en copia limpia sin agregado previo | Falla con el JSON ausente; punto 5 |
| S1.3 y S1.4 como preparación | Pasan; generan los agregados en la copia temporal |
| `scripts/s2_2_features.py` | Pasa; features nuevas, 160 columnas |
| `scripts/s2_3_frame_models.py` | Pasa; siete folds y modelo completo reentrenados; w=12, spellF1 0,6289, AUC vivo 0,9973 |
| `scripts/s2_4_decoder.py --trials 120` | Pasa; 120 pruebas en cada fold exterior y 120 finales; reproduce parámetros finales y 0,666002 |
| `scripts/s2_5_v1_report.py` y `scripts/s2_6_v1_errors.py` | Pasan; reproducen score público 0,518919 y diagnóstico |
| `scripts/s2_1_degradation.py --config configs/degrade_v2.json` | Termina correctamente: baseline 0,298121 y AUC adversario de calidad 0,7528; tres semillas, sin recalibrar ni cambiar ninguna config |
| `scripts/make_submission_v1.py`, dos veces tras preparar los agregados | Ambas pasan, mismo SHA-256 completo que el publicado |
| Independencia de etiquetas de la degradación v3 | `labels=None`, todo H y etiquetas barajadas dejan idénticos P/B, slots, mitades y dirección |
| Viterbi frente a enumeración exhaustiva independiente | 20 casos aleatorios pequeños: mismo óptimo entre todas las secuencias con duraciones válidas |
| Control de caches/OOF | Features de prueba de los siete partidos y ATD idénticas a las originales; siete OOF reentrenados con diferencia máxima 0 |

S2.4–S2.6 se ejecutaron en paralelo al reentrenamiento leyendo inicialmente los OOF guardados del análisis original. Después se verificó que los siete OOF nuevos eran idénticos, por lo que esas ejecuciones corresponden también a las predicciones reconstruidas. S2.5/S2.6 leyeron el CSV original; las dos construcciones nuevas reproducen el mismo hash.

Los comandos, salidas y sondas se conservan temporalmente en `/tmp/bcn-codex-review-003/`: `runs/results*.json`, `runs/`, `label_independence_probe.py`, `shortcut_probe.py`, `conditional_probe.py`, `mini_signal_probe.py`, `decoder_bruteforce.py`, `oracle_fold_probe.py` y `cache_probe.py`. No forman parte de los entregables ni son nuevas implementaciones del proyecto.

**Límites:** no se ha implementado ni ejecutado una nueva validación completamente anidada; por eso no se ofrece una cifra de LOMO «corregida». Tampoco hay verdad del resto del BAR–ATH para cuantificar el optimismo allí. Las ventanas centradas y las observaciones futuras de tracking son admisibles en este problema offline; no las he tratado como fuga temporal. No he encontrado Optuna ajustando los cinco parámetros del decodificador contra la verdad pública: los ajusta en Sportec. Los públicos ya han servido para orientar el desarrollo y detectar problemas, por lo que sus cinco minutos no constituyen un test independiente nuevo de generalización.

**Observación adicional sobre plausibilidad:** `scripts/s2_5_v1_report.py:89–92` cuenta un acierto si aparece cualquier D en `[inicio de SET PIECES − 10 s, fin de etiqueta]`. Una predicción siempre D también obtendría 100 %. El 98 % informado es cobertura de esos intervalos amplios, no precisión de detección de saques ni validación de sus fronteras.

**Veredicto:** v1 reproducida y útil como reserva; lectura directa de etiquetas no encontrada; evaluación anidada incompleta y huellas sintéticas de D aún presentes. El orden de Fase 3 tiene apoyo para estudiar cambios/posesiones cortas, pero las exclusiones de D fuera de apagones y la relegación del balón no quedan demostradas. Solo se ha añadido este informe; no se han cambiado código, configs, resultados anteriores ni datos originales.
