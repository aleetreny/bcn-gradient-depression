# Codex review 004 · Fase 3, v1.1, v2 y v3

Fecha: 2026-10-24. Rama `Ale`, después de `git pull --ff-only origin Ale`: `b3041caa806f165b27af3e61bbeafdd089f4d1ee`. Tags: `v1.1` → `2f872a1414a5422ecdefad09bb9ba5f9c1344547`; `v2` → `943f8fe969d86a426137831a5cea315b86db0e3d`; `v3` → `662ddfb1fe9c4ce76bd3002d5fd607a302746536`. Entre el tag v3 y HEAD solo cambia documentación.

## Resumen sencillo

El reparto de partidos para entrenar el GBM, M3, la TCN y ajustar el decodificador está bien corregido. La degradación y las features nuevas no leen las etiquetas. Hay, sin embargo, una excepción anterior al entrenamiento: **el simulador se eligió usando etiquetas de los públicos y de dos partidos que después se presentan como prueba**. Por eso los números no son una comprobación independiente de toda la cadena.

La v3 es una candidata razonable, con una mejora pequeña en Sportec y señales favorables en Sample. Eso no demuestra que sea superior en el BAR-ATH completo. **Sample ayuda a contrastar algunas señales de equipo, pero sus parones y sus etiquetas no son los del reto.** Tampoco es cierto que v3 gane a todas las alternativas en los tres jueces.

Los 39 tests pasan. He reproducido v1.1 al byte, los decodificadores de F3.1/F3.3/F3.7 y varias comprobaciones de modelos. Para v2/v3 he verificado los CSV existentes y sus réplicas; **no he completado nuevas construcciones CPU**, siguiendo la indicación de evitar repetir ejecuciones largas.

No he cambiado código ni avanzado de fase. Los experimentos de revisión se han ejecutado fuera del checkout.

## Hallazgos, ordenados por gravedad

### 1. [P1] La selección del simulador usa los públicos y dos partidos exteriores: el anidado no cubre toda la cadena

**Archivo:línea:** `scripts/s2_1b_dead_ball.py:88-109`; partidos de calibración en `scripts/s2_1b_dead_ball.py:37`.

El script carga la verdad pública, selecciona sus D y calcula el objetivo de calibración (88-91). Después compara 27 configuraciones, calcula sus estadísticas **dentro de los D reales de J03WMX y J03WQQ** (101), minimiza la pérdida contra el objetivo público (103-106) y guarda el ganador como `configs/degrade_v3.json` (109). Por tanto, las etiquetas intervienen en **elegir** las constantes; no se usan solo para auditar, como dicen la cabecera y la procedencia de la configuración.

Esto tiene dos consecuencias diferentes:
- El público es ahora un conjunto de desarrollo: el AUC 0,976 y sus scores posteriores no son una prueba independiente de transferencia. No significa que esté prohibido usarlo; cambia lo que se puede concluir de él.
- Cuando J03WMX o J03WQQ son el partido exterior, su verdad ya influyó en el simulador común. Los modelos y el decodificador sí los excluyen, pero la cadena completa no.

Comprobación: recomputando el objetivo público y la pérdida sobre la parrilla guardada se obtiene el ganador publicado (0,7 / 3,5 / 1,5). Al retirar el último de los cuatro parones públicos, el ganador cambia a 0,7 / 3,5 / 2,5. Es una dependencia efectiva del etiquetado, no solo una lectura sin efecto. Esta sonda reutiliza las estadísticas simuladas de la parrilla; no reentrena los sistemas.

**Qué corregir:** declarar el público como desarrollo y reservar una evaluación sin usar para ajustar el simulador; para un LOMO de toda la cadena, ajustar este componente únicamente con datos interiores o fijarlo con una fuente de calibración separada. Hasta medirlo, no atribuir a esta vía una inflación concreta de 0,715 ni dar por demostrado que sea grande o pequeña.

### 2. [P2] El Sample del tablero evalúa modelos MPS; el fichero reproducible usa modelos CPU distintos

**Archivo:línea:** `scripts/s3_8_candidates.py:98-106`, `scripts/s3_5_sample_transfer.py:62-64`, frente a `scripts/make_submission.py:100-105`; `experiments/log/032_f3_7_control_anidado.md:51-54`.

Las pruebas de Sample usan los modelos completos entrenados en MPS. La construcción final fuerza CPU y vuelve a entrenar la TCN. No se conserva el mismo estimador por el mero hecho de usar semilla 99. La propia tabla lo muestra en público: la variante `ens_mean` de MPS da 0,5412 y F1 de D 0,8242, mientras el CSV CPU da 0,5697 y 0,865. El tablero combina el Sample de un estimador con el público de otro.

Recalculé los scores desde los artefactos existentes: las predicciones MPS y el CSV v3 CPU difieren en **2664 frames**, de los cuales **251** están etiquetados en público. No es solo redondeo de la tabla. No he medido Sample con los pesos CPU finales y no afirmo que su transferencia sea peor; el problema es presentar como medido para ese estimador un 0,803 obtenido con otro.

**Qué corregir:** distinguir ahora las filas MPS/CPU y dejar la transferencia CPU como pendiente. Si los pesos CPU están guardados, evaluarlos directamente en Sample; si no, conservarlos en la siguiente construcción necesaria. No es preciso repetir una construcción larga para corregir la documentación. El LOMO con TCN en MPS sigue siendo evidencia para esa receta; no es una comprobación del entrenamiento CPU concreto. Determinismo del constructor y validez de la evaluación son controles diferentes.

### 3. [P2] La caché por contenido se arregló en las features, pero no en predicciones y modelos

**Archivo:línea:** `scripts/s3_7_nested_full.py:86-88` y `scripts/s3_8_candidates.py:86-88`. Mismo patrón en `scripts/s2_4b_nested_lomo.py:74-76`, `scripts/s3_1_transition_head.py:92-94`, `scripts/s3_3_ensemble_nested.py:66-69`, `scripts/s3_5_sample_transfer.py:46-50`.

Una predicción existente se acepta solo porque existe su nombre bajo `nested_<nombre_config>_<features>`. No se valida el contenido de la configuración, los datos, las etiquetas, las features, los parámetros del modelo, su semilla o la implementación. Los `models.pkl` tienen el mismo problema. Editar una configuración conservando el nombre puede reconstruir correctamente las features y, aun así, ajustar el decodificador sobre probabilidades de la configuración anterior. Una reanudación parcial puede mezclar ejecuciones.

Reproducción fuera del proyecto: ejecuté el `main()` real de F3.7 con siete partidos pequeños y estimadores que registran las llamadas. La primera pasada hizo 147 entrenamientos, con exclusión correcta del partido exterior y del objetivo. Cambié todas las features y una constante del simulador sin cambiar el nombre de configuración. La segunda pasada hizo **cero entrenamientos** y aceptó los 147 ficheros antiguos.

**Qué corregir:** añadir identidad de los insumos y del entrenamiento a las cachés de predicciones/modelos, o validar un manifiesto antes de reanudar. Como control de las cachés actuales, reentrené un GBM interior (T=J03WMX, J=J03WN1, semilla 1000), una M3 exterior (J03WMX, semilla 0) y una TCN exterior (J03WMX, semilla 0, MPS): sus probabilidades coinciden exactamente con las guardadas. Además, las 54 tablas parquet de features reconstruidas que comparé coinciden al byte con las originales. Este fallo demuestra una vía para resultados obsoletos; no demuestra por sí solo que las cifras actuales estén obsoletas.

### 4. [P2] La aceptación de v3 cambia la regla y exagera el acuerdo entre jueces

**Archivo:línea:** `experiments/log/032_f3_7_control_anidado.md:33-35` y `:58`; regla en `experiments/RESULTADOS.md:9`.

La excepción a +0,01 está declarada, pero no hay una nueva regla cuantitativa que explique cuánto pesa Sample o cuándo basta para compensarlo. La mejora es aproximadamente **+0,00544** en siete partidos y una sola semilla de evaluación: J03WOH pierde 0,0205 y J03WPY mejora solo 0,0001. Seis signos positivos no demuestran que el efecto sea estable frente a semillas, degradaciones o dominios. La comparación también cambia las semillas interiores y añade las 60 features de cambio al GBM, además de las 24 de control (`s3_3_ensemble_nested.py:82`, `s3_7_nested_full.py:50-55,107`), por lo que no aísla únicamente el efecto de las 24 features.

**No gana en los tres jueces.** En la propia tabla del log 032, `ens_m3` gana en Sample (0,809 > 0,803), el GBM con control gana en público (0,584 > 0,570 de la v3 CPU) y v3 gana en LOMO. Tampoco hay un mismo orden completo entre LOMO y Sample: el GBM con control queda por debajo de v2 en LOMO y por encima en Sample.

Sample es útil como contraste auxiliar, pero su visibilidad separa D con mucha más facilidad: en los tres partidos solo se ve el balón en **16,2 %, 3,9 % y 0,24 % de D**, frente al **66,4 %** público. La regla «sin balón = D» ya acierta **88,7 %, 93,2 % y 95,7 %**. Sus etiquetas salen de reglas de eventos, no de la bandera objetivo (`src/data/metrica_sample.py:96-159`). No equivale a tener tres partidos más con la verdad del reto.

Control adicional sobre los modelos MPS: excluyendo D y los tres segundos a cada lado, la mejora media pasa de +0,0182 a aproximadamente **+0,0076**, con mejora en 2/3 partidos. Este score sobre posesiones recortadas es un diagnóstico, no un nuevo benchmark oficial. Apoya una señal útil de equipo, pero no que todo el +0,019 se transfiera al ATD.

**Qué corregir:** conservar v3 como candidata provisional y explicitar el compromiso entre jueces, separando la confirmación de las features de control de la elección del ensemble. Mantener v2 y el GBM sencillo como alternativas; retirar la afirmación de victoria en los tres jueces. Las pocas posesiones por minuto en el partido sin verdad son una alerta de plausibilidad, no prueba suficiente para inferir su número real o su principal fuente de error.

### 5. [P2] La TCN normaliza con toda la secuencia: su contexto real no son los 20 segundos documentados

**Archivo:línea:** `src/models/tcn.py:36`; entrenamiento con recortes en `:109-118`, inferencia por parte completa en `:138-140`.

`GroupNorm(8, c)` recibe tensores `[batch, canales, tiempo]` y calcula estadísticas también sobre el tiempo. Aunque las convoluciones tengan campo receptivo de 511 frames, esta normalización conecta cada predicción con toda la parte. Durante entrenamiento sus estadísticas proceden de 60 segundos; durante inferencia, de unos 45 minutos, incluidos apagones. Hay una dependencia global y un cambio de contexto de normalización que la descripción «unos 20 s» oculta.

Comprobación con la TCN completa de v2 y datos reales de Sample 1: al comparar una inferencia de 60 s con la de la parte completa, las probabilidades de los 300 frames centrales difieren hasta **0,222**, pese a tener suficiente margen para las convoluciones. Otra sonda conserva intactos los frames 14000-16000 y sustituye por ceros solo los lejanos: en el frame 14591, a 591 frames del cambio más próximo, P(D) pasa de **0,019 a 0,938**. Esta segunda intervención es artificial: demuestra la dependencia, no mide una caída real de score.

**Qué corregir:** hacer coherentes entrenamiento e inferencia en la normalización/contexto, o documentar y validar expresamente la dependencia global y la sensibilidad a apagones y duración. No es una fuga de etiquetas ni exige un modelo causal; el problema es el contexto distinto y la supuesta localidad.

### 6. [P3] El log 023 no mide retrospectivamente el sesgo del 0,666 de la v1

**`experiments/log/023_s2_4b_lomo_anidado.md:32`** concluye que el 0,666 de la v1 no estaba inflado por el anidado porque, con las probabilidades nuevas de `degrade_v3`, el protocolo anterior da 0,664 y el nuevo 0,668. Esa comparación mide el efecto en **la nueva degradación**, no en la v1 de `degrade_v2`. La conclusión respaldada es que, en este experimento nuevo y con estas semillas de Optuna, corregir el anidado no redujo el score. No permite cuantificar retrospectivamente el sesgo de la v1. La vía de modelos/decodificador sí está corregida; no hace falta repetir la v1 salvo que se quiera mantener esa afirmación histórica.

## Comprobaciones que sí pasan

- **Separación de partidos en los modelos y el decodificador:** F3.1 (`scripts/s3_1_transition_head.py:105-113`), F3.3 (`scripts/s3_3_ensemble_nested.py:79-99`, con la TCN exterior de `scripts/s3_2_tcn_screen.py:83-90`) y F3.7 (`scripts/s3_7_nested_full.py:103-109,132-144`) excluyen T de los entrenamientos interiores y exteriores; los interiores excluyen además J. No he encontrado la fuga de probabilidades de la revisión 003. La excepción es la calibración previa del simulador descrita en el hallazgo 1.
- La calibración isotónica del GBM agrupa las variantes del mismo partido (`src/models/gbm.py:63-79`). Las estadísticas de normalización de la TCN se estiman solo con sus datos de entrenamiento (`src/models/tcn.py:84-87`).
- **Degradación v4 y control sin lectura directa de etiquetas:** en J03WMX, quitar las etiquetas, ponerlas a cero o barajarlas deja idénticos P, B, slots, dirección y features de control. Permutar los slots conserva las features de control, con diferencia máxima de redondeo de 1,9e-6.
- **Decodificador con M3:** comprobado contra enumeración exhaustiva de todas las secuencias de siete frames en 45 casos aleatorios, incluyendo costes dependientes del tiempo y duraciones mínimas. El objetivo óptimo coincide en todos.
- **39 tests pasados**, análisis estático sin errores y comprobaciones públicas oficiales correctas.
- **F3.1 y F3.3:** repetí las 3150 pruebas de Optuna de las tres variantes sobre sus probabilidades interiores guardadas. Las 21 predicciones exteriores coinciden exactamente. Medias: M3 **0,6786239475**; ensemble A **0,7096409515**; B **0,7097619214**.
- **F3.7:** ejecuté el script real con `--decode-only`, 4200 pruebas de Optuna. La tabla completa y las 28 predicciones coinciden exactamente. Media `ens_mean`: **0,7152075338**.
- **M6:** el script de D corto reproduce el descarte: no se elige insertar D en ninguno de los siete folds; la media se mantiene en 0,7097619214.
- **Sample MPS:** repetí la inferencia de los modelos completos y reconstruí sus features: v2 **0,780516 / 0,746541 / 0,826265**, v3 **0,812121 / 0,755352 / 0,840377**. Se reproducen los números publicados para esa evaluación.
- **v1.1:** construcción completa con el código actual, hash exacto **`5731045efe596ddc358928c102997fdf1e723e754e6ff953fc95b1e9f734a06f`**. El constructor ya genera automáticamente el JSON de estadísticas de degradación cuando falta; su hash reconstruido es `ed08956d7d847cb276aafd112e193a7dc60ceb870423687dbfce194004806d71`.
- **TCN CPU:** dos entrenamientos de una prueba pequeña, con algoritmos deterministas, producen probabilidades y señal de cambio idénticas. Para v2/v3 comprobé los CSV ya construidos en el análisis original y sus réplicas: son idénticos entre sí y sus hashes coinciden con los publicados: v2 `0d626fa4d58e823cf50756a3623e90bdeab994baa87256a0ef2ba788b69e8da5`; v3 `fbb875184365706e208ba5666142566a22cea4c43939fcf7a8014ac91098e8e4`. Sus scores públicos recalculados son 0,5058823529 y 0,5696969697. Esto valida los artefactos disponibles, **no constituye dos reconstrucciones independientes mías**.

## Alcance y reproducción

Fuente exacta exportada con `git archive HEAD` a `/tmp/bcn-codex-review-004/repo`, con los datos locales disponibles y el entorno Python 3.11.16 de `.venv311` (Torch 2.14.1). Las notas, sondas y registros están en `/tmp/bcn-codex-review-004/`; no son nuevos componentes del proyecto.

Comandos principales, ejecutados en esa copia:

```sh
python -m pytest -q tests
python -m pyflakes src tests scripts/s3_*.py scripts/make_submission.py
python scripts/check_official.py
python scripts/s3_7_nested_full.py --features v1ctl --decode-only --out experiments/outputs/audit004_nested
python scripts/make_submission_v1.py --degrade-config configs/degrade_v3.json --decoder-config configs/decoder_v1_1.json --out submissions/audit_v1_1.csv
```

Las construcciones CPU v2/v3 se iniciaron con un envoltorio externo que llama al `main()` original y conserva modelos para diagnóstico. **Se detuvieron antes de terminar la TCN, a petición del usuario; no cuentan como reproducciones completas.** No queda ningún entrenamiento de revisión ejecutándose. El JSON y las features reconstruidos durante ese intento sí pudieron contrastarse con los originales. No se han cambiado el constructor ni sus parámetros en el checkout.

**Límite:** la repetición de los LOMO completos reutiliza sus probabilidades interiores/exteriores guardadas para reoptimizar todos los decodificadores; solo reentrené las tres comprobaciones de modelos indicadas y completé la construcción de v1.1. Las construcciones v2/v3 quedaron interrumpidas. No he repetido los 147 entrenamientos anidados de F3.7 ni todo el cribado de las variantes descartadas. No he generado una cifra corregida del LOMO que incluya el ajuste interior del simulador. Las sondas sintéticas verifican dependencias/cachés y el algoritmo del decodificador; no sustituyen una validación del partido completo, cuya verdad no está disponible.

Solo se añade este informe al checkout; sin cambios de código y sin commit.
