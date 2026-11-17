# 05 · Plan de implementación y pruebas

Tenemos **acceso completo a los datos del partido** y ya están descargados todos los datos públicos (`data_external/`). Así podemos diagnosticar directamente sobre el Barça–Athletic, calibrar con estadísticas reales, revisar predicciones frame a frame y **poner a prueba, uno por uno, los métodos que han funcionado a otros** (ver la matriz de la Fase 3).

## Resumen (lo importante en 2 minutos)

**Filosofía:** primero medir, luego modelar. Cada paso es pequeño, produce un **output concreto** (una tabla, un gráfico o un CSV) que revisamos juntos, y acaba con una **decisión escrita**: seguir, modificar o descartar. Nada se da por bueno porque "suena bien" en un paper.

**Tres jueces para decidir si algo funciona:**

| Juez | Qué es | Para qué sirve | Para qué NO |
|---|---|---|---|
| **Sportec "un partido fuera"** | Entrenar con 6 partidos, puntuar con spellF1 en el 7.º, rotando. Unas 2.000 posesiones | **Elegir** entre métodos y ajustar parámetros | Medir cosas exclusivas de Metrica (Smart Tagging) |
| **Los 5 min públicos del Barça–Athletic** | 17 posesiones con la verdad real | Comprobar que nada está roto y calibrar **≤ 3 parámetros** físicos | Elegir entre versiones (ya se demostró que engaña) |
| **Panel de plausibilidad del partido completo** | Posesiones/min por tramo de 15 min, % de D, D cerca de saques, % de posesiones cortas | Detectar sistemas raros (demasiadas o muy pocas posesiones) | Puntuar |

**Fases** (se abordan una a una; la siguiente empieza cuando revisamos la anterior):

| Fase | Qué sale |
|---|---|
| 0 · Cimientos | Lectores de los 4 datasets en un formato común, evaluador LOMO, registro de experimentos, baseline reproducido al frame |
| 1 · Diagnóstico | Cómo se ve en el tracking cada cambio de posesión real, qué regla imita la "bandera" de Metrica, anatomía de los huecos del balón, cuánto se puede ganar en cada tipo de error |
| 2 · Sistema v1 | Árboles + decodificador ajustado a spellF1. **Primera submission válida, congelada como red de seguridad** |
| 3 · Torneo de mejoras | Los métodos de otros trabajos (PathCRF, Ball Radar, Lang, Borghesi, Kaggle DFL, CropDrop…) más ideas propias, probados uno a uno por orden de ganancia potencial |
| 4 · Selección final | Ensemble, robustez, candidata final más una de reserva |
| 5 · Congelar y entregar | Reproducible byte a byte, informe técnico, envío |

**La gran pregunta de cada fase:** Fase 1, ¿dónde se pierde la nota? Fase 2, ¿cuánto gana un sistema sólido? Fase 3, ¿qué apuesta ataca el mayor agujero que queda?

## Cómo trabajamos paso a paso

1. **Un paso = un commit en `Ale`** (o una sub‑rama `ale/<paso>` si es grande) más un **informe de paso** en `experiments/log/NNN_<paso>.md` con esta plantilla:

   ```markdown
   # NNN · <paso>  (fecha, commit)
   Comando: `...`        Config: configs/<...>.yaml        Semilla: ...
   ## Resultado
   (tabla de métricas; rutas de los gráficos en experiments/outputs/NNN/)
   ## Hallazgos
   - ...
   ## Decisión
   seguir / modificar (cómo) / descartar (por qué)   → siguiente paso
   ```

2. **Registro único** `experiments/registry.csv`, con una fila por ejecución: `id, fecha, commit, paso, config, lomo_spellf1_media, lomo_std, folds_mejorados, public_spellf1, public_macrof1, posesiones_por_min, notas`.
3. **Outputs pesados** (predicciones, gráficos, caches) en `experiments/outputs/`, fuera de git solo por tamaño.
4. **Regla de aceptación** de una mejora: LOMO spellF1 medio ≥ +0,01, mejor en ≥ 5 de 7 partidos **y** sin empeorar los 5 min públicos más allá del ruido (bootstrap). Si la mejora solo es medible en Metrica (Smart Tagging, reglas de toque), aplicar la regla de los ≤ 3 parámetros y el panel de plausibilidad.
5. **Revisión conjunta** de cada informe antes del paso siguiente. Los hallazgos importantes se suben a una lista viva en `experiments/HALLAZGOS.md`.

## Fase 0 · Cimientos

| Paso | Hacer | Output a revisar | Puerta |
|---|---|---|---|
| **S0.1 Entorno** | `uv venv --python 3.11 .venv311`; `requirements-research.txt` (numpy, pandas, pyarrow, scipy, scikit‑learn, lightgbm, optuna, kloppy, matplotlib, tqdm; torch más adelante). Máquina: M5 Pro, 48 GB, MPS para redes | `python scripts/check_official.py` pasa; versiones impresas | Pasa |
| **S0.2 Lector BAR‑ATH** | `src/data/metrica_atd.py`: parser numpy del `.txt` → `P[T,46,3]` (x, y en m; velocidad en m/s), `B[T,3]`, máscaras (`slot_visible`, `ball_xy_visible`, `ball_speed_only`), equipo por slot, cuál es H, dirección de ataque normalizada (ver [06](06_como_son_nuestros_datos.md)). Cache en `local_data/cache/` | Tabla resumen frente a DATA.md: balón 56,9 % total / 59,3 % evaluado, 20,5 objetos de media en frames evaluados, 46 slots, 199.319 frames | Todo cuadra |
| **S0.3 Lector Sportec** | `src/data/sportec.py`: mismo formato, slots anónimos, etiquetas H/A/D (`BallStatus`, `BallPossession`) | Por partido: frames, % D, posesiones/min, % balón visible, duración media de posesión | 7 partidos cargados y coherentes |
| **S0.4 Lector Metrica Sample** | `src/data/metrica_sample.py`: juegos 1–3 y etiquetas derivadas de eventos (RECOVERY / BALL LOST → cambio; BALL OUT / FAULT → D hasta SET PIECE) | Estadísticas de etiquetas y una línea de tiempo de 5 min frente a los eventos | Posesiones/min plausibles; si no, se aparca |
| **S0.5 Evaluador** | `src/eval/`: (a) score en los públicos con bootstrap; (b) LOMO sobre N partidos; (c) **descomposición de errores** (cambio tarde/pronto con desfase en frames, robo corto perdido, cambio espurio, D perdido, D inventado); (d) gráfico de línea de tiempo verdad vs predicción; (e) alta en el registro | Reproduce los scores publicados de los 4 baselines en los públicos (smooth50 = 0,3256) | Idénticos |
| **S0.6 Baseline reimplementado** | Más cercano + mayoría ±50 en nuestro formato | BAR‑ATH: **idéntico frame a frame** a `evaluation/results/baseline_smooth50.csv`. Sportec limpio: primera fila LOMO de referencia | Idéntico |

## Fase 1 · Diagnóstico

| Paso | Pregunta | Hacer | Output a revisar | Decisión que habilita |
|---|---|---|---|---|
| **S1.1 Anatomía de los cambios reales** | Cuando la verdad cambia de equipo, ¿qué muestra el tracking? | Para cada una de las 16 fronteras y 4 tramos D públicos, ventana ±3 s: balón visible / solo velocidad / ausente, equipo más cercano por frame, distancia por equipo, Smart Tagging activo | Tabla de 20 filas + 20 mini‑gráficos. Desfase entre el cambio del equipo más cercano y el de la verdad | Si la mayoría de cambios ocurren **con balón visible**, la prioridad es clavar el momento (S1.2). Si caen en huecos, la prioridad es el balón (M2a–c, M13) |
| **S1.2 ¿Qué regla imita la bandera de Metrica?** | ¿"Más cercano", "último toque" o "zona de control"? | 3 reglas con ≤ 3 parámetros cada una, evaluadas en los públicos **solo en frames con balón visible**: acuerdo por frame y desfase mediano de las transiciones | Tabla regla × acuerdo × \|desfase\| mediano | La regla ganadora se convierte en las features centrales (M1) y orienta la semántica del decodificador |
| **S1.3 Anatomía de los huecos del balón** | ¿Cómo y dónde se pierde el balón? | Partido completo: distribución de longitudes de hueco; contexto al empezar (velocidad del balón, jugadores a < 3 m, zona); contexto de los frames con "solo velocidad"; frecuencia con la que el equipo más cercano cambia a través de un hueco; % de transiciones públicas dentro de huecos | Histogramas + tabla → `local_data/degradation_params.json` | Parámetros de la degradación (S2.1); tamaño del premio de rellenar el balón |
| **S1.4 Anatomía de los slots** | ¿Cómo de roto está el tracking de jugadores? | Vida de cada fragmento de slot, objetos por equipo y frame, duplicados, saltos imposibles (> 1 m/frame), significado de `position_type` (¿portero?) | Distribuciones → al mismo JSON | Modelo de ruido de jugadores para la degradación; feature de portero |
| **S1.5 Semántica Sportec vs Metrica** | ¿Las banderas de Sportec se comportan como las de Metrica? | Duraciones, % < 2 s, posesiones/min, % de cambios directos H↔A, % de D y duración de D. En Sportec, momento del cambio de bandera frente al pase o recepción del evento | Tabla lado a lado: Sportec / Metrica Sample / públicos BAR‑ATH | Hasta qué punto se fía uno de Sportec para (a) D y (b) el momento del cambio de equipo. Fija cómo se ajusta el decodificador |
| **S1.6 Smart Tagging frente a la verdad** | ¿Sirve como ancla? | Inicios de SET PIECES vs finales de D públicos (desfase); `Team` de las fases vs equipo de la verdad; saques de banda con N/A | Tabla de desfases y de acuerdo | Si los desfases son estables (< 1 s), entran como anclas en el decodificador (M5) |
| **S1.7 Margen de mejora por tipo de error** | ¿Dónde está la nota? | Experimentos **oráculo** sobre el baseline (en los públicos y en Sportec LOMO): arreglar con la verdad un tipo de error cada vez (D perfecto, momento de cambio perfecto, balón completo en Sportec, sin cambios espurios) y repuntuar | Tabla "si arregláramos X ganaríamos Y" | **Ordena la Fase 3** por ganancia potencial |

## Fase 2 · Sistema v1

| Paso | Hacer | Output a revisar | Puerta |
|---|---|---|---|
| **S2.1 Degradación calibrada** | Estropear Sportec con los parámetros de S1.3/S1.4: huecos de balón por rachas condicionados al contexto, slots fragmentados, jugadores perdidos y duplicados, ruido. **Validación adversaria:** un GBM que intenta distinguir frames de BAR‑ATH de frames de Sportec degradado | AUC del clasificador adversario antes/después y las 10 features que más delatan el dominio | AUC baja claramente (orientativo: < 0,7). Si no, iterar sobre las features delatoras |
| **S2.2 Features v1** | `src/features/`: misma función para todas las fuentes (ver Anexo A). Tests: sin fugas de etiqueta, máscaras correctas, invariancia al orden de los slots | Tabla de features con % de NaN por fuente | Tests verdes |
| **S2.3 Modelos por frame** | GBM_vivo y GBM_equipo, LOMO, calibración isotónica | Métricas por frame por fold, curvas de calibración, spellF1 con argmax + mediana (todavía sin decodificador) | > baseline LOMO |
| **S2.4 Decodificador** | Viterbi semi‑Markov H/A/D (Anexo A [C]) y Optuna maximizando el spellF1 LOMO en Sportec degradado | Mejores parámetros, **gráfico de meseta**, spellF1 por fold, posesiones/min | Mejora clara sobre S2.3; elegir el centro de la meseta |
| **S2.5 v1 sobre BAR‑ATH** | Generar el CSV, `validate_submission.py`, score público con bootstrap, **panel de plausibilidad** comparado con Sportec, revisión visual de 3 ventanas de 2 min al azar | `submissions/v1.csv` + panel + gráficos | Válido y plausible → **tag `v1`, red de seguridad** |
| **S2.6 Descomposición de errores de v1** | Igual que S1.7 sobre v1 | Tabla de errores por tipo | Reordena la Fase 3 |

## Fase 3 · Torneo de mejoras: los métodos de otros, puestos a prueba

Cada mejora es un **interruptor independiente** en la config y se prueba con el mismo protocolo. Se juegan por orden de ganancia potencial (S1.7/S2.6), no por orden de la tabla. La columna "Viene de" dice qué trabajo previo la inspira y qué resultado publicado esperamos acercarnos a reproducir (detalle en [01](01_que_han_conseguido_otros_trabajos_y_retos.md) y [03](03_que_aprendemos_de_cada_paper.md)).

### Paso previo: reproducir a los demás antes de usarlos

Antes de adaptar un método ajeno, comprobamos que lo usamos bien **reproduciendo su resultado en sus propios datos**. Si no sale, el problema es nuestro y no del método.

| Control | Qué reproducir | Resultado publicado a alcanzar |
|---|---|---|
| **C1 PathCRF** | Inferencia de `saved/140` en el partido de test de Sportec con su `tutorial.ipynb` | Edge acc ≈ 69,6 %, acierto de equipo ≈ 92,4 %, F1 de eventos ≈ 75,7 % |
| **C2 Ball Radar** | Entrenar su modelo en Metrica Sample con su repo (ya incluye el preprocesado) | Acierto de equipo ≈ 85 % sin balón; ≈ 95–97 % con un 10–20 % del balón visible |
| **C3 Lang** | Su detector vivo/parado (features de jugadores con desfases ±0,4…80 s + AdaBoost + mediana de 3,64 s) en Sportec LOMO | Acierto por frame ≈ 92 %, F1 de interrupciones ≈ 80 % |

### Matriz de experimentos

| # | Mejora | Viene de | Hipótesis para nuestro caso | Cómo se valida |
|---|---|---|---|---|
| M1 | **Reglas de toque y zona de control** como features (S1.2) | Vidal‑Codina 2022 (databallpy); Link & Hoernig 2017 | Clavan el momento del cambio cuando el balón se ve, que es justo lo que más pesa en spellF1 | LOMO + públicos |
| M2a | **Relleno clásico del balón** (interpolación de huecos < 1 s; "balón en el jugador probable") | Práctica estándar | Gana gran parte del premio de los huecos cortos sin entrenar nada | LOMO degradado (en Sportec se conoce el balón oculto) |
| M2b | **Imputador del balón aprendido** (Set Transformer + Bi‑LSTM jerárquica, con el poseedor como tarea intermedia) | Ball Radar (KDD 2023): 85,9 % de equipo sin balón, 97,4 % viendo el 20 % | Con el 59 % del balón visible deberíamos estar en su régimen alto, salvo que nuestros huecos son "difíciles" | Tras C2. Error de posición en huecos simulados + LOMO |
| M2c | **Trayectoria del balón globalmente coherente**: camino mínimo / Viterbi a nivel de jugador (estado = slot que lleva el balón o "balón en vuelo"; transiciones limitadas por la velocidad física del balón) | Kaggle DFL 2.º (K_mat): Dijkstra sobre candidatos de balón en vez del mejor por frame | Una secuencia coherente aporta lo que un frame aislado no puede | LOMO + públicos |
| M3 | **Cabeza de transición** (P de cambio en t ± 3 frames) con coste del decodificador dependiente de t | *Action spotting* de Kaggle DFL; PathCRF (transiciones dinámicas) | Mejora la puntualidad, que se mide en spellF1 @ IoU 0,7 | LOMO |
| M4a | **Detector de D estilo Lang** como modelo propio (features colectivas con desfases ±80 s) | Lang 2022 (Sci Rep): 92 % por frame | Mejor D que un GBM genérico; sus fronteras (±1,7 s) importan poco según nuestra simulación | Tras C3. LOMO |
| M4b | **Señales rápidas de D**: balón fuera del campo, frenazo colectivo, colocación de saque, balón con solo velocidad | Propio (4 §3) | Afinan los bordes de D que Lang deja borrosos | LOMO |
| M5 | **Smart Tagging** como anclas del decodificador o features de segunda etapa | Propio; dato exclusivo de Metrica | Mejores reinicios y equipo que saca | Públicos + plausibilidad, ≤ 3 parámetros |
| M6 | **D corto como separador** | Propio (simulación de spellF1, [04 §3](04_como_puntua_spellf1_y_otros_hallazgos.md)) | Menos coste por robos cortos dudosos | LOMO |
| M7 | **Metrica Sample** como datos extra y como folds extra | Ball Radar lo usa para entrenar | Mismo proveedor, menos cambio de dominio | LOMO ampliado a 10 partidos |
| M8 | **SkillCorner** para preentrenar robustez (jugadores extrapolados o fuera de cámara) | SkillCorner Open Data | Robustez frente a slots perdidos | LOMO en Sportec degradado |
| M9a | **Dos cabezas vs una ternaria** (vivo/parado × equipo vs H/A/D directo) | Borghesi 2023 (ESWA): 86,2 % posesión, 89,2 % vivo/parado; comparan las dos formulaciones | Factorizar es mejor | LOMO, con GBM y con TCN |
| M9b | **TCN sobre conjuntos de slots** con agregación invariante (suma / self‑attention) | Borghesi 2023 | Capta patrones temporales que el GBM no ve | LOMO; valor en ensemble |
| M9c | **Cabeza temporal 1D U‑Net** como alternativa a la TCN | Kaggle DFL 3.º (Camaro) | Mejores fronteras por su resolución multiescala | LOMO |
| M9d | **Tareas auxiliares** en la red (posición del balón, jugador poseedor) | Peral (CropDrop) y PathCRF: las pérdidas auxiliares aportan 3–6 puntos | Mejor representación con pocos partidos | LOMO |
| M10a | **PathCRF `saved/140` como feature** (P de equipo por frame desde sus aristas) | PathCRF (KDD 2026): 92,4 % de equipo sin balón | Señal sin balón de otra familia de modelos | Tras C1. Antes: > 85 % de equipo en Sportec degradado |
| M10b | **Transiciones aprendidas (capa CRF) vs costes ajustados a mano** en el decodificador | PathCRF: "Non‑CRF + Viterbi con restricciones" ya da la mayor parte de la ganancia | El decodificador a mano basta; el CRF entrenado añade algo en las fronteras | LOMO |
| M11 | **Tipos de enmascarado al entrenar**: i.i.d. por frame vs rachas vs slot entero | Peral (CropDrop): enmascarar trayectorias enteras empeora; por jugador y secuencia completa es mejor que por frame | El enmascarado que imita nuestros huecos generaliza mejor | AUC adversaria (S2.1) + LOMO |
| M12 | **Autoentrenamiento en BAR‑ATH** (pseudo‑etiquetas de frames muy seguros → reentrenar) | Práctica estándar de adaptación de dominio | Adapta el modelo a nuestro tracking | Simulado en Sportec (limpio → degradado); luego públicos |
| M13 | **Detector de balón sobre el vídeo táctico**, autosupervisado con ATD + homografía (vídeo de 5,2 GB) | Equipo SyntaxEror (issue #2 del repo oficial) | Rellena huecos con observaciones reales | Recall en frames con balón conocido ocultado a propósito; después públicos + plausibilidad |
| M14 | **ATD de retransmisión** sincronizado como segunda fuente de balón | Propio | Otras oclusiones, otros huecos | % de huecos tácticos cubiertos > 10 % |

## Fase 4 · Selección final

1. **Ensemble** de probabilidades por frame de los modelos que hayan pasado, buscando **diversidad** (GBM + red + PathCRF), con media de logits ponderada en LOMO. Referencia: en Kaggle DFL (Camaro) pasar de 1 a 6 modelos subió de 0,768 a 0,827.
2. **Reajuste único** del decodificador con el ensemble.
3. **Robustez:** variar ±20 % cada parámetro del decodificador. La candidata debe estar en una meseta (spellF1 LOMO estable) y con posesiones/min estables en BAR‑ATH.
4. **Reentrenar con todos los datos** (los 7 de Sportec, más Metrica Sample si pasó M7) con la configuración elegida, como hizo el ganador de Kaggle DFL (Team Hydrogen) tras validar con un split fijo.
5. Elegir **candidata final** y **reserva conservadora** (la mejor versión con menos piezas).

## Fase 5 · Congelar y entregar

- `make submission`: desde los datos crudos hasta el CSV, dos ejecuciones con **hash idéntico**, en CPU y con tiempo de ejecución medido (RULES.md exige regenerar el mismo fichero).
- Informe técnico ([docs/TECHNICAL_REPORT_TEMPLATE.md](../../docs/TECHNICAL_REPORT_TEMPLATE.md)) con las tablas de ablaciones del registro.
- Formulario de envío (SUBMISSION.md): CSV + URL del repo + informe. Para la entrega el repo tiene que ser público; publicarlo sin la carpeta de datos (RULES.md).
- Enviar con al menos un día de margen sobre la fecha límite (17 nov).

## Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| La semántica de Sportec difiere de la bandera de Metrica | S1.5. Usar Sportec sobre todo para D y para el comportamiento del decodificador; calibrar la reactividad con S1.2 |
| La degradación no se parece al ATD real | Validación adversaria (S2.1), varias degradaciones, meseta en vez de pico |
| Sobreajustar los 5 min públicos | Solo ≤ 3 parámetros calibrados con ellos; las decisiones se toman en LOMO |
| No reproducible | Semillas, dependencias fijadas, hash del CSV comprobado en `make submission` |
| Falta de tiempo | v1 congelada al acabar la Fase 2 como red de seguridad; la Fase 3 se corta por prioridad, no por orden |

## Estructura del repo prevista

```
src/data/        metrica_atd.py · sportec.py · metrica_sample.py · skillcorner.py   (→ formato común P/B/máscaras/etiquetas)
src/features/    features invariantes a slots (Anexo A)
src/models/      gbm.py · tcn.py · calibration.py
src/decode/      viterbi_hsmm.py · postprocess.py
src/eval/        harness.py (públicos, LOMO, descomposición de errores, oráculos) · plots.py · registry.py
configs/         una YAML por experimento
experiments/     log/NNN_*.md · registry.csv · HALLAZGOS.md · outputs/ (ignorada)
scripts/         download_public_datasets.sh · inspect_data_structure.py · spellf1_error_costs.py · make_submission.py
```

## Anexo A · Arquitectura de referencia (v1)

```
ATD (46 slots, balón con huecos) ──┐
Smart Tagging ─────────────────────┼─► [A] features por frame (invariantes a slots, con máscaras)
Estructura publicada (partes, KO) ─┘            │
                                                ▼
                         [B] GBM_vivo: P(vivo_t)      GBM_equipo: P(H_t | vivo_t)
                             GBM_trans: P(cambio de equipo en t ± 3 frames)   (M3)
                                                │
                                                ▼
              [C] decodificador semi-Markov H/A/D con costes de transición dependientes de t
                                                │
                                                ▼
              [D] post-proceso: D corto como separador, estructura fija (D antes del KO de la 2.ª parte)
                                                │
                                                ▼
                         CSV 199.319 filas  →  validate_submission.py
```

`P(H_t) = P(vivo_t)·P(H_t|vivo_t)`, `P(A_t) = P(vivo_t)·(1−P(H_t|vivo_t))`, `P(D_t) = 1−P(vivo_t)`.

### [A] Features (en metros, con la dirección de ataque normalizada por parte)

Por frame t (si falta algo, NaN más una **máscara**; nunca tratar "falta el balón" como D):

- **Balón:** x, y, vx, vy, velocidad, aceleración, `visible`, `solo_velocidad`, `frames_desde_visto`, `frames_hasta_visto` (el problema es offline y se puede mirar al futuro), última y próxima posición conocida, distancia de la última posición a cada línea, `fuera_del_campo`.
- **Balón ↔ equipos:** distancia mínima de cada equipo al balón (y a la última posición conocida), diferencia entre ambas, nº de jugadores de cada equipo en radios de 2/5/10 m, velocidad relativa del más cercano.
- **Toques (Vidal‑Codina, M1):** `toque_H`, `toque_A` (balón a < 1–1,5 m con cambio de dirección > θ o de velocidad > δ), equipo del último toque y frames desde él.
- **Colectivas por equipo:** centroide, dispersión, líneas defensiva y adelantada, anchura, velocidad media/mediana/p90, fracción casi parada (< 1,5 m/s), jugadores en cada área, nº de objetos detectados, posición del portero si `position_type` lo identifica.
- **Contexto temporal (Lang):** desplazamientos y medias móviles en ±{0,4; 0,8; 2; 4; 8} s, y ±{20; 40; 80} s para el detector de D.
- **Smart Tagging (M5):** dentro de fase H/A, frames desde/hasta el inicio de un set piece, tipo de set piece, `Team` (N/A como categoría).
- **Estructura:** tiempo desde el saque inicial de la parte.

### [B] Modelos

LightGBM con `monotone_constraints` donde tenga sentido físico, entrenado con Sportec degradado (varias degradaciones por partido como aumento de datos), CV agrupada por partido y calibración isotónica por fold.

### [C] Decodificador

```
score(path) = Σ_t w_e · log P(s_t | x)
            − Σ_t [s_t ≠ s_{t−1}] · cost(s_{t−1} → s_t, t)

cost(H↔A, t)    = λ_HA − β · logit P_trans(t)
cost(vivo→D, t) = λ_LD − γ · señal_parada(t)
cost(D→vivo, t) = λ_DL − γ' · señal_reinicio(t)
duración mínima: d_min_HA, d_min_D   (sub‑estados en cadena)
```

Restricción dura: frames 110.808–113.741 = D. Optuna sobre `{w_e, λ_HA, λ_LD, λ_DL, β, γ, γ', d_min_HA, d_min_D}` maximizando el spellF1 LOMO, quedándonos con el centro de la meseta.

### [D] Post‑proceso

- **D corto como separador (M6):** una posesión predicha < τ_len con confianza media < τ_conf se reescribe como D ([04 §3](04_como_puntua_spellf1_y_otros_hallazgos.md)). τ se ajusta en LOMO.
- Sin X en frames evaluados (macroF1 es el desempate).
