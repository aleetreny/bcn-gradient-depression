# Tablero de sistemas candidatos

Este documento sirve para **decidir al final qué algoritmo presentamos**. Cada sistema que se construye (Fase 2 en adelante) añade aquí una fila con las mismas columnas, medidas con el mismo código (`src/eval/`). El detalle de cada fila está en su informe de `experiments/log/` y en `experiments/registry.csv`.

## Cómo se decide

| Juez | Qué mide | Peso en la decisión |
|---|---|---|
| **LOMO Sportec degradado** | spellF1 medio de 7 folds (entrenar con 6 partidos, puntuar el 7.º) sobre Sportec estropeado para parecerse al ATD (S2.1). Media ± desviación entre folds | **Principal.** Una mejora cuenta si sube ≥ +0,01 de media y mejora ≥ 5 de 7 folds |
| LOMO Sportec limpio | Lo mismo sin degradar | Contexto: cuánto se pierde por la calidad del tracking |
| Públicos BAR‑ATH (5 min) | spellF1 oficial con IC bootstrap 5–95 % | Solo control: que no empeore fuera del ruido. **No sirve para elegir** (17 posesiones). **Desde el 24‑10 las reglas prohíben ajustar o entrenar con ellos:** el simulador se recalibró sin etiquetas y salió la misma configuración (log 041) |
| Metrica Sample (3 partidos, mismo proveedor) | spellF1 con etiquetas derivadas de eventos; en el Sample el balón falta en los parones (D más fácil) | Segundo juez de transferencia: que el orden de los candidatos se mantenga fuera de Sportec |
| Plausibilidad BAR‑ATH completo | Posesiones/min, % D, mediana de posesión y % < 2 s predichos en los 158.857 frames evaluados | Detectar sistemas raros. Referencias: Sportec 4,6 pos./min y 45 % D; Metrica Sample 3,9 y 35 % (S0.3–S1.5) |

Candidata final y reserva (Fase 4): la mejor en LOMO degradado que pase los controles, y la mejor versión con menos piezas.

**Regla de la Fase 4** (escrita el 24‑10 antes de ver la v4, por la revisión 004):
1. **v4 (= v3 con TCN local) sustituye a la v3** si no es peor fuera del ruido. Arregla un defecto conocido (dependencia global de la TCN), así que basta con que no sea peor:
   - LOMO anidado ≥ v3 − 0,002;
   - Sample, medido con los pesos CPU del fichero, ≥ 0,803 − 0,005;
   - posesiones/min del BAR‑ATH dentro de ±10 % de la v3.
2. **Candidata final frente a la reserva simple** (GBM con control). La final es el ensemble si:
   - gana a la reserva en el LOMO por ≥ +0,01;
   - y no pierde en el Sample por más de 0,01.

   Si no, se envía la reserva. Ahora mismo: +0,028 en el LOMO y +0,004 en el Sample.
3. **Los públicos solo bloquean un fichero roto:** spellF1 público < 0,40 (el baseline saca 0,33; con IC de ±0,18 un umbral más alto bloquearía ficheros buenos por ruido) o F1 de D < 0,75. No eligen entre candidatos.

## Sistemas

| Sistema | Paso | LOMO degradado | LOMO limpio | Públicos [IC 5–95 %] | F1 vivo/parado públicos · F1 de D | Pos./min · % D · mediana · % < 2 s (BAR‑ATH) | Estado |
|---|---|---|---|---|---|---|---|
| Baseline oficial: más cercano + mayoría ±50 | S0.6 / S2.1 | 0,298 (0,299 con la semilla 0) | 0,308 ± 0,026 | 0,326 [0,214–0,447] | 0,899 · 0,323 | 5,61 · 32,3 % · 4,6 s · 31 % | Referencia |
| GBM dos cabezas (160 features v1) + suavizado ±12 | S2.3 | 0,629 ± 0,032 (7/7 > baseline) | 0,611 (w 50; entrenado degradado) | 0,400 [0,296–0,496] | 0,963 · 0,829 | 3,80 · 47,3 % · 4,1 s · 30 % | Candidato 1 (superado por el 2) |
| **GBM dos cabezas + decodificador semi‑Markov** (λ_HA 2,16, d_min 10/106) | S2.4 | **0,666 ± 0,029** (7/7 > candidato 1) *(no anidado del todo: revisión 003)* | — | **0,519 [0,368–0,660]** | **0,968** · 0,847 | 3,14 · 48,6 % · 5,2 s · 19 % | Candidato 2 = **v1, reserva** (tag `v1`, `make_submission_v1.py`, SHA‑256 `4491ebe4…`) |
| **v1.1 = GBM dos cabezas + decodificador, entrenado con `degrade_v3`** (λ_HA 5,30, d_min 8/48) | S2.4b / 024 | **0,668 ± 0,034 (anidado, juez nuevo)** | — | 0,495 [0,358–0,611] | 0,971 · **0,865** | 2,86 · 49,9 % · 6,3 s · 12 % | Referencia de la Fase 3 (superada por v1.2) (tag `v1.1`, SHA‑256 `5731045e…`) |
| **v1.2 = v1.1 + M3** (cabeza de transición, coste de cambio dependiente del tiempo) | F3.1 / 027 | **0,679 ± 0,033** (anidado; +0,011, 7/7 vs v1.1) | — | **0,533 [0,364–0,691]** | 0,971 · 0,871 | 2,58 · 49,0 % · 6,9 s · 11 % ⚠ pocas posesiones | Candidato (superado por v2) (SHA‑256 `98989a3f…`, `make_submission.py --transition`) |
| **v2 = media(GBM, TCN) + M3/TCN cambio + decodificador** | F3.3 / 029 | **0,710 ± 0,040** (anidado; +0,031, 7/7 vs v1.2) | — | 0,506 [0,329–0,682] | 0,972 · 0,868 | 2,60 · 49,6 % · 6,4 s · 15 % | Candidato (superado por v3) (tag `v2`, SHA‑256 `0d626fa4…`, 58 min en CPU) |
| **v3 = v2 con features de control en los tres modelos** | F3.7 / 032 | **0,715 ± 0,031** (anidado; +0,005 vs v2, 6/7) | — | **0,570 [0,387–0,752]** | 0,967 · 0,865 | 2,39 · 49,1 % · 7,2 s · 12 % ⚠ pocas posesiones | Candidato (superado por v4) (tag `v3`, SHA‑256 `fbb87518…`); Sample 0,803 con modelos MPS |
| **v4 = v3 con TCN local, cambio de M3** | F3.11 / 040 | **0,714 ± 0,036** (anidado; −0,001 vs v3, empate) | — | 0,613 [0,438–0,788] | 0,969 · 0,850 | 2,47 · 49,6 % · 6,8 s · 11 % ⚠ pocas posesiones | **Candidata final** (tag `v4`, SHA‑256 `dbc36070…`, 52 min CPU); Sample **0,812 con los pesos CPU del fichero** |
| GBM con features de control (reserva simple) | F3.7 / 032 | 0,687 (anidado) | — | 0,584 [0,443–0,714] | 0,973 · 0,882 | 2,80 · 49,2 % | **Reserva** (SHA‑256 `f067c06e…`, reproducible); Sample 0,799 |

*Notas:* "LOMO degradado" = Sportec con `configs/degrade_v2.json` (degradación v3, S2.1/S2.3): los modelos se prueban con la semilla 0, ± desviación entre folds. **El LOMO degradado sobreestima la detección de D** (AUC vivo 0,997 en LOMO frente a 0,953 en los públicos), así que las mejoras de D se confirman también con la columna "F1 vivo/parado públicos". En Sportec degradado el baseline predice 5,2 posesiones/min y un 36 % de D, parecido a su comportamiento en el BAR‑ATH. En S1.7 el baseline sobre Sportec con los huecos del ATD **trasplantados** daba 0,210. Es otra degradación, así que no se compara directamente con la columna "LOMO degradado".

**Revisión 003:** las cifras "LOMO degradado" de S2.3 y S2.4 usan `degrade_v2` (con la huella del balón pegado al jugador en los parones) y, en S2.4, un decodificador ajustado con predicciones de modelos que vieron el partido de prueba. **No son el juez definitivo.** El juez nuevo es el **LOMO anidado completo con `degrade_v3`** (S2.4b). El F1 vivo/parado oficial toma juego como positivo; por eso se añade el F1 de la clase D.

**Transferencia a Metrica Sample (spellF1 medio en 3 partidos):** baseline 0,423 · v1.1 0,770 · v1.2 0,782 · v2 0,784 · GBM con control 0,799 · v3 0,803 · ens. cambio M3 0,809. Las cifras con TCN son de los modelos entrenados en **MPS** (F3.8), no de los pesos CPU de los ficheros reproducibles. La TCN aporta allí poco y las features de control mucho. **No hay el mismo orden completo que en el LOMO:** el GBM con control queda por debajo de la v2 en el LOMO y por encima en el Sample.

**Revisión 004 (cómo leer las cifras):**
1. El simulador (`degrade_v3`) se calibró con los D públicos y con los D reales de J03WMX y J03WQQ. Por eso los públicos son desarrollo y, cuando el partido exterior es uno de esos dos, la cadena completa no está anidada. No se ha medido cuánto infla el 0,715.
2. Ningún candidato gana en los tres jueces: la v3 gana en el LOMO, `ens_m3` en el Sample y el GBM con control en público. La v3 es **provisional**; las features de control sí están confirmadas.
3. En el Sample el balón casi no se ve en D, así que mide sobre todo la señal de equipo.
