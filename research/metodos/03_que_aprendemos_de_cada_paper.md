# 03 · Literatura y modelos: qué nos llevamos de cada uno

## Resumen

| Trabajo | Qué demuestra (en cristiano) | Qué cogemos | Qué NO hacemos |
|---|---|---|---|
| **PathCRF** (KDD 2026) | Decidir la secuencia completa con reglas de "qué transiciones son posibles" mejora mucho los cambios de posesión, aunque el acierto por frame casi no cambie | La idea de **decodificador con restricciones**, y sus pesos como feature extra opcional | Usar sus pesos tal cual: piden 22 jugadores fijos e ignoran el balón parado |
| **Ball Radar** (KDD 2023) | Con solo los jugadores se adivina el equipo con balón el 86 % del tiempo, y viendo parte del balón se reconstruye el resto bastante bien | **Velocidad y aceleración** como features; **rellenar el balón** en los huecos | Entrenar su red completa sin antes demostrar que los huecos son el problema |
| **Lang** (Sci Rep 2022) | Un modelo sencillo de árboles detecta el "balón parado" con un 92 % de acierto, mirando cómo se mueven los jugadores ±80 s | **El detector de D**: features de movimiento colectivo con contexto pasado y futuro | Confiar en sus fronteras: fallan ~1,7 s de media |
| **Borghesi TCN** (ESWA 2023) | Separar "¿vivo o parado?" de "¿qué equipo?" funciona bien; las redes temporales sobre tracking dan 86–89 % | **Factorizar** D vs (H/A); TCN como segundo modelo | Hacerla el primer sistema |
| **Peral, CropDrop** (arXiv 2025) | Ocultar información a propósito al entrenar hace al modelo robusto | **Degradar el entrenamiento** con huecos realistas | Su parte de vídeo (datos privados, ignora D) |
| **Kaggle DFL** | Ganan quienes validan bien y post‑procesan secuencias | Disciplina de validación por partido | Modelos de vídeo |
| **Vidal‑Codina 2022 / databallpy** | Reglas físicas simples (radio de control, cambio brusco de la velocidad del balón) detectan toques y posesión individual | **Features de "toque"** cuando hay balón | — |

**Mensaje principal:** los papers coinciden en que lo que más mejora los *cambios* de posesión es (1) una buena señal por frame que **no dependa de ver el balón** y (2) un **decodificador temporal** que impida transiciones absurdas. Una arquitectura más sofisticada aporta poco comparado con eso.

## Desarrollo

### PathCRF: detalles útiles para implementar

- Estado = arista (u,v) entre 26 nodos (22 jugadores + 4 "fuera" por línea de banda/fondo). (u,u) = el jugador u controla; (u,v) = balón viajando de u a v.
- "Masked CRF" = matriz de transición con **ceros estructurales**. Desde (u,u) solo se puede ir a (u,u) o (u,v); desde (u,v) solo a (u,v) o (v,v). El decodificador Viterbi se restringe a eso.
- Que "Non‑CRF + VCD" (decodificación Viterbi con restricciones sobre un modelo sin CRF) ya llegue a 73,14 F1 frente a 75,69 del modelo completo indica que **la mayor parte de la ganancia se consigue sin entrenar un CRF**. Basta con decodificar con restricciones sobre las probabilidades de un clasificador cualquiera. Es justo lo que haremos con H/A/D.
- Pérdidas auxiliares (`L_coarse`, `L_emit`) aportan de 3 a 6 puntos. Si hacemos red, conviene añadir pérdidas auxiliares (p. ej. posición del balón, poseedor individual).
- **Usarlo como feature (E5.2):** reconstruir un conjunto de 22 "jugadores" por frame desde los 46 slots (Hungarian matching frame a frame más relleno por interpolación corta), ejecutar `saved/140` en tramos de juego vivo y sacar P(equipo) = suma de probabilidades de las aristas cuyo emisor o receptor es de ese equipo. Meterlo como feature al GBM. Antes, medir su acierto de equipo en Sportec degradado.

### Ball Radar: detalles útiles

- Entrada: ventanas de 10 s, 22 jugadores × (x, y, vx, vy, speed, acc). Set Transformer (invariante al orden) → Bi‑LSTM jerárquica: primero el poseedor, luego la posición del balón.
- Régimen de imputación: con el 20 % del balón visible, la posición sale a ~1,3 m y el equipo al 97,4 %. Nosotros vemos el ~59 % (aunque los huecos están **concentrados** en duelos y zonas densas). **Hipótesis:** un imputador entrenado con huecos estructurados podría dejar el acierto de equipo por encima del 95 % en frames sin balón. Se comprueba en Sportec degradado.
- Entrenado con Metrica Sample → mismo formato de coordenadas que nosotros. Trae el código de preprocesado de Metrica.
- No hay pesos: hay que entrenarlo (cabe en CPU/MPS con 7 partidos, pero es trabajo de varios días).
- Alternativa barata (E3.3): imputación clásica del balón. Interpolación lineal/cúbica en huecos cortos (< 1 s), y en los largos "balón = posición del jugador del equipo que más probablemente lo tiene" según features colectivas.

### Lang 2022: receta del detector vivo/parado

- Features por jugador: x, y, speed, acc. Desfases temporales ±{0,4; 0,8; 2; 4; 20; 40; 80} s. Los más útiles están en ±4 s, sobre todo en el pasado.
- Como no tenemos identidades, **cambiar las features por jugador por agregados de equipo**: media/mediana/percentiles de velocidad por equipo, fracción de jugadores "parados" (< 1,5 m/s), dispersión, centroide, número de jugadores dentro del área, distancia mínima a cada línea, etc. Desfases iguales.
- Post‑proceso: filtro de mediana de 3,64 s (91 frames). Nosotros lo sustituimos por el decodificador semi‑Markov.
- Sus límites: 51 % de inicios a ≤ 1 s y error medio de 1,8 s. Cuidado, porque **spellF1 castiga mucho 1 s de error** (perfecto + 1 s tarde = 0,65). Hay que añadir señales "rápidas" para afinar las fronteras de D:
  - el balón sale del rectángulo del campo (si es visible);
  - la última posición conocida del balón está cerca de una línea;
  - caída brusca y sincronizada de la velocidad de los 20 jugadores (el pitido);
  - colocación de saque (barrera, jugadores en el área para un córner, un jugador en la banda para el saque de banda);
  - inicio de un `SET PIECES` en Smart Tagging.

### Borghesi TCN

- TCN por objeto (embeddings independientes) + agregación invariante (suma / self‑attention / conv2D). Comparan una salida ternaria con dos binarias (vivo/parado; equipo).
- Para nosotros: una TCN sobre **conjuntos de slots enmascarados**. Entrada por slot: (x, y, vx, vy, speed, team, máscara) más el balón (x, y, máscara, tiempo desde la última observación). Pooling por equipo con attention y dos cabezas (`P(vivo)`, `P(H|vivo)`). Ventana ±4–8 s, centrada (el reto es offline, podemos mirar al futuro).
- Prioridad: segundo modelo (E4.x), solo si el GBM se estanca. Con 7 partidos de entrenamiento el riesgo de sobreajuste es alto. Se mitiga con la degradación (aumento de datos) y pre‑entrenando con SkillCorner.

### Vidal‑Codina et al. 2022 ("Automatic event detection in football using tracking data")

- Posesión individual por reglas: el balón entra en el radio de control de un jugador (~1–1,5 m) y hay cambio de dirección/velocidad del balón → toque. Implementado en `databallpy` con umbrales configurables (radio, velocidad del balón, ángulo).
- Para nosotros: generar features `toque_H(t)` y `toque_A(t)` cuando el balón es visible. Son las mejores señales de **dónde exactamente** cambia la posesión. Sospechamos que el "equipo poseedor" del tracking corregido de Metrica se calcula de forma parecida (ver [04](04_como_puntua_spellf1_y_otros_hallazgos.md)).

### Fuentes

- [PathCRF arXiv](https://arxiv.org/abs/2602.12080) · [código + pesos](https://github.com/hyunsungkim-ds/pathcrf)
- [Ball Radar arXiv](https://arxiv.org/abs/2306.08206) · [código](https://github.com/hyunsungkim-ds/ballradar)
- [Lang et al. 2022 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- [Borghesi et al. 2023 (DOI)](https://doi.org/10.1016/j.eswa.2023.119780) · [repositorio PoliTo](https://iris.polito.it/handle/11583/2976470)
- [Peral et al. arXiv 2512.19528](https://arxiv.org/abs/2512.19528)
- [databallpy: posesión de equipo y jugador](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
