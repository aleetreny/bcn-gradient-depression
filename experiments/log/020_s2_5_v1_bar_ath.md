# 020 · S2.5 v1 sobre el BAR‑ATH  (2026-10-23, commit 2487cd0 / 843915c; tag `v1`)

Comandos: `.venv311/bin/python scripts/make_submission_v1.py` (construcción de principio a fin) y `.venv311/bin/python scripts/s2_5_v1_report.py`
Config: `configs/degrade_v2.json` + `configs/decoder_v1.json`        Semillas: modelo 99; degradación 1, 2 y variante; ventanas visuales 5

## Qué es la v1

GBM de dos cabezas (S2.3, 160 features v1) entrenado con los 7 partidos de Sportec degradados, más el decodificador semi‑Markov (S2.4) y dos restricciones duras: D fuera de las partes y en la 2.ª parte antes del saque (frames 110.808–113.740). LightGBM en modo determinista. **Reproducible:** dos construcciones seguidas dan el mismo fichero, SHA‑256 `4491ebe4408a769c64a7625c866a67dc1275302df6eccd90f9a8ffd0f2ddf020` (`submissions/v1.csv`, fuera de git).

## Resultado

**Validez:** `validate_submission.py` → OK. 199.319 filas: H 26,5 %, A 15,1 %, D 58,4 % (incluye los frames no evaluados), 320 posesiones de 10,4 s de media.

**Públicos:** spellF1 **0,519 [0,368–0,660]**, macroF1 0,865, F1 vivo/parado 0,968, acierto de equipo 0,901; 20 posesiones (verdad 17).

**Panel de plausibilidad** (media de tramos de 15 min):

| | pos./min | % D | mediana de posesión | % < 2 s |
|---|---|---|---|---|
| **BAR‑ATH v1** | **3,14** | **48,6** | 5,2 s | 18,7 |
| BAR‑ATH baseline | 5,75 | 32,8 | 4,3 s | 32,4 |
| Sportec, el mismo sistema (fuera de fold) | 3,90 | 45,3 | 4,9 s | 22,3 |
| Sportec, verdad | 4,68 | 44,7 | 3,8 s | 29,4 |

Por tramo (BAR‑ATH v1): % D de 23,5 / 47,2 / 61,2 / 59,8 en la 1.ª parte (0–15', 15–30', 30–45', añadido) y 42,3 / 43,1 / 62,9 / 49,0 en la 2.ª. Posesiones/min entre 2,2 y 4,9.

**Saques (cobertura):** la v1 predice algún D dentro de [inicio − 10 s, fin] del **98 %** de los 58 SET PIECES evaluados (baseline 91 %). *Ojo:* predecir siempre D también daría 100 %; es cobertura de intervalos amplios, no precisión ni fronteras. Por tipo: córner, falta, saque de puerta y saque de centro 100 %; saque de banda 96 %.

**Revisión visual** (3 ventanas de 2 min al azar, fuera de los públicos; `window_1..3.png`):
1. **Frame 22.114:** la v1 es estable donde el baseline parpadea, con un D de ~10 s en un hueco largo del balón. Razonable.
2. **Frame 140.370:** posesiones largas de H con cambios cortos de A y un D de ~20 s en un hueco del balón. Razonable.
3. **Frame 160.684:** **95 s seguidos de D** dentro del apagón del minuto de vídeo ~106 (sin jugadores ni balón). La v1 y el baseline coinciden.

## Hallazgos

1. **Puerta superada:** válida, reproducible al byte y plausible. **Queda congelada como red de seguridad (tag `v1`).**
2. **Se queda corta en posesiones:** en Sportec predice 3,9 por minuto cuando la verdad tiene 4,7 (un 17 % menos), y en el BAR‑ATH 3,1. Encaja con S2.4: el decodificador une posesiones y sigue faltando detectar cambios, sobre todo los cortos (19 % de posesiones < 2 s frente al 29 % de la verdad en Sportec).
3. **% de D plausible en total (49 %), pero con tramos del 60–63 %.** Parte viene de los apagones: **ante un apagón el sistema predice D por defecto**. Si en esos minutos el juego seguía, ahí se pierde. Riesgo a estudiar en la Fase 3 (prior del decodificador en apagones o uso del vídeo).
4. Detecta los saques mucho mejor que el baseline (98 % frente a 91 %).

## Decisión

**Seguir → S2.6** (descomposición de errores de la v1 con los oráculos de S1.7) para reordenar la Fase 3. El tablero marca la v1 como red de seguridad.

## Corrección tras la revisión de Codex 003 (2026-10-23)

Revisión: [codex_review_003](../reviews/codex_review_003.md).

- **Saques:** el 98 % es solo cobertura, como se aclara arriba.
- **Construcción desde cero:** `make_submission_v1.py` necesitaba un JSON no versionado (`local_data/degradation_params.json`). Ahora lo regenera desde el ATD crudo si falta. Verificado en una copia limpia (`git archive`, sin cachés): mismo SHA‑256 `4491ebe4…`; hash de los parámetros `ed08956d…`.
- **Caché de features:** ahora se identifica por el contenido del partido (posiciones, balón, slots, partes) y la versión del código. Ya no puede devolver features viejas si cambia una config.
- **Panel de Sportec "el mismo sistema":** usa el decodificador ajustado con los mismos 7 partidos, así que es optimista. Se repetirá con las predicciones por fold del LOMO anidado (S2.4b).
