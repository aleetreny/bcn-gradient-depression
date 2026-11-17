# 009 · S1.3 Anatomía de los huecos del balón  (2026-10-23, commit bd9c087)

Comando: `.venv311/bin/python scripts/s1_3_ball_gaps.py`        Config: parón aproximado = velocidad media de los jugadores < 1,2 m/s (media móvil de 2 s); apagón = < 12 slots        Semilla: n/a
Salida para S2.1: `local_data/degradation_params.json` (solo agregados; copia en `experiments/outputs/009/`).

## Resultado

**Indicador de parón, validado en los 5 min públicos** (no hay verdad del partido completo): P(lento | D) = 0,49, P(lento | juego) = 0,045 y P(D | lento) = **0,73**. Es un indicador razonable, pero no la verdad: detecta la mitad de los D y acierta 3 de cada 4 veces.

**Los huecos son pocos y largos.** En los frames evaluados hay 150 huecos (64.643 frames, el 40,7 %), con duraciones p25/p50/p75/p90/p99 de 6,1 / **10,7** / 21,1 / 33,8 / 104,8 s.

| Duración | huecos | % de los frames sin balón | velocidad del balón al perderse (m/s, mediana) | jugadores a < 3 m | % cerca de una línea (< 3 m) | % apagón | % lento | % con tercer canal | % con SET PIECES dentro |
|---|---|---|---|---|---|---|---|---|---|
| < 0,5 s | 16 | 0,1 | 4,0 | 2 | 6 | 0 | 6 | 6 | 38 |
| 0,5–1 s | 1 | 0,0 | 0,7 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1–2 s | 0 | — | — | — | — | — | — | — | — |
| 2–5 s | 1 | 0,2 | 1,2 | 1 | 0 | 0 | 100 | 0 | 0 |
| 5–10 s | 53 | 14,5 | 2,2 | 1 | 28 | 0 | 31 | 7 | 47 |
| 10–30 s | 60 | 41,4 | 1,8 | 1 | 42 | 2 | 57 | 11 | 43 |
| > 30 s | 19 | 43,8 | 1,4 | 1 | 32 | 12 | 73 | 13 | 74 |

**Dónde caen los frames sin balón:**

| Contexto | huecos | minutos sin balón | mediana | mismo equipo más cercano antes y después | balón visible en ese contexto |
|---|---|---|---|---|---|
| Juego en movimiento | 83 | **13,3** | 7,6 s | 57 % | **78 %** (69 min de juego en movimiento) |
| Lento (posible parón) | 64 | 26,6 | 17,0 s | 61 % | 26 % |
| Apagón (< 12 slots) | 3 | 3,2 | 37,2 s | 50 % | 5 % |

De los frames sin balón, el 56 % cae en tramos lentos, el 8,5 % en apagones y el **36 % en juego en movimiento** (23.015 frames, unos 15 min).

**Cuándo empieza un hueco** (probabilidad de que empiece en un frame con balón visible):
- **Según la velocidad del balón:** 0,0040 a 0–2 m/s; 0,0017 a 2–5; 0,0008 a 5–10; 0,0003 a 10–15; 0,0001 a 15–25 m/s.
- **Según los jugadores a menos de 3 m:** 0,0014 con 0–1 jugadores; 0,0024 con 2; 0,0042 con 3–4.

**Tercer canal del balón:** de sus 6.576 frames evaluados, el 88 % cae dentro de huecos de más de 10 s y el 68 % en tramos lentos.

**Transiciones públicas dentro de huecos:** 1 de 20, un inicio de D (#5 de S1.1). Ningún cambio de equipo.

Gráfico: `experiments/outputs/009/ball_gaps.png`. Tablas: `gaps_by_length.csv`, `gaps_by_context.csv`, `summary.json`.

## Hallazgos

1. **El tracker del balón funciona "a todo o nada":** casi no hay huecos de entre 0,5 y 5 s. Todo el balón que falta está en huecos de 5 s o más. Lo más probable es que el ATD **rellene él mismo los huecos cortos** (interpolación). Eso explicaría que el balón esté "suavizado" y que los toques necesiten un radio de 2–3 m (S1.2). *Hipótesis.*
2. **El balón se pierde sobre todo cuando va lento y entre varios jugadores, no en los balones rápidos.** La probabilidad de que empiece un hueco es unas 10 veces mayor a menos de 2 m/s que a 10–15 m/s, y unas 3 veces mayor con 3–4 jugadores cerca. Es lo contrario de lo que suponía el plan (balones en el aire).
3. **Más de la mitad de los frames sin balón (56 %) caen en tramos lentos**, que en público son D 3 de cada 4 veces. **Que falte el balón durante mucho tiempo es una buena señal de D**, junto con el tercer canal (68 % en tramos lentos).
4. **Premio de rellenar el balón:** unos 13 min de juego en movimiento sin balón (el 16 % del juego en movimiento), en 83 huecos de mediana 7,6 s. En 36 de esos 83 huecos (43 %) el equipo del jugador más cercano es distinto a un lado y a otro del hueco. *(Corregido: esto no es un recuento de posesiones reales ocultas; ver abajo.)*
5. **La interpolación clásica de huecos de menos de 1 s (M2a) no sirve aquí:** solo hay 75 frames en huecos de menos de 1 s. Para recuperar esos 13 min hacen falta métodos que adivinen el balón en huecos largos: Ball Radar (M2b), trayectoria coherente (M2c) o el vídeo (M13).
6. **Consecuencia para la degradación de Sportec (S2.1):** hay que imitar huecos **largos** (de mediana unos 10 s y mínimo unos 5 s), más probables con el balón lento, con varios jugadores cerca y en los parones. Nada de huecos cortos al azar.

## Decisión

**Seguir**, con dos cambios de prioridad:
- **M2a se descarta**, porque no hay huecos cortos que rellenar.
- La **D se apoya en "hueco largo + jugadores lentos + tercer canal"** (M4b).

La prioridad principal sigue siendo clavar el momento del cambio (S1.1 y S1.2), porque el 78 % del juego en movimiento tiene el balón visible. Siguiente: S1.4 (anatomía de los slots).

## Corrección tras la revisión de Codex 002 (2026-10-23)

Revisión: [codex_review_002](../reviews/codex_review_002.md), punto 8.

- El 43 % es **desacuerdo del "más cercano" entre los dos extremos del hueco**, no un límite inferior de cambios reales de posesión. Un pase entre compañeros puede dejar momentáneamente a un rival como el más cercano en un extremo (S1.1 ya mostró que esa regla falla con el balón visible). Que el equipo coincida tampoco descarta varios cambios dentro del hueco.
- "Lento" y "en movimiento" son **indicadores**, no D y juego confirmados: el indicador detecta la mitad de los D públicos y no se sabe cómo se comporta en los huecos largos del partido completo. Los parámetros de degradación conservan esos nombres.
