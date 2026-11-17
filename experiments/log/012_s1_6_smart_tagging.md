# 012 · S1.6 Smart Tagging frente a la verdad  (2026-10-23, commit ec49d0a)

Comando: `.venv311/bin/python scripts/s1_6_smart_tagging.py`        Config: golpeo = balón a > 5 m/s durante 3 frames saliendo del sitio esperado; equipo de referencia = último toque r 3 m (S1.2)        Semilla: n/a

## Resultado

### 1. ¿Marca SET PIECES el momento del saque?

Las etiquetas SET PIECES duran **35 s** (los saques de centro, 5 s). Golpeo físico detectado en el tracking, en la ventana [inicio − 10 s, fin + 10 s], en frames evaluados:

| Tipo | n | golpeo encontrado | golpeo − inicio de la etiqueta (frames): p25 / mediana / p75 | Team = N/A |
|---|---|---|---|---|
| Saque de banda | 26 | 25 | −94 / +76 / +178 | 100 % |
| Saque de puerta | 10 | 9 | −128 / −22 / +47 | 0 % |
| Córner | 12 | 5 | +135 / +174 / +177 | 0 % |
| Falta | 5 | 5 | +97 / +154 / +287 | 0 % |
| Saque de centro tras gol | 4 | 3 | +80 / **+83** / +84 | 100 % |
| Saque de centro inicial | 1 | 0 | — | 100 % |

**Contra la verdad pública** (4 reanudaciones), inicio del SET PIECES más cercano − reanudación real: −1.892 (no hay etiqueta cerca), +1.133 (tampoco), −305 (una falta que empieza 12 s antes) y **−69** (un saque de puerta, 2,8 s antes).

### 2. ¿Dice bien el equipo?

Equipo con balón según la fase: su `Team` o, en las fases "Defending…", el contrario (llevan al equipo que defiende; cada una ocupa los mismos frames que su fase de ataque gemela).

| Fase | instancias | acuerdo con el último toque (partido completo) | acuerdo con la verdad pública (frames en juego) |
|---|---|---|---|
| Progression / Defending In Middle Third | 66 | **0,82** | 0,73 (2.707 frames) |
| Build Up / Defending In Attacking Third | 22 | 0,74 | 1,00 (296) |
| Creating Chances / Defending In Defensive Third | 28 | 0,72 | 1,00 (189) |
| Set Pieces (con equipo) | 27 | 0,71 | 0,96 (504) |
| Ball In Final Third | 93 | 0,69 | 0,54 (3.039) |
| Ball In The Box | 72 | 0,67 | 0,44 (965) |
| Players In The Box | 97 | 0,66 | 0,46 (1.866) |
| Long Ball | 29 | 0,63 | 0,90 (184) |
| Attacking Transition / Defensive Transition (invertida; antes sin invertir: 0,41 / 0,33) | 24 + 24 | 0,59 | 0,67 (277) |
| Goals | 4 | 0,54 | — |
| **Todas** | | **0,698** | **0,655** (13.496) |

Como referencia, el propio último toque acierta el 0,87 de los frames en juego contra la verdad pública. Hay 36 instancias con Team = N/A: 32 SET PIECES (todos los saques de banda y de centro), 2 Creating Chances y 2 Defending In Defensive Third.

### 3. Cobertura

| | |
|---|---|
| frames evaluados dentro de alguna fase | 75,8 % |
| dentro de algún SET PIECES | 27,6 % |
| sin ninguna etiqueta | 17,7 % |
| balón visible: dentro de fases / sin etiquetas | 62 % / 48 % |
| jugadores lentos (posible parón): dentro de fases / sin etiquetas | 27 % / 55 % |
| D en la verdad pública: dentro de fases / sin etiquetas | 16 % / 31 % |

Gráfico: `experiments/outputs/012/set_piece_offsets.png`. Tablas: `set_pieces.csv`, `set_pieces_by_type.csv`, `team_agreement_by_code.csv`, `summary.json`.

## Hallazgos

1. **SET PIECES no sirve como ancla fina del saque.** El desfase golpeo − etiqueta va de varios segundos antes a varios después según el tipo, con rangos intercuartílicos de 5 a 10 s. Ninguna de las 4 reanudaciones públicas tiene una etiqueta a menos de 2 s. La puerta del plan pedía desfases estables de menos de 1 s y no se cumple. *Matiz:* mi detector de golpeo puede equivocarse (sobre todo en los saques de banda), pero la prueba con la verdad pública no depende de él.
2. **Excepción:** el saque de centro tras gol es estable (golpeo a +83 frames del inicio de la etiqueta, 3 de 3). Es poco frecuente, pero puede anclar un D largo.
3. **El equipo de las fases es una pista moderada:** coincide un 69 % con el último toque y un 65 % con la verdad pública. Funcionan mejor las fases cortas y de construcción (Progression, Build Up, Creating Chances: 72–82 %) que las largas de ataque (Players/Ball In The Box: 44–67 %), que abarcan varias posesiones. Hay que **invertir el equipo de las fases "Defending…" y "Defensive Transition"**.
4. **Que no haya ninguna etiqueta es pista de parón:** sin etiquetas, el posible parón sube del 27 % al 55 % y el D público del 16 % al 31 %.
5. Los saques de banda nunca dicen el equipo (N/A), como avisaba DATA.md.

## Decisión

**Modificar:** Smart Tagging **no entra como ancla dura del decodificador** (M5 en su versión de anclas, descartado). Entra como **features suaves** del modelo por frame:
- fase activa y equipo implícito (con las fases defensivas invertidas);
- "dentro o fuera de alguna etiqueta";
- dentro de un SET PIECES, con su tipo;
- el saque de centro tras gol como ancla blanda.

Con esto termina el diagnóstico de la Fase 1 salvo S1.7 (margen de mejora por tipo de error, con oráculos).

## Corrección tras la revisión de Codex 002 (2026-10-23)

Revisión: [codex_review_002](../reviews/codex_review_002.md), punto 5.

- El texto pedía invertir "Defensive Transition", pero el código solo invertía las fases que empiezan por "Defending". Ya se invierte con un mapa explícito (`DEFENSIVE_CODES`). Sus 24 instancias pasan de 0,41/0,33 a **0,59/0,67**, igual que su gemela. Los agregados suben de 0,693/0,648 a **0,698/0,655**.
- Los agregados cuentan varias veces los frames con fases solapadas: miden acuerdo sobre pares fase‑frame, no una predicción única por frame.
- La conclusión (pista suave, no ancla) no cambia.
