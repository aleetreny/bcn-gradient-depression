# 02 · Datos externos con etiquetas de posesión

## Resumen

- **Problema:** solo tenemos 5 minutos etiquetados (17 posesiones). Eso no basta ni para entrenar ni para elegir entre dos versiones de nada.
- **Solución:** entrenar y validar con partidos **públicos** que ya traen la posesión etiquetada, y usar nuestro partido solo para la predicción final.
- **Dataset principal: Sportec/DFL.** 7 partidos completos, 25 fps (igual que el nuestro), y por cada frame `vivo/parado` y `equipo poseedor`. Es literalmente H/A/D gratis, con licencia CC‑BY 4.0.
- **Problema de Sportec:** es tracking "limpio" (balón casi siempre visible, jugadores identificados). El nuestro pierde el balón el 41 % del tiempo y no tiene identidades. Por eso hay que **estropear Sportec a propósito** para que se parezca al nuestro antes de entrenar.
- **Secundarios:** Metrica Sample (mismo proveedor y formato que el nuestro, pero hay que derivar las etiquetas) y SkillCorner (tracking de retransmisión imperfecto, 10 fps).
- **Reglas:** el reto permite datos externos si son accesibles y se documentan. Los datos externos van en una carpeta local ignorada por git y se descargan con un script.

## Desarrollo

### Ranking de utilidad

| # | Dataset | Etiquetas | Encaje | Uso |
|---|---|---|---|---|
| 1 | **Sportec Open DFL** (IDSSE) | `BallStatus` + `BallPossession` por frame | Alto (25 fps, mismas 3 clases) | Entrenar los clasificadores, ajustar el decodificador **directamente con spellF1**, validación LOMO |
| 2 | **Metrica Sample Data** (3 partidos) | Eventos → derivar H/A/D | Alto en formato y proveedor, medio en etiquetas | Probar loaders EPTS, adaptación al proveedor, 3 folds extra si la derivación es fiable |
| 3 | **SkillCorner Open Data** (10 partidos) | `possession.group` por frame; D = fuera de `phases_of_play` | Medio (10 fps, retransmisión) | Robustez ante jugadores no detectados; pre‑entrenamiento del módulo de equipo |
| 4 | SoccerNet / SoccerTrack v2 | Acciones en vídeo | Bajo | Solo si se hace visión por computador |

### Sportec: cómo cargarlo

```python
# pip install kloppy
from kloppy import sportec
MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
ds = sportec.load_open_tracking_data(match_id="J03WMX")   # descarga y cachea
df = ds.to_df()   # frame_id, period_id, timestamp, ball_x, ball_y, ball_z, <player>_x/_y/_s ...,
                  # ball_state ('alive'|'dead'), ball_owning_team_id
```

Etiqueta: `D` si `ball_state == 'dead'`; si no, `H`/`A` según `ball_owning_team_id` sea el local o el visitante (`ds.metadata.teams[0]` es local en kloppy, **comprobarlo**).

Alternativa: `floodlight.io.datasets.IDSSEDataset(match_id="all")` devuelve los objetos `Code` de posesión y estado del balón.

Ubicación: `data_external/sportec_dfl/` (ignorada por git). Descarga: `bash scripts/download_public_datasets.sh sportec`. Los XML se leen con `kloppy.sportec.load_tracking(...)` apuntando a los ficheros locales, sin volver a descargarlos.

### Ya descargado: `data_external/` (local, ignorado por git)

Se regenera con `bash scripts/download_public_datasets.sh [sportec|metrica|skillcorner|pathcrf|ballradar]`. El script reanuda descargas a medias y comprueba el md5 de Sportec.

```
data_external/
├─ sportec_dfl/            21 XML de figshare (~2,6 GB): por partido, matchinformation + events_raw + positions_raw_observed
├─ metrica_sample_data/    3 partidos
│   ├─ Sample_Game_1|2/    tracking CSV (Home/Away por separado) + RawEventsData.csv
│   └─ Sample_Game_3/      tracking EPTS .txt + metadata.xml + events.json
├─ skillcorner_opendata/   10 partidos A-League (tracking .jsonl vía git-lfs, eventos dinámicos, fases), bodypose
├─ pathcrf_repo/           código + 5 checkpoints (saved/100…140, ~37 MB cada uno)
└─ ballradar_repo/         código + eventos de Metrica preprocesados (sin checkpoints)
```

Notas al inspeccionar lo descargado:

- **Metrica Sample Game 3** usa el mismo formato EPTS que nuestro ATD (`frame:jugadores:balón`, separados por `;`), pero **solo con x, y** (sin velocidad) y con jugadores identificados. Un mismo parser sirve para los dos cambiando el número de campos.
- **Eventos de Metrica Sample** (Game 1): PASS 799, RECOVERY 278, BALL LOST 257, CHALLENGE 233, SET PIECE 77, BALL OUT 51, SHOT 24, FAULT RECEIVED 22, CARD 4. Con `RECOVERY` / `BALL LOST` (cambio de equipo) y `BALL OUT` / `FAULT RECEIVED` → `SET PIECE` (tramos D) se puede derivar H/A/D por frame (E1.6).

### Comprobaciones obligatorias antes de entrenar (E1.x)

1. **Semántica de cambio de posesión (crítico).** Medir en Sportec la distribución de duración de las posesiones (rachas máximas H o A) y compararla con la verdad pública (mediana ≈ 8–10 s, mínimas de 0,76 s y 1,32 s, ≈ 3,4 posesiones/min de juego). Si Sportec tiene muchas menos posesiones cortas, su `BallPossession` es más "lento" que la de Metrica y habrá que **recalibrar el decodificador** en lugar de copiar sus duraciones.
2. **Momento del cambio:** en Sportec, alinear los cambios de `BallPossession` con eventos de pase/recuperación sincronizados y medir el desfase típico (¿cambia en la recepción? ¿en el pase interceptado?).
3. **D:** proporción de frames D y duración de los tramos D frente a la verdad pública (4 tramos de 5,5–20,4 s en 5 min).
4. **Posesión durante D:** Sportec mantiene un equipo durante el balón parado (igual que el tracking de Metrica según CHALLENGE.md). La etiqueta D **pisa** al equipo.
5. Huecos y NaN por partido, cambios de periodo y orientación (los equipos cambian de campo en la 2.ª parte: normalizar la dirección de ataque).

### Degradación de Sportec a "estilo Metrica" (clave para transferir)

Medir sobre nuestro ATD para estimar las distribuciones de huecos, identidades y ruido:

- Distribución de la **longitud de los huecos de balón** (rachas NaN) y su probabilidad condicionada a: densidad de jugadores alrededor del último balón visto, altura/velocidad del balón, zona del campo.
- Número de objetos por frame (media 20,5), tasa de aparición/desaparición de slots y duplicados (>22).
- Ruido posicional: compararlo con el suavizado de Sportec (Sportec está filtrado, el ATD no).

Luego, al entrenar con Sportec:

- **Ocultar el balón** con un proceso de rachas (p. ej. semi‑Markov ajustado a esas longitudes), con más probabilidad en zonas congestionadas y en balones rápidos. No usar un enmascaramiento i.i.d. por frame: Peral et al. y Ball Radar indican que el enmascaramiento estructurado es lo que generaliza.
- **Romper identidades:** reasignar jugadores a slots anónimos que nacen y mueren (cortar trayectorias en fragmentos con longitudes como las del ATD), borrar 0–3 jugadores al azar por tramo y duplicar alguno.
- Añadir ruido gaussiano o de "saltos" a las posiciones y **recalcular velocidades desde posiciones ruidosas**, igual que haremos en Metrica.

Validar siempre en dos condiciones: Sportec limpio y Sportec degradado. Solo cuenta lo que mejora en el **degradado**.

### Metrica Sample: derivación de etiquetas (opcional, E1.6)

Los eventos de Metrica traen tipos como PASS, RECOVERY, BALL LOST, BALL OUT y SET PIECE, con su frame. Las etiquetas se derivarían así:

- D desde `BALL OUT` / falta hasta el siguiente `SET PIECE`.
- Equipo = el del último evento con control, cambiando en `RECOVERY` o en la recepción rival.

Tiene ruido, pero viene del **mismo proveedor**. Útil como 8.º–10.º fold de validación o para fine‑tuning. Ball Radar ya trae eventos de Metrica preprocesados en `data/metrica_events/`.

### SkillCorner (opcional)

`{id}_tracking_extrapolated.jsonl`: `possession.group` ∈ {home, away}, `player_data[].is_detected`. D = frames fuera de cualquier fase de `{id}_phases_of_play.csv`. Remuestrear a 25 fps (o trabajar a 10 fps y escalar). Útil para entrenar robustez ante jugadores extrapolados o ausentes.

### Licencias y documentación

Sportec: CC‑BY 4.0, citar a Bassek et al. 2025. SkillCorner: MIT en el repo, pide crédito. Metrica Sample: uso responsable con atribución. Documentarlo todo en el informe técnico (RULES.md lo exige).

### Fuentes

- [Bassek et al. 2025, Scientific Data (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/) · [figshare](https://doi.org/10.6084/m9.figshare.28196177) · [kloppy Sportec](https://kloppy.pysport.org/user-guide/loading-data/sportec/) · [floodlight IDSSEDataset](https://floodlight.readthedocs.io/en/latest/modules/io/datasets.html)
- [SkillCorner Open Data](https://github.com/SkillCorner/opendata)
- [Metrica Sample Data](https://github.com/metrica-sports/sample-data)
