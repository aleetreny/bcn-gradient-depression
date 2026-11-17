# 06 · Cómo son nuestros datos (BAR‑ATH) y cómo tratarlos

> Este análisis inicial se elaboró con la estructura de los archivos mediante [`scripts/inspect_data_structure.py`](../../scripts/inspect_data_structure.py). Incluye formatos, recuentos y patrones de valores faltantes.

## Resumen

- Tenemos en local 4 ficheros del vídeo táctico: **tracking** (196 MB), **metadatos del tracking**, **Smart Tagging** y **homografía**. No están los vídeos ni nada de la retransmisión.
- El tracking es un texto con **una línea por frame (199.319)**. Cada línea trae siempre **46 "huecos" de jugador** con (x, y, velocidad) y **el balón** con (x, y, velocidad). Si algo no se ve, sus campos vienen vacíos.
- **El tracking NO trae ni "balón vivo/muerto" ni "equipo con balón".** Hay que inferir las dos cosas, que son justo la respuesta.
- El balón falta en el **43 %** de los frames. En un **5,5 %** falta la posición pero viene el tercer canal del balón. **Ese canal no es una velocidad** (S0.2, [log 002](../../experiments/log/002_s0_2_lector_bar_ath.md)): no correlaciona con el desplazamiento del balón y aparece en pocos tramos largos. Su significado queda para S1.3.
- La velocidad de los jugadores **ya viene en m/s**, aunque el XML diga `speed_kph` (comprobado en S0.2 contra el desplazamiento entre frames: ratio 0,99). **No dividir entre 3,6.** Las coordenadas van de 0 a 1 y hay que multiplicarlas por 105,3 × 68 m.
- Los metadatos traen la **dirección de ataque de cada equipo en la 1.ª parte**, útil para normalizar el campo.

## Desarrollo

### Inventario local (`data/tactical/`)

| Fichero | Tamaño | Estructura |
|---|---|---|
| `BAR-ATH_tactical_atd.txt` | 195,7 MB | 199.319 líneas, `frame:slots:balón` |
| `BAR-ATH_tactical_atd.xml` | 59 KB | Metadatos FIFA EPTS |
| `BAR-ATH_tactical_smart_tagging.xml` | 261 KB | SportsCode XML, 638 instancias |
| `BAR-ATH_tactical_homography.csv` | 23,2 MB | 199.319 filas × 12 columnas |
| `BAR-ATH_tactical.mp4` y `broadcast/` | — | **No descargados** (5,2 GB + 14 GB). Solo hacen falta para E6.x |

### Tracking crudo (`atd.txt`)

```
<frame_id> : <slot1>;<slot2>;...;<slot46> : <balón>
slot  = x,y,speed     (3 campos, siempre presentes; vacíos si el slot no se ve)
balón = x,y,speed     (3 campos)
```

- `frame_id` va de 1 a 199.319. **Fila de la submission = `frame_id − 1`.**
- Todas las líneas tienen exactamente 3 secciones y 46 entradas de 3 campos (9.168.674 entradas en total). El formato es fijo y fácil de leer con numpy, sin necesidad de kloppy.
- Canales (del XML): `position_x` (normalized), `position_y` (normalized), `speed_kph`. **El nombre engaña:** para jugadores el valor ya está en m/s (S0.2: mediana velocidad/desplazamiento = 0,99; kloppy devuelve el mismo número sin convertir, de ahí que DATA.md hable de m/s). Para el balón el canal no es una velocidad.
- Slots visibles por frame: media de **18,98** sobre todos los frames (DATA.md: 20,5 sobre los evaluados). Llega hasta 35 cuando hay duplicados.

**Patrones de huecos del balón** (x, y, velocidad):

| Patrón | Frames | % |
|---|---|---|
| x, y presentes; velocidad vacía | 103.579 | 52,0 % |
| x, y y velocidad presentes | 9.909 | 5,0 % |
| todo vacío | 74.769 | 37,5 % |
| **x, y vacíos pero velocidad presente** | 11.062 | **5,5 %** |

→ Balón con posición: 113.488 (56,9 %), igual que DATA.md. La velocidad del balón casi nunca viene junto con la posición, pero en 11k frames viene **sin** posición. La hipótesis inicial (el tracker extrapola el balón) queda en duda tras S0.2: el canal no correlaciona con la velocidad del balón (r = −0,09), varía muy despacio y sale en 51 tramos de mediana 7,5 s (máx. 4,7 min), concentrados en tramos con tracking degradado. **Feature provisional:** `ball_speed_only` (máscara) más el valor crudo; su significado se estudia en S1.3.

### Metadatos (`atd.xml`, FIFA EPTS)

- `Teams/Team` ×2, cada uno con el parámetro `attack_direction_first_half`. **Usarlo para normalizar** que cada equipo ataque siempre hacia +x (se invierte en la 2.ª parte).
- `Players/Player` ×46 con `teamId` → **equipo de cada slot** (22 slots de un equipo y 24 del otro). `Name` / `ShirtNumber` = índice del slot, no son personas.
- Parámetros por slot: `position_type` y `position_index` con **2 valores distintos** cada uno, y `position_x` / `position_y` casi constantes. Probablemente distinguen al **portero** del resto. Comprobarlo en local (por ejemplo, si el slot está siempre cerca de su portería); si es así, es una feature útil.
- Parámetros globales: `first_half_start/end` rellenos; `second_half_*` y prórrogas vacíos. Las fronteras oficiales ya están en `games.half_video_blocks`.
- `FieldSize`: 105 × 68 en metadatos; `games.py` usa 105,3 × 68,0, medido con el tracking corregido. Usar el de `games.py`.
- `TrackingType = Optical`, `FrameRate = 25`.

### Smart Tagging (`smart_tagging.xml`)

```
file/ALL_INSTANCES/instance ×638 : ID, code, start, end, label×N (group, text)
file/ROWS/row ×14                : code + color R,G,B   (catálogo de los 14 códigos)
```

`start` y `end` son segundos desde el frame 0 del vídeo táctico (frame = `round(s·25)`). Cada `label` tiene un `group` (Team, Half, Type, Side…) y un `text`. Hay 1.762 labels, unas 2,8 por instancia. Loader sin dependencias: `starter/load_smart_tagging.py`.

### Homografía (`homography.csv`)

Columnas `frame, h00…h22, reprojection_error, valid`, 199.319 filas, `frame` = fila de la submission. Solo hace falta si se proyecta al vídeo (E6.1). Para todo lo demás se trabaja en coordenadas de campo.

### Cómo tratarlos (pipeline de carga recomendado)

1. **Parser numpy** del `.txt` → arrays `P[199319, 46, 3]` (x, y, speed) y `B[199319, 3]`, con NaN donde haya vacíos. Guardarlo como `.npz` en una carpeta local ignorada (p. ej. `local_data/cache/`), nunca en git.
2. Máscaras: `slot_visible[t, s]`, `ball_xy_visible[t]`, `ball_speed_only[t]`.
3. Metros: `x·105,3`, `y·68,0`; la velocidad de jugadores ya está en m/s (sin `/3,6`). **Recalcular vx, vy** con diferencias centradas y un suavizado ligero (Savitzky‑Golay, ventana de unos 5–7 frames), solo dentro de tramos contiguos visibles del mismo slot.
4. Equipo por slot desde `Player@teamId`. Saber cuál es Barça (H) leyendo `Team/Name`.
5. Normalizar la dirección: con `attack_direction_first_half`, reflejar x (e y) para que H ataque siempre hacia +x en ambas partes.
6. Features **invariantes al slot** (agregados por equipo, k vecinos más cercanos al balón), como se describe en [05_plan_para_ganar.md](05_plan_para_ganar.md). Nunca usar el índice de slot como identidad.
7. **Mismo código para Sportec:** convertir Sportec al mismo formato `P`/`B` (sin identidades, slots anónimos) para que el modelo vea exactamente la misma representación.

### Reglas: qué puede verse y dónde están escritas

- **Datos de la competición:** se usan para las actividades del hackathon; mantener los archivos licenciados separados del repositorio público.
- [RULES.md](../../RULES.md), sección "The data": no redistribuir y no publicar datos derivados que permitan reconstruirlos.
- Nuestro [README.md](../../README.md) y [docs/LOCAL_DATA_SETUP.md](../../docs/LOCAL_DATA_SETUP.md) documentan las rutas de datos y la organización del trabajo.
- Criterio de este documento: describir la estructura y los patrones del conjunto de datos, manteniendo los archivos licenciados y derivados reconstruibles fuera de Git.
