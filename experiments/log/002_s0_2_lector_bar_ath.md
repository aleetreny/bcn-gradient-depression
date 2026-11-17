# 002 · S0.2 Lector BAR‑ATH  (2026-10-21, commit 588c778)

Comando: `.venv311/bin/python scripts/report_s0_2_bar_ath.py`        Config: n/a        Semilla: n/a
Código: `src/data/metrica_atd.py` · tests: `tests/test_metrica_atd.py` (5 verdes, datos sintéticos)

## Resultado

Parseo completo del `.txt` en **2 s** (pandas C sobre el texto con los separadores cambiados a comas). Cache `.npz` en `local_data/cache/`.

| Comprobación | Nuestro lector | DATA.md / doc 06 |
|---|---|---|
| frames (ids 1…199.319 contiguos) | 199.319 | 199.319 |
| frames evaluados | 158.857 | 158.857 |
| slots (H / A) | 46 (24 / 22) | 46 |
| balón x, y visible, total | 113.488 (56,9 %) | 113.488 (56,9 %) |
| balón x, y visible, evaluados | 94.214 (59,3 %) | 94.214 (59,3 %) |
| objetos visibles de media, evaluados | 20,49 | 20,5 |
| objetos visibles de media, total | 18,98 | 18,98 |
| patrones del balón (xy / xy+c3 / nada / solo c3) | 52,0 / 5,0 / 37,5 / 5,5 % | 52,0 / 5,0 / 37,5 / 5,5 % |

**Contraste independiente con kloppy 3.18** en 3.987 frames muestreados por todo el partido (3.177 evaluados, 75.657 posiciones de jugador): diferencia máxima 1,1e‑7 (redondeo float32), patrón de NaN idéntico, velocidad idéntica. Esto confirma también que fila = `frame_id − 1`.

**Dirección de ataque** (x media del portero, campo de 105,3 m):

| | 1.ª parte crudo | 1.ª parte normalizado | 2.ª parte crudo | 2.ª parte normalizado |
|---|---|---|---|---|
| Portero H (slot 24) | 10,7 | 10,7 | 93,4 | **11,9** |
| Portero A (slot 46) | 89,2 | 89,2 | 18,5 | **86,8** |

Tras normalizar, H defiende siempre x ≈ 0 y ataca hacia +x. `position_type = Goalkeeper` identifica de verdad a los porteros (slots 24 y 46).

Gráfico: `experiments/outputs/002/overview.png` (balón visible, tercer canal y slots por equipo, por minuto). Datos: `summary.json`, `report.log`.

## Hallazgos

1. **`speed_kph` ya está en m/s para los jugadores.** Mediana velocidad declarada / desplazamiento entre frames = 0,992 (r = 0,93, 2,9 M pares); p50 1,4 m/s, p99 6,4 m/s, máx. 9,4 m/s. El doc 06 decía dividir entre 3,6, lo que habría dejado todas las velocidades 3,6 veces más bajas. Corregido en el lector y en el doc 06.
2. **El tercer canal del balón no es una velocidad.** No correlaciona con el desplazamiento del balón (r = −0,09), cambia muy despacio (mediana 0,013 por frame) y aparece solo en 51 tramos (mediana 7,5 s, máx. 4,7 min), sobre todo cuando el tracking va mal. Se guarda crudo, sin escalar. Hipótesis para S1.3: un indicador de calidad o de incertidumbre del tracker.
3. **Hay 3 apagones de tracking de ~1 min durante el juego** (minutos de vídeo 39, 59 y 106), con < 12 slots visibles y balón visible solo el 4,5 % del tiempo. Frames evaluados con < 12 slots: 5.747 (3,6 %). Con ≥ 16 slots el balón se ve el 62,7 % del tiempo. ¿Repeticiones o cortes de la cámara táctica? Se estudia en S1.3/S1.4. En esos tramos el modelo tendrá que apoyarse en el decodificador y en Smart Tagging.
4. La visibilidad del balón cambia mucho de un minuto a otro (del 0 % al 100 %): hay minutos enteros de juego casi sin balón aparte de los apagones.
5. En el descanso el balón se ve casi siempre, pero hay pocos jugadores. No se evalúa, así que no importa.

## Decisión

Seguir → S0.3 Lector Sportec (mismo formato `Match`). Pendiente para la Fase 1: significado del tercer canal del balón y naturaleza de los apagones. Las vx, vy suavizadas (Savitzky‑Golay) se calcularán en `src/features/` (S2.2), no en el lector.
