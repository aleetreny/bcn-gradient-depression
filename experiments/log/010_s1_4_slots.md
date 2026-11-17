# 010 · S1.4 Anatomía de los slots  (2026-10-23, commit e4d29d5)

Comando: `.venv311/bin/python scripts/s1_4_slot_anatomy.py`        Config: apagón = < 12 slots visibles        Semilla: n/a
Salida para S2.1: clave `players` en `local_data/degradation_params.json`.

## Resultado

| Métrica (frames evaluados) | BAR‑ATH ATD | ATD sin apagones | Sportec J03WMX (limpio) |
|---|---|---|---|
| slots | 46 | 46 | 32 (uno por persona) |
| fragmentos (tramos visibles de un slot) | 1.219 | 1.224 | 32 |
| fragmentos que nacen por minuto | 11,5 | 12,0 | 0,3 |
| duración del fragmento p10 / p50 / p90 | 2,6 / 35,5 / 285 s | 2,5 / 36,7 / 287 s | 829 / 4.674 / 5.839 s |
| % del tiempo visible en fragmentos < 10 s | 1,3 % | 1,3 % | 0 % |
| % del tiempo visible en fragmentos > 5 min | 46 % | 45 % | 100 % |
| jugadores visibles por equipo (media H / A) | 10,2 / 10,3 | 10,5 / 10,5 | 11 / 11 |
| % frames con < 11 (H / A) | 38 / 34 | 35 / 32 | 0 |
| % frames con ≤ 9 (H / A) | 15 / 13 | 12 / 10 | 0 |
| % frames con > 11, duplicados (H / A) | 8,1 / 4,7 | 8,4 / 4,9 | 0 |
| saltos > 1 m entre frames, por minuto | 2,8 | 2,9 | 0,2 |
| vibración (2.ª diferencia mediana) | 0,0014 m | 0,0014 m | 0,0100 m |

**Duplicados:** en los frames con más de 11 slots, la pareja del mismo equipo más cercana está a 3,3–3,8 m (frente a 5,0–5,2 m con 11), y nacen fragmentos 3–4 veces más a menudo (0,008 frente a 0,002 por frame). Los slots fuera del campo suben del 0,6 % al 1,3–1,5 %. "Dos jugadores del mismo equipo a menos de 1 m" no sirve para detectar duplicados: sale igual en Sportec (6,0 %) que en el ATD (6,4 %), porque son jugadores reales juntos.

**Porteros** (`position_type = Goalkeeper`):

| | slot | visible | fragmentos | dist. mediana a su portería | % a < 20 m de su portería | si falta, ¿otro slot del equipo junto a su portería? |
|---|---|---|---|---|---|---|
| H | 24 | 92,9 % | 11 | 11,0 m | 96,3 % | 0,7 % |
| A | 46 | 92,5 % | 17 | 7,5 m | 88,3 % | 2,9 % |

Ningún otro slot pasa más del 80 % de su tiempo a menos de 20 m de una portería.

**¿Dónde están los que faltan?** Con ≤ 9 jugadores visibles, el centro del equipo está a 11,5 m del balón (en x); con 11 o más, a 10,7 m. La diferencia es mínima: con este dato no se ve que falten sistemáticamente lejos del balón.

Gráfico: `experiments/outputs/010/slots.png`. Tablas: `slots_summary.csv`, `summary.json`.

## Hallazgos

1. **Las identidades duran bastante:** el 45 % del tiempo visible está en fragmentos de más de 5 min, y la mitad de los fragmentos dura más de 37 s. Nacen unos 12 por minuto. Las features invariantes al slot siguen siendo obligatorias, pero hay continuidad para calcular velocidades y trayectorias cortas.
2. **Faltan pocos jugadores:** casi siempre hay 10 u 11 por equipo, con 9 o menos solo el 10–12 % del tiempo fuera de apagones. El jugador perdido no parece estar ligado a la posición del balón, así que **las features colectivas (centroides, líneas) son fiables**.
3. **Los duplicados nacen al reengancharse un jugador:** el tracker abre un slot nuevo cerca antes de cerrar el viejo. Afectan al 5–8 % de los frames. **Riesgo para la regla del "más cercano":** un duplicado puede doblar la presencia de un equipo cerca del balón. Las features de recuento ("n jugadores a menos de 3 m") deben tolerarlo.
4. **El ATD está muy suavizado:** vibra unas 7 veces menos que Sportec. Para degradar Sportec hay que **suavizarlo** (o aceptar la diferencia), no añadirle ruido. Encaja con S1.3: el ATD procesa sus trayectorias.
5. **El slot de portero es un portero de verdad y es estable** (93 % visible, 88–96 % del tiempo junto a su portería). Cuando falta, ningún otro slot ocupa su sitio. → feature de portero fiable: posición, si tiene el balón en las manos, saques de puerta.
6. Hay 2,9 saltos imposibles (> 1 m en un frame) por minuto, casi siempre en reenganches. Conviene tratarlos como cortes de fragmento al calcular velocidades.

## Decisión

Seguir. Para la degradación (S2.1) se usarán: fragmentación de unos 12 fragmentos/min, 10–11 jugadores por equipo, duplicados en el 5–8 % de los frames y Sportec suavizado. Feature de portero confirmada.
