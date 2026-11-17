# 004 · S0.4 Lector Metrica Sample  (2026-10-22, commit d2cf71b)

Comando: `.venv311/bin/python scripts/report_s0_4_metrica_sample.py`        Config: n/a        Semilla: 0 (orden de slots)
Código: `src/data/metrica_sample.py` · tests: `tests/test_metrica_sample.py` (3 verdes, eventos sintéticos)

## Resultado

Tracking con kloppy (CSV en los partidos 1–2, EPTS en el 3; ~10 s por partido la primera vez, luego cache). Etiquetas derivadas de los eventos con estas reglas:
- El equipo con balón es el del último evento con balón (PASS, CARRY, SHOT, RECOVERY, SET PIECE, BALL LOST).
- BALL LOST cede el balón al rival en su frame final.
- D va desde BALL OUT, tiro fuera o gol, falta, o BALL LOST por fuera de juego, mano o fin de parte, hasta el siguiente SET PIECE o el pase de saque de puerta.
- Una RECOVERY con el balón parado (el portero recoge un balón que salió) no reanuda el juego.

| Partido | min | % D | posesiones | pos./min | media (s) | mediana (s) | % < 2 s | tramos D | D media (s) | balón visible | reinicios sin registrar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 94,7 | 39,3 | 369 | 3,90 | 9,3 | 4,8 | 23,0 | 89 | 25,1 | 62,1 % | 11 |
| 2 | 93,4 | 36,6 | 356 | 3,81 | 10,0 | 4,7 | 28,7 | 90 | 22,8 | 59,4 % | 9 |
| 3 | 95,1 | 27,6 | 390 | 4,10 | 10,6 | 5,4 | 23,6 | 90 | 17,5 | 68,3 % | 4 |

**Comparación de las tres fuentes de etiquetas:**

| Fuente | % D | pos./min | mediana (s) | media (s) | % < 2 s | D media (s) | % cambios directos |
|---|---|---|---|---|---|---|---|
| Sportec (7) | 44,9 | 4,58 | 3,7 | 7,2 | 30,0 | 25,3 | 76,8 |
| **Metrica Sample (3, derivadas)** | **34,5** | **3,94** | **4,9** | **10,0** | **25,0** | 21,8 | 76,1 |
| BAR‑ATH públicos (5 min) | 19,6 | 3,40 | 7,6 | 14,2 | 11,8 | 14,7 | 75,0 |

**Eventos y tracking comparten reloj:** en el frame de inicio de cada evento con balón, el balón está a una mediana de **8–9 cm** del jugador más cercano del equipo que actúa, y el 95–96 % de las veces a menos de 2 m. Si se desplazan los eventos ±1 s, esa distancia sube a 0,4–3 m.

Gráficos: `experiments/outputs/004/timeline_game1.png` (5 min del partido 1: etiqueta, balón visible y eventos) y `spell_durations.png`. Tablas: `per_game.csv`, `comparison.csv`.

## Hallazgos

1. **Puerta superada:** 3,8–4,1 posesiones/min, plausible y entre Sportec y los públicos. Los reinicios sin registrar son pocos (4–11 por partido) tras corregir dos detalles: orden SET PIECE/PASS en el mismo frame y RECOVERY del portero.
2. **Metrica Sample queda a medio camino** entre Sportec y los públicos en todas las métricas de duración. Al ser el mismo proveedor que el BAR‑ATH, la verdad del reto podría parecerse más a estas etiquetas que a las de Sportec. Esto refuerza que la bandera de Sportec es más reactiva (más posesiones cortas).
3. **En Metrica Sample el balón desaparece del tracking justo cuando está parado:** se ve el 92–94 % del tiempo con el balón en juego y solo el 0,2–16 % en D. La regla "sin balón = D" acierta el 89–96 % de los frames. Por tanto, **sus huecos de balón no se parecen a los nuestros** (los nuestros son fallos del tracker), lo que importa para M7 y para la degradación (S2.1).
4. **Vistazo al BAR‑ATH (a verificar en S1.1/S1.3):** en los 5 min públicos, con el balón en juego, el ATD lo ve el 93,5 % del tiempo, igual que en el Sample. En D lo sigue viendo el 66 %, así que "sin balón = D" acierta solo el 82 %. Si el 59 % de visibilidad del partido completo se reparte igual, los huecos se concentrarían en D y en los apagones, y no tanto en el juego. Merece confirmarlo pronto.
5. La derivación pierde algunos parones: hay de 9 a 12 tramos por partido de más de 5 s con el balón en juego según las etiquetas pero sin balón en el tracking (~4 min por partido). Probablemente sean faltas o balones fuera sin registrar, como en la línea de tiempo entre los segundos 432 y 463.

## Decisión

Seguir → S0.5 Evaluador. Se queda la derivación v1. Mejora barata pendiente para cuando se use Metrica Sample (M7): marcar D en los tramos largos sin balón dentro del juego, ya que en el Sample "sin balón" es casi siempre parado.
