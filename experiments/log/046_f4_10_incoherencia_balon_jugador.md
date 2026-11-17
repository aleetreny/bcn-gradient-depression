# 046 · F4.10 Simulador v6: incoherencia balón–jugador ("fantasmas")  (2026-10-24, commit 41d6279)

**Escrito antes de ver los resultados**, con los criterios fijados de antemano.

## Motivo

Los logs 042–045 muestran que lo que distingue al ATD del BAR‑ATH es la **incoherencia entre balón y jugadores**: golpeos sin nadie a < 4 m en el 25–28 % de los casos (5 % en el simulador, 1 % en Metrica Sample). No es la forma recta del balón (F4.5) ni algo que un detector genérico de vídeo arregle (F4.9).

## Qué se cambia

Componente `ghosts` en `src/data/degrade.py`, con su propio flujo aleatorio, de modo que las configs anteriores no cambian:
- **ocultación junto al balón:** episodios de 0,5–3 s en los que desaparece el jugador más cercano al balón (si está a < 3 m);
- **balón desplazado:** episodios de 0,4–2 s en los que el balón se desvía 2–6 m, entrando y saliendo suavemente.

Las dos tasas se calibran **sin etiquetas** (`scripts/s4_10_ghost_calibration.py`), igualando en los frames de juego según el indicador de parón las dos estadísticas del ATD completo:
- balón lento sin nadie a < 3 m = 0,243;
- golpeo sin nadie a < 4 m = 0,283.

Resultado → `configs/degrade_v6.json`.

## Criterios (fijados antes)

Cribado con el GBM v1ctl (`scripts/s4_11_ghost_screen.py`, decodificador de cada fold sin reajustar):
1. Metrica Sample del GBM entrenado con v6 ≥ el actual − 0,003.
2. LOMO con el partido de prueba degradado como siempre (v3): v6 ≥ v3 − 0,005, es decir, no estropea el dominio normal.
3. LOMO con el partido de prueba con fantasmas (v6): entrenado con v6 ≥ entrenado con v3 + 0,01, es decir, gana robustez donde el BAR‑ATH se parece.

Solo si se cumplen los tres, `scripts/night_v6.sh` lanza la corrida completa:
- anidado con GBM + M3 + TCN local;
- decodificadores finales;
- fichero reproducible v6;
- Sample con sus pesos CPU.

La v6 sustituiría a la v4 con la regla de la Fase 4: LOMO anidado (con prueba v3) ≥ v4 − 0,002 y Sample con pesos CPU ≥ 0,812 − 0,005.

## Resultado

*(pendiente: corrida nocturna)*
