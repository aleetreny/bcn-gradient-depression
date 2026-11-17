# 023 · S2.4b LOMO anidado completo (revisión 003, punto 1)  (2026-10-23, commit 3c04e78)

Comando: `.venv311/bin/python scripts/s2_4b_nested_lomo.py --config configs/degrade_v3.json --trials 120`        Config: `configs/degrade_v3.json`        Semillas: GBM interior 100 + fold, exterior = fold; Optuna = fold

## Protocolo

Para cada partido exterior T:
1. **Probabilidades interiores:** para cada uno de los otros 6 partidos J, un GBM de dos cabezas entrenado y calibrado con los 5 restantes (**ni T ni J**) predice J.
2. **Decodificador:** Optuna (120 pruebas) sobre esas 6 predicciones interiores; centro de la meseta.
3. **Exterior:** un GBM entrenado con los 6 partidos predice T, se decodifica con los parámetros del paso 2 y **T se puntúa una sola vez**.

En total, 49 entrenamientos del GBM. Las predicciones por fold están en `local_data/oof/nested_degrade_v3/`.

## Resultado

| Fold | **LOMO anidado** | macroF1 | F1 vivo/parado oficial | F1 de D (precisión / recall) | posesiones pred. / verdad | suavizado w 12 | protocolo S2.4 (no anidado) | λ_HA | λ_LD | λ_DL | d_min equipo / D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | **0,615** | 0,909 | 0,969 | 0,957 (0,941 / 0,974) | 355 / 511 | 0,583 | 0,614 | 2,9 | 26,8 | 30,1 | 6 / 16 |
| J03WN1 | **0,631** | 0,896 | 0,966 | 0,970 (0,952 / 0,988) | 364 / 500 | 0,573 | 0,609 | 4,6 | 30,3 | 35,0 | 16 / 33 |
| J03WOH | **0,699** | 0,924 | 0,974 | 0,972 (0,969 / 0,975) | 308 / 402 | 0,677 | 0,692 | 6,5 | 34,9 | 1,7 | 4 / 79 |
| J03WOY | **0,647** | 0,899 | 0,969 | 0,964 (0,967 / 0,961) | 373 / 456 | 0,613 | 0,646 | 4,1 | 35,3 | 38,6 | 8 / 9 |
| J03WPY | **0,701** | 0,929 | 0,984 | 0,981 (0,977 / 0,985) | 343 / 421 | 0,660 | 0,702 | 4,1 | 32,8 | 35,6 | 4 / 25 |
| J03WQQ | **0,704** | 0,929 | 0,985 | 0,978 (0,975 / 0,981) | 365 / 401 | 0,663 | 0,703 | 3,4 | 28,0 | 30,7 | 7 / 76 |
| J03WR9 | **0,680** | 0,922 | 0,971 | 0,962 (0,934 / 0,992) | 318 / 370 | 0,640 | 0,684 | 4,1 | 0,9 | 29,6 | 10 / 9 |
| **media** | **0,668 ± 0,034** | 0,915 | 0,974 | **0,969** (0,959 / 0,979) | | 0,630 | 0,664 | | | | |

**7/7 folds** mejoran respecto al suavizado.

## Hallazgos

1. **Nuevo juez: LOMO anidado con `degrade_v3` = 0,668 ± 0,034.**
2. **La fuga del anidado era despreciable:** con las mismas probabilidades exteriores, el protocolo no anidado de S2.4 da 0,664 (el anidado incluso da algo más, por el ruido de Optuna). *(Corregido tras la revisión 004:* esto se midió con la degradación nueva. Con estas semillas de Optuna, corregir el anidado no redujo el score, pero **no cuantifica retrospectivamente el sesgo del 0,666 de la v1**, que usaba `degrade_v2`.)
3. **D sigue siendo más fácil en Sportec que en el BAR‑ATH:** F1 de D 0,969 en el LOMO frente a 0,847 de la v1 en los públicos. Allí falla sobre todo el **recall** (0,75 frente a 0,98): se predice D de menos, previsiblemente en las fronteras (S1.5: Metrica parece reanudar después de que el balón se mueva). **D sigue abierto** para la Fase 3.
4. Con la degradación v3 los parámetros del decodificador cambian: λ_HA de 3 a 6,5 (antes ~2,2) y d_min de D a menudo corto (9–79 frames; antes ~106). Es lógico: los D simulados son menos "fáciles" y el decodificador confía más en el modelo.
5. Sigue faltando detectar cambios: en Sportec se predicen 308–373 posesiones frente a 370–511 reales.

## Decisión

**El LOMO anidado con `degrade_v3` pasa a ser el juez principal del tablero.** Siguiente: construir **v1.1** (el mismo sistema con `degrade_v3`) sobre el BAR‑ATH y repetir S2.5/S2.6 con las predicciones por fold.
