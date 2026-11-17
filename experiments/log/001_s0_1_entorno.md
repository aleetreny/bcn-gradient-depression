# 001 · S0.1 Entorno  (2026-10-21, commit 65549c9)

Comando: `bash scripts/setup_research_env.sh` y `.venv311/bin/python scripts/check_official.py`
Config: `requirements-research.txt` (+ `requirements-research-lock.txt`)        Semilla: n/a

## Resultado

Entorno `.venv311` (CPython 3.11.16, macOS arm64, M5 Pro, 18 núcleos, 48 GB), reconstruido desde cero con el script.

| Paquete | Versión | Prueba de humo |
|---|---|---|
| numpy | 1.26.4 | = `requirements-lock.txt` del evaluador |
| pandas | 2.2.3 | = `requirements-lock.txt` del evaluador |
| pyarrow | 25.0.0 | parquet ida y vuelta: ok |
| scipy | 1.17.1 | optimize: ok |
| scikit-learn | 1.9.1 | logística + isotónica: ok |
| lightgbm | 4.7.0 | entrenamiento con OpenMP: ok |
| optuna | 5.0.0 | TPE: ok |
| kloppy | 3.18.0 | lectores sportec/metrica: ok (misma versión que `starter/`) |
| matplotlib | 3.11.2 | savefig Agg: ok |
| tqdm, pyyaml, pytest | 4.70.1, 6.0.3, 9.1.1 | import |
| torch | no instalado | se añade en el primer paso con redes |

`check_official.py`: exit 0. Los 4 baselines validan (`OK`) y el scoreboard es **byte a byte idéntico** al generado antes con Python 3.9:

| Baseline | macroF1 | spellF1 | posesiones (verdad 17) |
|---|---|---|---|
| smooth125 | 0,5812 | 0,4242 | 16 |
| smooth50 | 0,6207 | **0,3256** | 26 |
| smooth25 | 0,6200 | 0,2868 | 36 |
| raw | 0,5972 | 0,0378 | 131 |

Logs: `experiments/outputs/001/check_env.log`, `experiments/outputs/001/check_official.log`.

## Hallazgos

- **pyarrow 26 necesita NumPy 2 en tiempo de ejecución** y no lo declara en sus metadatos, así que el resolvedor lo elige igualmente. Fijado a 25.0.0 para mantener NumPy 1.26.4 (el del evaluador congelado).
- **LightGBM en macOS necesita `libomp`** y no hay Homebrew en la máquina. El script hace que LightGBM use la `libomp.dylib` que ya trae scikit-learn (rpath + firma ad hoc). Ventaja: un solo runtime OpenMP por proceso (comprobado con threadpoolctl), lo que evita los bloqueos de tener dos copias. Si se instala Homebrew + `brew install libomp`, el script se salta el parche.
- El evaluador da el mismo resultado en 3.9 y 3.11: el entorno de investigación no altera las puntuaciones.

## Decisión

Seguir → S0.2 Lector BAR‑ATH (`src/data/metrica_atd.py`).
