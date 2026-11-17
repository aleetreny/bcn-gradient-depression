# 039 · F4.2 ¿Entran los 5 minutos públicos en el entrenamiento final?  (2026-10-24, commit 68710a5)

Comando: `.venv311/bin/python scripts/s4_2_public_training.py`        Salida: `experiments/outputs/041/`

## Qué se mide

El GBM v1ctl completo (7 Sportec × 3 variantes de degradación, semilla 99), comparado con el mismo GBM más los 7500 frames etiquetados de los públicos, añadidos 1 o 10 veces (la repetición hace de peso).

Se juzga con los tres partidos de Metrica Sample, del mismo proveedor que el BAR‑ATH. La nota pública no sirve como juez: al entrenar con ella sería una nota sobre datos vistos.

## Resultado (spellF1, decodificador `decoder_v1ctl_gbm.json`)

| copias de los públicos | Sample (1 / 2 / 3) | media | sin D ±3 s |
|---|---|---|---|
| 0 | 0,812 / 0,749 / 0,837 | 0,7995 | 0,8414 |
| 1 | 0,821 / 0,750 / 0,840 | **0,8038** | 0,8426 |
| 10 | 0,815 / 0,748 / 0,829 | 0,7974 | 0,8373 |

## Hallazgos

1. Con una copia, el Sample sube +0,004 y la señal de equipo sin D +0,001. Es del orden de lo que cambia un GBM al añadir cualquier dato: no se distingue de ruido.
2. Con diez copias empeora (−0,002 y −0,004): 5 minutos con mucho peso sobreajustan.
3. Los públicos son el **único control con ATD real** que queda sobre el fichero final. Ya son conjunto de desarrollo, porque sirvieron para calibrar el simulador, pero aún sirven para detectar un fichero roto o raro.

## Decisión

**No entran en el entrenamiento final.** La ganancia no se distingue de ruido, y conservar el control vale más. El reentrenamiento final usa los 7 partidos de Sportec, igual que en el LOMO. Metrica Sample tampoco entra: M7 ya mostró que sus etiquetas derivadas no ayudan.
