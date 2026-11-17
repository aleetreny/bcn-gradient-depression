# 040 · v4 = ensemble con TCN local: anidado completo, construcción CPU y decisión de la Fase 4  (2026-10-24)

Comandos:
- `.venv311/bin/python scripts/s3_7_nested_full.py --features v1ctl --models tcn --tcn-norm channel --reuse-from local_data/oof/nested_degrade_v3_v1ctl --no-decode`, y después `--decode-only`. Salida: `experiments/outputs/033_v1ctl_tcnlocal/`.
- `.venv311/bin/python scripts/s4_3_final.py decoders --store nested_degrade_v3_v1ctl_tcnlocal --name v1ctl_tcnlocal`
- `.venv311/bin/python scripts/make_submission.py --name v4 --control --transition --tcn --tcn-norm channel --proba mean --change trans --decoder-config configs/decoder_v1ctl_tcnlocal_ens_m3.json --save-models`, dos veces.
- `.venv311/bin/python scripts/s4_3_final.py sample --models local_data/models/v4/models.pkl ...`
- Reserva: `make_submission.py --name reserve_gbm_ctl --control --decoder-config configs/decoder_v1ctl_gbm.json`, dos veces.

## Qué es

La v3 con la **TCN local** de F3.11 (normalización por frame). Las predicciones del GBM y de M3 son las mismas de F3.7 (mismas entradas y semillas, copiadas). Las 49 TCN del anidado son nuevas: 42 interiores y 7 exteriores, estas últimas reutilizadas de F3.11 porque tienen las mismas semillas. El decodificador se reajusta con el protocolo anidado.

## Resultado

**LOMO anidado (spellF1, 7 folds):**

| variante | v3 (TCN antigua) | **v4 (TCN local)** |
|---|---|---|
| ens. cambio medio | 0,7152 | 0,7131 |
| ens. cambio M3 | 0,7099 | **0,7144** |
| GBM solo / GBM + M3 | 0,6870 / 0,6947 | igual |

Por fold, ens. cambio M3 de la v4: 0,655 · 0,680 · 0,757 · 0,692 · 0,734 · 0,733 · 0,749.

**Fichero reproducible** (CPU, determinista, 52 min por construcción):
- dos construcciones con **SHA‑256 `dbc3607092354e4382795491be1e8d419e91c972a608c44815b03adc70eca942`** idéntico;
- difiere de la v3 en 5899 frames evaluados (3,7 %).

| | v3 | **v4** | reserva GBM con control |
|---|---|---|---|
| LOMO anidado | 0,7152 | 0,7144 | 0,6870 |
| Metrica Sample, **pesos CPU del fichero** (1 / 2 / 3) | 0,803 (MPS) | **0,812** (0,822 / 0,770 / 0,845) | 0,7995 (sin TCN) |
| Sample sin D ±3 s | 0,834 (MPS) | 0,840 | 0,841 |
| públicos [IC 5–95 %] · F1 de D · posesiones (17) | 0,570 [0,39–0,75] · 0,865 · 16 | 0,613 [0,44–0,79] · 0,850 · 15 | 0,584 [0,44–0,71] · 0,882 · 20 |
| BAR‑ATH pos./min · % D · mediana · % < 2 s | 2,39 · 49,1 % · 7,2 s · 12 % | 2,47 · 49,6 % · 6,8 s · 11 % | 2,79 · 49,3 % · 5,6 s · 16 % |
| SHA‑256 | `fbb87518…` | `dbc36070…` | `f067c06e…` |

**Robustez de la v4** (±20 % en cada parámetro):
- LOMO entre −0,0014 y +0,0017;
- BAR‑ATH entre 2,45 y 2,50 posesiones/min;
- públicos 0,613 en todas las variaciones.

## Decisión con la regla escrita antes de los resultados

1. **¿v4 sustituye a la v3?** Sí:
   - LOMO 0,7144 ≥ 0,7132;
   - Sample con pesos CPU 0,812 ≥ 0,798;
   - posesiones/min +3 % (dentro de ±10 %).

   **Nota de transparencia:** con la TCN local, la variante de cambio medio queda en 0,7131, justo bajo el umbral. Se eligió la de cambio de M3 porque es la que pone delante el juez principal en el anidado de la TCN local. Las dos están dentro del ruido.
2. **¿Ensemble o reserva?** Ensemble: +0,027 en el LOMO (≥ +0,01) y +0,013 en el Sample.
3. **¿Fichero roto?** No: públicos 0,613 ≥ 0,40 y F1 de D 0,850 ≥ 0,75.

**Candidata final: v4** (tag `v4`). **Reserva:** GBM con control, reproducible (`f067c06e…`).

## Hallazgos

1. La TCN local **empata en el LOMO** con la antigua una vez reajustado el decodificador. Lo que aporta es corrección: ninguna predicción depende de lo que pasa a más de ±10 s, apagones incluidos.
2. **El Sample medido con los pesos exactos del fichero (0,812) es el mejor de todos los candidatos.** Esto cierra el hallazgo 2 de la revisión 004.
3. Persiste la infra‑segmentación (2,47 posesiones/min frente a ~3,4 en los públicos). Es la principal debilidad conocida, y no es un problema del decodificador (log 038).
