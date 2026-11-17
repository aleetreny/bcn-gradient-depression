# 038 · F4.1 Robustez del decodificador (±20 %)  (2026-10-24)

Comando: `.venv311/bin/python scripts/s4_1_robustness.py --variant ens_mean --atd-inputs local_data/v3_inputs.npz --decoder-config configs/decoder_v1ctl_ens_mean.json` (y `--variant gbm`)        Salida: `experiments/outputs/040/`

## Qué se mide

Cada parámetro del decodificador se mueve un −20 %, −10 %, +10 % y +20 %, de uno en uno: λ_HA, λ_LD, λ_DL, d_min de equipo y de D, y β_HA.
- **LOMO anidado:** sobre los parámetros propios de cada fold (los ajustados en sus predicciones interiores), aplicados a la predicción exterior guardada.
- **BAR‑ATH:** sobre el decodificador final, aplicado a las probabilidades guardadas de la construcción v3. Se miran posesiones/min, % D y nota pública.

## Resultado

**v3 (ens. cambio medio):**

| parámetro | ΔLOMO (de −20 % a +20 %) | BAR‑ATH pos./min | públicos |
|---|---|---|---|
| base | 0,7152 | 2,39 | 0,570 |
| λ_HA | −0,0013 … +0,0001 | 2,37–2,42 | 0,570 |
| λ_LD, λ_DL | ≤ 0,0002 | 2,39 | 0,570 |
| d_min equipo | −0,0013 … +0,0009 | 2,39 | 0,570 |
| d_min D | ≤ 0,0001 | 2,37–2,41 | 0,570–0,588 |
| β_HA | −0,0014 … −0,0005 | 2,39–2,41 | 0,570–0,606 |

**Reserva simple (GBM con control):** base 0,6870; todas las variaciones entre −0,0008 y +0,0022.

## Hallazgos

1. **Los dos candidatos están en una meseta.** Ningún cambio del ±20 % mueve el LOMO más de 0,002; el fold peor tampoco cambia. La plateau de Optuna (mediana del 10 % mejor) hace su trabajo.
2. **Las posesiones/min del BAR‑ATH casi no dependen del decodificador.** Con λ_HA ±20 % pasan de 2,37 a 2,42. El cociente predichas/reales en Sportec va de 0,826 a 0,838. **La infra‑segmentación no es un problema de ajuste del decodificador:** los cambios que faltan no tienen evidencia en las probabilidades. Bajar la penalización no los saca.
3. La nota pública se mueve algo con d_min D y β_HA (0,57–0,61), pero son 1–2 posesiones de 17: ruido.

## Decisión

**Robustez superada** para la v3 y para la reserva. No hace falta regularizar más el decodificador. Se repetirá con la v4 cuando termine su anidado.
