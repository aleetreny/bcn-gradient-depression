# 036 · F3.11 TCN con normalización local (revisión 004, hallazgo 5)  (2026-10-24, commit 831986b)

Comando: `.venv311/bin/python scripts/s3_11_tcn_local.py`        Salida: `experiments/outputs/036/`        Código: `src/models/tcn.py` (`norm="channel"`)

## Qué se cambió

La TCN de la v2/v3 usa `GroupNorm`, que normaliza también **a lo largo del tiempo**:
- cada predicción depende de toda la parte;
- en entrenamiento las estadísticas salen de recortes de 60 s; en inferencia, de ~45 min con apagones incluidos.

La revisión lo demostró: cambiando frames lejanos, P(D) de un frame pasaba de 0,02 a 0,94.

Ahora la normalización es **por frame, sobre los canales** (LayerNorm). La red pasa a ser local de verdad (511 frames, ~20 s): un test comprueba que los frames lejanos no cambian la salida (`tests/test_tcn.py`). Todo lo demás es igual: features v1ctl, semillas y 25 épocas en MPS.

## Resultado

**LOMO exterior** (decodificador de cada fold ajustado para la TCN antigua: desventaja para la nueva):

| Fold | ens. cambio medio, antigua | **nueva** | ens. cambio M3, antigua | **nueva** |
|---|---|---|---|---|
| J03WMX | 0,660 | 0,661 | 0,646 | 0,655 |
| J03WN1 | **0,689** | 0,671 | **0,688** | 0,667 |
| J03WOH | 0,751 | **0,759** | 0,744 | **0,754** |
| J03WOY | **0,701** | 0,690 | **0,698** | 0,691 |
| J03WPY | 0,724 | **0,740** | 0,723 | **0,736** |
| J03WQQ | 0,743 | 0,742 | 0,738 | 0,738 |
| J03WR9 | 0,738 | **0,756** | 0,733 | **0,747** |
| **media** | 0,7152 | **0,7171** (4/7) | 0,7099 | **0,7126** (5/7) |

La TCN sola rinde igual por frame: AUC vivo 0,9974 frente a 0,9976, acierto de equipo 0,912 frente a 0,912.

**Transferencia** (modelos completos en MPS con los decodificadores finales de F3.8):

| | Sample (1 / 2 / 3) | Sample sin D ±3 s | públicos · F1 de D · posesiones (verdad 17) | BAR‑ATH pos./min · % D |
|---|---|---|---|---|
| antigua, ens. cambio medio | 0,803 (0,812 / 0,755 / 0,840) | 0,834 | 0,541 · 0,824 · 17 | 2,43 · 49,0 % |
| **nueva, ens. cambio medio** | **0,808** (0,818 / 0,767 / 0,839) | **0,845** (3/3) | 0,625 · 0,814 · 15 | 2,46 · 49,6 % |
| antigua, ens. cambio M3 | 0,809 | 0,840 | 0,570 · 0,818 · 16 | 2,50 · 49,0 % |
| **nueva, ens. cambio M3** | **0,810** | **0,845** | 0,588 · 0,812 · 15 | 2,50 · 49,6 % |

## Hallazgos

1. **La TCN local no pierde nada y gana un poco en todos los jueces:**
   - LOMO +0,002/+0,003, aun con el decodificador de la antigua;
   - Sample +0,005;
   - Sample sin D ±3 s +0,011 (3/3), que es la parte que mide la señal de equipo.

   Ninguna mejora llega al umbral de +0,01 en el LOMO.
2. **El problema de la revisión era real, pero no era la causa de la mala transferencia de la TCN.** La TCN sigue aportando poco en el Sample; la normalización global no lo explicaba.
3. Lo importante es la **robustez**: en el BAR‑ATH hay apagones largos, y con GroupNorm cambiaban las estadísticas de toda la parte. Con la versión local, una predicción solo depende de sus ±10 s. Los jueces de Sportec no pueden medir este riesgo.
4. Públicos +0,08: es ruido (17 posesiones) y ahora es conjunto de desarrollo. No cuenta.

## Decisión

**Seguir (adoptar):** `norm="channel"` pasa a ser la TCN de cualquier candidato futuro. Es más correcta, igual o mejor en todos los jueces y elimina una dependencia oculta. **No** se lanza un LOMO anidado completo solo para esto (~2 h con ganancia esperada de +0,002). Entrará en la próxima corrida anidada que lleve algo nuevo, y el decodificador se reajustará entonces.
