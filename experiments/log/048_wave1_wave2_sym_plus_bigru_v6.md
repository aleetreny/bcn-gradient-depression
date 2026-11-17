# Plan Maestro y Resultados Finales — Laboratorio Paralelo (`exp_lab`)

> [!NOTE]
> **Aislamiento total garantizado (`0%` interferencia con Claude):** Todos los experimentos, cachés OOF, entrenamientos y archivos de *submission* se han ejecutado exclusivamente dentro de `<Ale's local workspace>`. El repositorio principal `<repo>/` permanece intacto.

---

## 1. Tabla Maestra de los 4 Jueces (Oleada 1 `v5` + Oleada 2 `v6`)

Cada candidato se evalúa bajo un protocolo estricto de **cero *overfitting*** (ningún modelo ni hiperparámetro se entrena ni ajusta sobre BAR-ATH ni sobre Metrica Sample):
1. **Juez 1A (Sportec `degrade_v3` 7-Fold Nested LOMO):** Validación cruzada anidada por partido completo (7 partidos DFL).
2. **Juez 1B (Sportec `degrade_v6` Ghost-Ball 7-Fold LOMO):** Robustez ante balones fantasma e incoherencia de interpolación.
3. **Juez 2 (Metrica Sample Data — 3 Partidos Completos):** Transferencia *zero-shot* al proveedor oficial del hackathon (con y sin margen `D ± 3s`).
4. **Juez 3 (BAR-ATH Plausibilidad Partido Completo + Gate Público 5 min):** Plausibilidad física en los `199,319` frames (`poss/min`, `%D`) y `spellF1` / `D_F1` en la ventana pública (`11,185..18,684`).

| ID / Candidato | Descripción Técnica | Juez 1A: LOMO `v3` (`std`) | Juez 1B: LOMO `v6` (Ghost) | Juez 2: Metrica Sample (`noD`) | Juez 3: BAR-ATH Pub (`D_F1` \| `pos/m` \| `%D`) | Submission Validada |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BASE (`v4_ens_mean`)** | Referencia oficial (`0.5*GBM + 0.5*TCN`, `dec_v4`, sin reparación) | `0.7131` (`0.0382`) | `0.7109` | `0.8110` (`0.8421`) | `0.6125` (`D=0.850` \| `2.46/m` \| `49.6%D`) | `submission_v4_ens_mean.csv` |
| **BASE + EXP-E2** | `v4_ens_mean` + Reparación de Apagones con Broadcast Top-K (`694` frames) | `0.7131` (`0.0382`) | `0.7109` | `0.8110` (`0.8421`) | `0.6125` (`D=0.867` \| `2.49/m` \| `50.1%D`) | — |
| **Rama `Bernardo` (`v2`)** | `HistGradientBoosting` (99 feats agregadas por equipo, sin TCN ni M3) + Viterbi 3 estados | `0.4910`–`0.5030` | — | `0.6059`–`0.6173` | No reportado (**~22 pp por debajo de `Ale`/`v6`**) | — |
| **EXP-A (`hyb_60_75`)** | Mezcla Híbrida Lineal + Log-Pool (`w_gbm=0.60, w_c=0.75`) + HSMM Ortogonalizado | `0.7175` (`0.0365`) | — | `0.8187` (`0.8454`) | `0.6125` (`D=0.850` \| `2.44/m` \| `49.5%D`) | — |
| **EXP-B (`v5_tta_m3`)** | Equivariancia Exacta `H ↔ A` + Rotación 180° en Inferencia (TTA sobre `v4`) | `0.7246` (`0.0338`) | — | `0.8243` (`0.8565`) | `0.5567` (`D=0.919` \| `2.29/m` \| `49.3%D`) | submission_v5_tta_m3.csv |
| **EXP-B+B2 (`v5_sym_linear`)** | `SymGBM_v1ctl + SymTCN + SymTrans` + `Linear(0.55, 0.50)` + `dec_v4` + Repair | `0.7378` (`0.0282`) | `0.7280` | `0.8214` (`0.8584`) | **`0.6345`** (`D=0.908` \| `2.37/m` \| `49.3%D`) | submission_v5_sym_linear.csv |
| **EXP-B+B2+A (`v5_sym_hybrid`)** | `SymGBM_v1ctl + SymTCN + SymTrans` + `Hybrid(0.55, 0.50)` + `dec_v4` + Repair | `0.7387` (`0.0282`) | `0.7314` | `0.8229` (`0.8584`) | `0.6207` (`D=0.902` \| `2.40/m` \| `49.3%D`) | submission_v5_sym_hybrid.csv |
| **EXP-F (`v5_sym_hybrid + dec_v5`)** | Re-optimización anidada del decodificador HSMM sobre probabilidades simétricas (`dec_v5_hyb`) | **`0.7410`** (`0.0311`) | `0.7354` | `0.8270` (`0.8590`) | `0.5935` (`D=0.902` \| `2.49/m` \| `49.4%D`) | — |
| **EXP-H (`SymBiGRU_TTA` solo)** | 3ª Familia: Red Recurrente Bidireccional Simétrica con `ChannelNorm` (`bigru.py`) sola | `0.7339` (`0.0262`) | — | `0.8229` (`0.8537`) | `0.6105` (en blend 3-familias) | — |
| **EXP-C+ (`v6_sym_plus_linear`)** | **`SymGBM_Plus` (352 feats: Física Nudos + Ghost + LSQ Bernardo) + `SymTCN` (`lin 0.55, dec_v4`)** | `0.7339` (`0.0316`) | `0.7280` | `0.8268` (`0.8614`) | **`0.6345` (`D=0.916` \| `2.34/m` \| `49.3%D`)** | submission_v6_sym_plus_linear.csv |
| **EXP-C++F (`v6_sym_plus_hybrid`)** | **`0.5*SymGBM_v1ctl + 0.5*SymGBM_Plus + SymTCN` (`hyb 0.55, dec_v5_hyb`)** | **`0.7393`** (`0.0325`) | `0.7354` | **`0.8306` (`0.8643`)** | **`0.6207` (`D=0.903` \| `2.50/m` \| `49.4%D`)** | submission_v6_sym_plus_hybrid.csv |
| **EXP-C++F (`v6_pure_plus_hybrid`)** | **`SymGBM_Plus + SymTCN` (`hyb 0.50, dec_v5_hyb`) — Récord en Juez 1B y Juez 2** | **`0.7385`** (`0.0320`) | **`0.7374`** | **`0.8343` (`0.8690`)** | **`0.6207` (`D=0.903` \| `2.46/m` \| `49.2%D`)** | submission_v6_pure_plus_hybrid.csv |
| **EXP-V6 (`v6_moe_super_champion`)** | **MoE Físico (`v1ctl+Plus` + Compuerta Ghost `SymGBM_v6` + Compuerta Gap `SymBiGRU`)** | `0.7353` (`0.0295`) | `0.7287` | `0.8240` (`0.8601`) | **`0.6207` (`D=0.915` \| `2.37/m` \| `49.3%D`)** | submission_v6_moe_super_champion.csv |
| **EXP-V6 (`v6_3family_seq`)** | **3 Familias (`SymGBM_v1ctl/Plus` + `SymGBM_v6` + `0.85*SymTCN + 0.15*SymBiGRU_HA`)** | `0.7357` (**`0.0279`**) | `0.7291` | `0.8267` (`0.8619`) | **`0.6345` (`D=0.903` \| `2.35/m` \| `49.3%D`)** | submission_v6_3family_seq.csv |

---

## 2. Descubrimientos Clave de la Oleada 2 (`EXP-C+`, `EXP-F`, `EXP-G`, `EXP-H`)

### A. `EXP-C+` (v1ctl_plus.py): Física Anclada en Nudos + Incoherencia Ghost + cinemática LSQ de Bernardo (352 features)
- Añadimos **108 nuevas variables estrictamente equivariantes** (`error de simetría = 0.000e+00`) sobre `v1ctl` (244 vars):
  1. **Geometría del balón anclada en nudos reales (Anti-Ghost-Ball):** Cuando el balón está ausente (`ball_gap > 0`), la interpolación lineal cruza el campo en línea recta generando poseedores fantasma. Anclamos la posición en el último nudo observado ($t_{\text{prev}}$) y el próximo nudo ($t_{\text{next}}$), calculando el ratio de transporte físico $v_{\text{implied}} = \|\mathbf{b}(t_{\text{next}}) - \mathbf{b}(t_{\text{prev}})\| / \Delta t$.
  2. **Detectores explícitos de incoherencia física (`ghost_incoh`):** Producto de la velocidad del balón interpolado por la distancia al jugador más cercano cuando nadie está a $<3\text{m}$.
  3. **Integración de la mejor idea de Bernardo (`lsq_vel`):** Regresión lineal por mínimos cuadrados en ventanas multi-escala ($W \in \{5, 13, 25\}$ frames) que se detiene en cortes de tracking ($>2.5\text{m}$/frame), proyectada en ventaja de llegada al balón (`time_to_ball`) y momento del equipo.
  4. **Continuidad a través de apagones (`gap_edge_continuity`):** Similitud coseno entre el estado antes y después de un corte de cámara.
- **Impacto demostrado:**
  - `SymGBM_Plus_TTA` solo sube a **`0.7042` LOMO** (+0.43 pp sobre `SymGBM_v1ctl_TTA` `0.6999`, y **+1.72 pp** sobre el `GBM` original de `v4` `0.6870`).
  - Al combinarlo con `SymTCN_TTA`:
    - Con mezcla lineal (`v6_sym_plus_linear`) alcanza **`0.6345` spellF1 (`D_F1 = 0.916`)** en BAR-ATH y **`0.8268` (`0.8614` noD)** en Metrica Sample.
    - Con mezcla híbrida (`v6_pure_plus_hybrid`) alcanza **`0.7385` en Juez 1A**, **`0.7374` en Juez 1B (+2.65 pp)** y **`0.8343` (`0.8690` noD, +2.33 pp sobre `v4`) en Juez 2**, manteniendo **`0.6207` (`D_F1 = 0.903`)** en Juez 3.

### B. `EXP-F` (exp_f_sym_decoder_and_boundary.py): Re-optimización Anidada del Decodificador HSMM (`dec_v5_hyb`)
- El decodificador `dec_v4` había sido ajustado sobre las probabilidades asimétricas y menos calibradas de `v4`. Re-optimizando `DecoderParams` en protocolo LOMO anidado (6 folds internos por cada fold externo) sobre el ensemble simétrico híbrido:
  - **Juez 1A sube a `0.7410` (+2.79 pp sobre `v4_ens_mean`)**, ganando en **7 de 7 folds**.
  - **Juez 1B sube a `0.7354` (+2.45 pp sobre `v4_ens_mean`)**.
  - **Juez 2 sube a `0.8270` (`0.8590` noD, +1.60 pp sobre `v4_ens_mean`)**.

### C. `EXP-G` (exp_g_ghost_sym.py): Entrenamiento Multi-Régimen `degrade_v6` (Ghost-Ball Incoherence)
- Entrenamos `SymGBM_v6` y `SymTrans_v6` bajo el régimen de degradación `degrade_v6` (donde los huecos de balón se rellenan con trayectorias fantasma incoherentes).
- **Prueba empírica de que BAR-ATH sufre de balones fantasma:** En BAR-ATH (Juez 3), el modelo de árboles `SymGBM_v6_TTA` **por sí solo (sin TCN)** salta de `0.5576` (`D_F1 = 0.866`) a **`0.6207` (`D_F1 = 0.916`, +6.31 pp)** frente a `SymGBM_v3_TTA`. En Juez 1B (`degrade_v6`), mejora +1.04 pp por sí solo y +1.46 pp al mezclarse 50/50 con `SymGBM_v3_TTA`.

### D. `EXP-H` (bigru.py & exp_h_bigru_sym.py): 3ª Familia de Modelos (`SymBiGRU_TTA`) y Desacoplamiento `D` vs `H/A`
- Construimos una 3ª familia arquitectónica genuina: un **BiGRU de 2 capas con `ChannelNorm` por frame** (invariante a derivas de distribución de hasta 45 minutos) y **atención multi-escala local** ($k \in \{3, 9, 25\}$), entrenado con aumentación simétrica `H ↔ A`.
- **Resultado individual:** `SymBiGRU_TTA` **por sí solo** alcanza **`0.7339` (`std = 0.0262`)** en Juez 1A (superando a `SymTCN_TTA` solo `0.7309`) y **`0.8229` (`0.8537` noD)** en Juez 2.
- **Descubrimiento de desacoplamiento (`moe_combine`):** En secuencias continuas de 45 minutos (68,000 frames en BAR-ATH), el estado oculto recurrente del BiGRU propaga contexto `H/A` a través de pausas largas, reduciendo la sensibilidad bruta a `D`, pero es extraordinariamente preciso distinguiendo **quién tiene el balón (`H` vs `A`)** durante el juego vivo y durante huecos de balón. Anclando `p_D` a `SymGBM + SymTCN` e inyectando `SymBiGRU_TTA` en la dirección condicional `p_H / (p_H + p_A)`, el modelo de 3 familias (`v6_3family_seq`) logra la **menor varianza entre partidos de todo el estudio (`std = 0.0279`, -27% vs `v4`)**, **`0.7357` en Juez 1A**, **`0.8267` (`0.8619` noD) en Juez 2** y **`0.6345` (`D_F1 = 0.903`) en Juez 3**.

---

## 3. Guía de Selección de Submissions Finales

Todas las *submissions* han superado el validador oficial (`possession_metrics.validate_submission(table, 199319)`): `199,319` filas, columnas `frame,label`, `100% DEAD` en descanso y frames no evaluados, y tiempos muertos físicamente plausibles (`49.2%–49.4% D`, `2.34–2.50` posesiones/min):

| Prioridad / Perfil | Archivo CSV | Juez 1A (`v3`) | Juez 1B (`v6`) | Juez 2 (Sample / `noD`) | Juez 3 (BAR-ATH Pub / `D_F1`) | Cuándo elegirla |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Opción 1 (Equilibrio Total 4 Jueces — Recomendada #1)** | submission_v6_sym_plus_hybrid.csv | **`0.7393`** (+2.62 pp) | **`0.7354`** (+2.45 pp) | **`0.8306` (`0.8643`)** | **`0.6207` (`D=0.903`)** | **Supera a `v4` simultáneamente en los 4 Jueces.** Combina `v1ctl` + `v1ctl_plus` (352 feats) + `SymTCN` + mezcla híbrida + `dec_v5_hyb`. |
| **Opción 2 (Máximo Transfer Metrica + Ghost-Ball)** | submission_v6_pure_plus_hybrid.csv | **`0.7385`** (+2.54 pp) | **`0.7374`** (+2.65 pp) | **`0.8343` (`0.8690`)** | **`0.6207` (`D=0.903`)** | **Récord en Juez 1B (`0.7374`) y Juez 2 (`0.8343` / `0.8690` noD)** y gana en los 4 Jueces a la vez. Ideal si confiamos al 100% en las nuevas variables físicas `v1ctl_plus`. |
| **Opción 3 (Máximo Score en BAR-ATH + 3 Familias)** | submission_v6_3family_seq.csv | **`0.7357`** (`std 0.0279`) | **`0.7291`** (+1.82 pp) | **`0.8267` (`0.8619`)** | **`0.6345` (`D=0.903`)** | **Une las 3 familias (`SymGBM` + `SymTCN` + `SymBiGRU`) + `SymGBM_v6`**, logra la menor varianza entre partidos (`0.0279`) y el máximo `spellF1` en BAR-ATH (`0.6345`). |
| **Opción 4 (Máximo `spellF1` y `D_F1` en BAR-ATH)** | submission_v6_sym_plus_linear.csv | **`0.7339`** (+2.08 pp) | **`0.7280`** (+1.71 pp) | **`0.8268` (`0.8614`)** | **`0.6345` (`D=0.916`)** | Versión lineal conservadora con `dec_v4` y features `v1ctl_plus`: empata el récord de `0.6345` en BAR-ATH con **`D_F1 = 0.916`** (+6.6 pp sobre `v4`). |
| **Opción 5 (MoE Físico Adaptativo)** | submission_v6_moe_super_champion.csv | **`0.7353`** (`std 0.0295`) | **`0.7287`** (+1.78 pp) | **`0.8240` (`0.8601`)** | **`0.6207` (`D=0.915`)** | Mixture-of-Experts gobernado frame a frame por las compuertas físicas de incoherencia de balón (`w_ghost`) y longitud de hueco (`w_gap`). |
