# 028 · F3.2 M9: TCN por frame — cribado  (2026-10-24, commit f9fe85c)

Comando: `.venv311/bin/python scripts/s3_2_tcn_screen.py`        Config: `configs/degrade_v3.json`        Semillas: TCN = índice del fold
Código: `src/models/tcn.py`

## Qué es

Red convolucional temporal no causal: 8 bloques residuales con dilataciones de 1 a 128, que ven unos 20 s de contexto, con 128 canales. Usa las 220 features normalizadas (mediana/IQR) e indicadores de ausencia. **Dos salidas por frame:** H/A/D y cambio de equipo a ±6 frames (multitarea). Se entrena con recortes de 60 s durante 25 épocas en MPS, unos 3 min por fold.

**Cribado, no veredicto:** solo LOMO exterior (una TCN por partido de prueba, entrenada con los otros 6). Todo se decodifica con los parámetros de la v1.1 de cada fold, que se ajustaron para el GBM: una desventaja para la TCN.

## Resultado

| | GBM (v1.1) | TCN sola | **media GBM + TCN** |
|---|---|---|---|
| spellF1 con el decodificador de la v1.1 | 0,668 | 0,661 (3/7 > GBM) | **0,695 (6/7 > GBM; +0,027)** |
| spellF1 con suavizado w 12 | 0,630 | **0,651** | 0,668 |
| AUC vivo/parado | 0,9968 | 0,9977 | 0,9979 |
| acierto de equipo por frame | 0,908 | 0,910 | **0,915** |
| log‑loss | 0,196 | 0,321 | **0,174** |

Media − GBM por fold (decodificado): +0,025 · +0,030 · **+0,066** · +0,020 · −0,002 · +0,007 · +0,041. La cabeza de cambio de la TCN tiene AP 0,566 (la de M3, 0,55).

## Hallazgos

1. **La TCN sola no supera al GBM con el decodificador** (0,661 frente a 0,668), aunque sí con suavizado (0,651 frente a 0,630). Está peor calibrada (log‑loss 0,32) y los parámetros del decodificador no son los suyos.
2. **La media de los dos mejora mucho** (+0,027, 6/7) incluso con esa desventaja: son modelos distintos que se equivocan en sitios distintos. Es lo que el plan esperaba del ensemble (en Kaggle DFL, pasar de 1 a 6 modelos subió de 0,768 a 0,827).
3. Su cabeza de cambio es tan buena como la de M3, otra señal para el decodificador.

## Decisión

**Seguir con evaluación anidada completa** del ensemble (F3.3): TCN interiores sin el partido exterior, el decodificador ajustado sobre ellas y dos señales de cambio (solo M3; media de M3 y la cabeza de la TCN).

*Nota aparte (bordes de D en públicos, v1.2):* el D empieza 2,5–3 s tarde en 2 de 4 parones y una reanudación se predice 6 s antes (el caso #20 de S1.1). Con 4 casos no se calibra ningún desplazamiento: anotado como hipótesis para una mejora de D.
