# Investigación de métodos — rama `Ale`

Qué han hecho otros (papers, datasets y retos parecidos) comprobado en las fuentes originales, cómo son nuestros datos y el plan de implementación y pruebas. Cada documento tiene un **Resumen** en lenguaje sencillo y un **Desarrollo** técnico pensado para que lo use un agente.

## Lo importante (2 minutos)

1. **Podemos entrenar con datos de verdad.** Sportec/DFL publica 7 partidos completos a 25 fps con *vivo/parado* y *equipo poseedor* en cada frame, es decir, nuestras etiquetas H/A/D. Pasamos de 17 posesiones etiquetadas a unas 2.000 y, sobre todo, a una **validación fiable** dejando un partido fuera cada vez.
2. **Hay que "estropear" esos datos** para que se parezcan a los nuestros: balón que desaparece el 41 % del tiempo, jugadores sin identidad. Si no, lo aprendido no se transfiere (PathCRF pierde 25 puntos al cambiar de proveedor).
3. **Lo que más puntúa es el decodificador temporal**, no la red: la pieza que decide cuándo cambia de verdad la posesión. Se ajusta **directamente con spellF1** en los partidos externos.
4. **El error más caro es llegar tarde a un cambio de equipo** (1 s de retraso cuesta un 22–35 %). Equivocarse 1–2 s en el inicio de un balón parado casi no cuesta.
5. **Perder un robo corto cuesta triple** (une a sus vecinos). La verdad tiene posesiones de 0,76 s y 1,3 s, así que suavizar mucho es malo. Truco a validar: ante la duda, un **D corto** separa posesiones y apenas penaliza.
6. **Primero árboles + decodificador; luego, si el diagnóstico lo pide, apuestas de techo alto:** rellenar el balón (Ball Radar o detector sobre el vídeo, que ya lo intenta otro equipo), PathCRF como señal y fusión con la retransmisión.
7. **Lectura correcta de los papers:** PathCRF acierta el *equipo* el 92 % del tiempo (su 70 % famoso mide otra cosa). Vemos el balón más que en los escenarios de Ball Radar. Y las redes (TCN) no deberían ser el primer sistema.

## Documentos

| Doc | Para qué |
|---|---|
| [01_que_han_conseguido_otros_trabajos_y_retos.md](01_que_han_conseguido_otros_trabajos_y_retos.md) | Resultados de papers, datasets y retos parecidos, con cada cifra comprobada en su fuente |
| [02_datos_publicos_para_entrenar.md](02_datos_publicos_para_entrenar.md) | Sportec, Metrica Sample, SkillCorner: cómo cargarlos, qué comprobar y cómo degradarlos |
| [03_que_aprendemos_de_cada_paper.md](03_que_aprendemos_de_cada_paper.md) | PathCRF, Ball Radar, Lang, TCN, CropDrop, Vidal‑Codina: qué nos llevamos de cada uno |
| [04_como_puntua_spellf1_y_otros_hallazgos.md](04_como_puntua_spellf1_y_otros_hallazgos.md) | Lo que no estaba en el informe: semántica de la verdad, sensibilidad de spellF1, competencia, fuentes extra de balón |
| [05_plan_para_ganar.md](05_plan_para_ganar.md) | **El plan de implementación y pruebas (v2):** pasos pequeños con output a revisar y decisión, tres jueces de validación, matriz que pone a prueba los métodos de otros trabajos y arquitectura de referencia |
| [06_como_son_nuestros_datos.md](06_como_son_nuestros_datos.md) | Formato real de nuestros ficheros (solo estructura, sin valores), trampas (`speed_kph` que en realidad es m/s, 46 slots, tercer canal del balón que no es velocidad) y cómo cargarlos |

Scripts asociados:

- [`scripts/spellf1_error_costs.py`](../../scripts/spellf1_error_costs.py): cuánto cuesta cada tipo de error en spellF1. Solo usa la verdad pública.
- [`scripts/download_public_datasets.sh`](../../scripts/download_public_datasets.sh): descarga Sportec, Metrica Sample, SkillCorner, PathCRF y Ball Radar en `data_external/` (ignorada por git).
- [`scripts/inspect_data_structure.py`](../../scripts/inspect_data_structure.py): describe la forma de nuestro bundle sin imprimir ningún valor.

## Estado

Investigación documental. **Aún no se ha ejecutado ningún experimento con Sportec ni con el bundle BAR‑ATH.** Todas las mejoras son hipótesis hasta que pasen la evaluación "un partido fuera" descrita en 05.
