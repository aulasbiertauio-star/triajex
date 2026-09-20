"""
Base de conocimiento del módulo RAG.

AVISO IMPORTANTE — LEER ANTES DE PRESENTAR
-------------------------------------------
Estos textos son **notas internas escritas por el equipo**, resúmenes de
conocimiento de manual sobre escoliosis y elaboración de informes. NO son
citas literales de guías clínicas ni de artículos, y el campo `fuente` lo dice
explícitamente en cada entrada.

Se incluyen para que el módulo RAG tenga contenido real que recuperar en vez de
devolver una lista vacía, que es lo que hacía antes. Antes de darle a esto
cualquier uso que no sea una demo, hay que sustituirlas por referencias
verificables con su cita completa.

No inventes nombres de guías, autores, años ni DOIs para rellenar este archivo.
Un jurado médico lo detecta, y el argumento central del proyecto es
precisamente la honestidad sobre lo que el sistema sabe y lo que no.
"""

from __future__ import annotations

FUENTE = "Nota interna del equipo TriajeX — pendiente de referencia verificable"

DOCUMENTOS: list[dict[str, str]] = [
    {
        "hallazgo": "Escoliosis",
        "titulo": "Definición por ángulo de Cobb",
        "texto": (
            "Se considera escoliosis una desviación lateral del raquis con un "
            "ángulo de Cobb igual o mayor a 10 grados en el plano coronal. Por "
            "debajo de ese valor se habla de asimetría espinal, no de escoliosis. "
            "El ángulo se mide entre la placa terminal superior de la vértebra "
            "límite craneal y la placa terminal inferior de la vértebra límite "
            "caudal de la curva."
        ),
    },
    {
        "hallazgo": "Escoliosis",
        "titulo": "Variabilidad de la medición",
        "texto": (
            "La medición del ángulo de Cobb tiene variabilidad entre observadores "
            "y en un mismo observador que suele situarse en el orden de unos 5 "
            "grados. Por eso una diferencia menor a ese margen entre dos estudios "
            "no se interpreta por sí sola como progresión de la curva. Cualquier "
            "medición automática hereda esta limitación y debe presentarse con su "
            "margen, nunca como un valor exacto."
        ),
    },
    {
        "hallazgo": "Escoliosis",
        "titulo": "Elementos que debe contener el informe",
        "texto": (
            "Un informe de escoliosis describe: localización de la curva "
            "(torácica, toracolumbar o lumbar), nivel del ápex, dirección de la "
            "convexidad, magnitud en grados de Cobb, rotación vertebral, balance "
            "coronal y sagital, y estimación de la madurez esquelética. Una "
            "herramienta de preevaluación que solo indica presencia o ausencia no "
            "sustituye ninguno de estos elementos."
        ),
    },
    {
        "hallazgo": "Escoliosis",
        "titulo": "Proyección y protocolo de adquisición",
        "texto": (
            "El estudio de referencia para valorar una curva es la radiografía de "
            "columna completa en bipedestación. La proyección posteroanterior se "
            "prefiere sobre la anteroposterior porque reduce la dosis recibida por "
            "mama y tiroides. Mezclar proyecciones o encuadres distintos dentro de "
            "un mismo conjunto de datos introduce diferencias sistemáticas entre "
            "grupos que un modelo puede aprender en lugar de la anatomía."
        ),
    },
    {
        "hallazgo": "Escoliosis",
        "titulo": "Madurez esquelética",
        "texto": (
            "El riesgo de progresión de una curva depende de la madurez "
            "esquelética del paciente. En la radiografía se valora mediante el "
            "signo de Risser, que describe la osificación de la apófisis ilíaca, y "
            "el estado del cartílago trirradiado. Una misma magnitud de curva tiene "
            "implicaciones distintas en un paciente inmaduro y en uno maduro."
        ),
    },
    {
        "hallazgo": "Escoliosis",
        "titulo": "Orientación general sobre magnitud de la curva",
        "texto": (
            "De forma orientativa, las curvas por debajo de unos 25 grados en "
            "pacientes en crecimiento se siguen con controles periódicos; entre "
            "unos 25 y 45 grados se plantea tratamiento ortésico; y por encima de "
            "unos 45 a 50 grados se valora la opción quirúrgica. Son rangos "
            "generales: la decisión depende de la madurez, la progresión "
            "documentada y el criterio del especialista."
        ),
    },
    {
        "hallazgo": "Escoliosis",
        "titulo": "Signos de alarma de causa no idiopática",
        "texto": (
            "La escoliosis idiopática del adolescente es la forma más frecuente, "
            "con una prevalencia del orden del 2 al 3 por ciento en población "
            "adolescente. Deben hacer sospechar una causa secundaria: curva "
            "torácica de convexidad izquierda, progresión rápida, dolor "
            "persistente, alteraciones neurológicas y aparición en edades "
            "tempranas. En esos casos se valora completar el estudio con resonancia."
        ),
    },
    {
        "hallazgo": "Normal",
        "titulo": "Qué significa un resultado negativo en esta herramienta",
        "texto": (
            "Un resultado por debajo del umbral operativo indica que el modelo no "
            "encontró un patrón compatible con escoliosis, no que la radiografía "
            "sea normal. El modelo se entrenó únicamente con escoliosis en "
            "proyección anteroposterior y no evalúa ninguna otra patología: "
            "fracturas, espondilolistesis, lesiones líticas o alteraciones de "
            "partes blandas quedan fuera de su alcance por completo."
        ),
    },
    {
        "hallazgo": "Normal",
        "titulo": "Límite estadístico del modelo actual",
        "texto": (
            "Con 24 casos positivos en desarrollo, ningún umbral del modelo alcanza "
            "un recall de 0,90. En el punto de operación elegido, el recall es 0,75: "
            "uno de cada cuatro casos reales no se marca. Por eso la herramienta se "
            "presenta como apoyo a la lectura y nunca como filtro de descarte."
        ),
    },
    {
        "hallazgo": "Escoliosis",
        "titulo": "Cómo leer el mapa de atención",
        "texto": (
            "El mapa de atención señala las regiones de la imagen que más "
            "contribuyeron a la puntuación del modelo. No identifica vértebras ni "
            "mide ángulos. Si la activación se concentra fuera de la banda central "
            "donde discurre el raquis, conviene sospechar que el modelo está "
            "aprovechando diferencias de encuadre o de adquisición entre grupos y "
            "no la anatomía de la columna."
        ),
    },
]
