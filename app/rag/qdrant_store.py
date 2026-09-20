"""
Módulo RAG: índice vectorial en Qdrant sobre la base de conocimiento propia.

POR QUÉ CAMBIÓ ESTE ARCHIVO
---------------------------
La versión anterior creaba `QdrantClient(path=":memory:")`, que arranca con un
almacén **vacío** en cada reinicio. `consultar()` creaba la colección y
devolvía lista vacía, así que el pre-reporte imprimía siempre "Base de
conocimiento RAG en modo autónomo" y nunca recuperaba nada. A cambio, el
arranque pagaba la descarga de `all-MiniLM-L6-v2` y la carga de torch.

Ahora:
  - Se indexan los documentos de `conocimiento.py` al arrancar. Hay contenido
    real que recuperar.
  - Los vectores son TF-IDF calculados con scikit-learn sobre el propio corpus.
    Con una decena de documentos cortos en español, TF-IDF recupera igual de
    bien que un embedding neuronal, es instantáneo, no descarga nada y no
    necesita torch. La búsqueda por similitud del coseno la sigue haciendo
    Qdrant.
  - Sin `sentence-transformers` ni torch: el arranque en frío pasa de decenas
    de segundos a inmediato, que es lo que hace falta en una demo en vivo.

Honestidad sobre lo que es: esto es recuperación léxica, no semántica. Si
alguien pregunta, se dice. `describir_metodo()` devuelve esa frase lista para
mostrar en la interfaz.
"""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, FieldCondition, Filter,
                                       MatchValue, PointStruct, VectorParams)
from sklearn.feature_extraction.text import TfidfVectorizer

from .conocimiento import DOCUMENTOS, FUENTE


@dataclass(frozen=True)
class DocumentoRecuperado:
    """Fragmento recuperado del índice."""
    texto: str
    fuente: str
    score: float
    hallazgo: str
    titulo: str = ""


class AlmacenRAG:
    """Índice en memoria. Se reconstruye en cada arranque; tarda milisegundos."""

    def __init__(self, coleccion: str = "criterios_escoliosis",
                 documentos: list[dict[str, str]] | None = None) -> None:
        self.coleccion = coleccion
        self.documentos = documentos if documentos is not None else DOCUMENTOS

        if not self.documentos:
            raise ValueError("La base de conocimiento está vacía.")

        corpus = [f"{d['titulo']}. {d['texto']}" for d in self.documentos]
        self.vectorizador = TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        )
        matriz = self.vectorizador.fit_transform(corpus).toarray().astype(np.float32)
        self.dim = int(matriz.shape[1])

        try:
            self.cliente = QdrantClient(location=":memory:")
        except Exception as exc:
            raise ConnectionError(f"No se pudo iniciar Qdrant en memoria: {exc}") from exc

        self.cliente.recreate_collection(
            collection_name=self.coleccion,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        self.cliente.upsert(
            collection_name=self.coleccion,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{coleccion}/{i}")),
                    vector=matriz[i].tolist(),
                    payload={
                        "texto": doc["texto"],
                        "titulo": doc["titulo"],
                        "hallazgo": doc["hallazgo"],
                        "fuente": FUENTE,
                    },
                )
                for i, doc in enumerate(self.documentos)
            ],
        )

    # ------------------------------------------------------------------
    @property
    def n_documentos(self) -> int:
        return len(self.documentos)

    @staticmethod
    def describir_metodo() -> str:
        return ("Recuperación léxica (TF-IDF + similitud del coseno en Qdrant) "
                "sobre una base propia de notas del equipo. No es búsqueda "
                "semántica ni literatura citada.")

    def _vector(self, texto: str) -> list[float]:
        return self.vectorizador.transform([texto]).toarray()[0].astype(np.float32).tolist()

    def consultar(self, hallazgo: str, top_k: int = 3) -> list[DocumentoRecuperado]:
        """Recupera los fragmentos más cercanos a la consulta, filtrados por hallazgo."""
        consulta = (
            f"Radiografía de columna con hallazgo de {hallazgo}. "
            "Criterios de informe, medición del ángulo de Cobb y protocolo de triaje."
        )
        try:
            respuesta = self.cliente.query_points(
                collection_name=self.coleccion,
                query=self._vector(consulta),
                query_filter=Filter(must=[FieldCondition(
                    key="hallazgo", match=MatchValue(value=hallazgo))]),
                limit=top_k,
                with_payload=True,
            )
            puntos = respuesta.points
        except Exception as exc:
            warnings.warn(f"Consulta RAG fallida: {exc}")
            return []

        return [
            DocumentoRecuperado(
                texto=str((p.payload or {}).get("texto", "")),
                titulo=str((p.payload or {}).get("titulo", "")),
                fuente=str((p.payload or {}).get("fuente", "desconocida")),
                score=float(p.score),
                hallazgo=hallazgo,
            )
            for p in puntos
        ]

    # ------------------------------------------------------------------
    def generar_pre_reporte(self, prediccion: Any,
                            documentos: list[DocumentoRecuperado]) -> str:
        """Pre-reporte por plantilla, sin modelo de lenguaje."""
        p = getattr(prediccion, "probabilidad_escoliosis", prediccion.confianza)
        umbral = getattr(prediccion, "umbral", 0.5)

        lineas: list[str] = [
            "### PRE-REPORTE ASISTENCIAL — requiere validación por radiólogo",
            "",
            f"- **Hallazgo:** {prediccion.etiqueta}",
            f"- **Probabilidad de escoliosis:** {p:.1%} "
            f"(umbral operativo {umbral:.2f}, calibrado en desarrollo)",
        ]

        acuerdo = getattr(prediccion, "acuerdo", None)
        if acuerdo and acuerdo != "—":
            lineas.append(f"- **Acuerdo entre los 5 modelos:** {acuerdo} "
                          f"(rango {prediccion.dispersion})")

        frac = getattr(prediccion, "atencion_en_columna", float("nan"))
        if frac == frac:  # descarta NaN
            nota = ("concentrada en la columna" if frac > 0.50
                    else "repartida fuera de la banda del raquis")
            lineas.append(f"- **Atención dentro de la banda de la columna:** "
                          f"{frac:.2f} — {nota} (la banda ocupa 0,40 del ancho)")

        if documentos:
            lineas += ["", "**Criterios de referencia recuperados:**"]
            for doc in documentos:
                lineas.append(f"- **{doc.titulo}** _(relevancia {doc.score:.2f})_")
                lineas.append(f"  {doc.texto}")
                lineas.append(f"  _{doc.fuente}_")
        else:
            lineas += ["", "_No se recuperó ningún criterio para este hallazgo._"]

        lineas += [
            "",
            f"_{self.describir_metodo()}_",
            "",
            "_Herramienta de preevaluación. No es un diagnóstico y no sustituye "
            "el criterio del radiólogo._",
        ]
        return "\n".join(lineas)
