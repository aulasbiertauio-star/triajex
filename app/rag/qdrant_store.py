"""Módulo RAG: conexión a Qdrant y recuperación de literatura médica."""
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, VectorParams, Distance
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class DocumentoRecuperado:
    """Fragmento de literatura médica recuperado de Qdrant."""
    texto: str
    fuente: str
    score: float
    hallazgo: str


class AlmacenRAG:
    """
    Cliente de Qdrant: usa almacenamiento local embebido en disco para
    garantizar estabilidad total en Streamlit Cloud sin requerir servidor de red.
    """

    def __init__(self, host: str, port: int, coleccion: str,
                 modelo_embedding: str) -> None:
        self.coleccion = coleccion
        try:
            # Forzar modo local embebido en disco (ideal para Cloud)
            self.cliente = QdrantClient(path="./qdrant_storage")
        except Exception as exc:
            try:
                self.cliente = QdrantClient(host=host, port=port, timeout=3)
            except Exception:
                raise ConnectionError(
                    f"No fue posible inicializar Qdrant: {exc}"
                ) from exc

        try:
            self.embedder: SentenceTransformer = SentenceTransformer(
                modelo_embedding)
        except Exception as exc:
            raise RuntimeError(
                f"Error al cargar el modelo de embeddings: {exc}") from exc

    def _embed(self, texto: str) -> list[float]:
        """Genera el vector de embedding de una consulta de texto."""
        vector: np.ndarray = self.embedder.encode(texto)
        return vector.tolist()

    def consultar(self, hallazgo: str, top_k: int) -> list[DocumentoRecuperado]:
        """Recupera los fragmentos más relevantes para el hallazgo de forma segura."""
        consulta = (
            f"Radiografía de columna vertebral con hallazgo de {hallazgo}. "
            "Protocolo de triaje y criterios de reporte estandarizado."
        )
        try:
            collections = [c.name for c in self.cliente.get_collections().collections]
            if self.coleccion not in collections:
                self.cliente.create_collection(
                    collection_name=self.coleccion,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                return []

            respuesta = self.cliente.query_points(
                collection_name=self.coleccion,
                query=self._embed(consulta),
                query_filter=Filter(must=[FieldCondition(
                    key="hallazgo", match=MatchValue(value=hallazgo))]),
                limit=top_k,
                with_payload=True,
            )
            resultado = respuesta.points
        except Exception as exc:
            import warnings
            warnings.warn(f"Aviso en consulta RAG: {exc}")
            return []

        documentos: list[DocumentoRecuperado] = []
        for punto in resultado:
            payload: dict[str, Any] = punto.payload or {}
            documentos.append(DocumentoRecuperado(
                texto=str(payload.get("texto", "")),
                fuente=str(payload.get("fuente", "desconocida")),
                score=float(punto.score),
                hallazgo=hallazgo,
            ))
        return documentos

    def generar_pre_reporte(self, prediccion: Any,
                            documentos: list[DocumentoRecuperado]) -> str:
        """Compone el pre-reporte estandarizado (plantilla, sin LLM)."""
        lineas: list[str] = [
            "### PRE-REPORTE ASISTENCIAL (no sustituye criterio médico)",
            f"- **Hallazgo principal:** {prediccion.etiqueta}",
            f"- **Confianza del modelo:** {prediccion.confianza:.2%}",
            "",
            "**Distribución de probabilidades:**",
        ]
        for etiqueta, prob in prediccion.top_k.items():
            lineas.append(f"- {etiqueta}: {prob:.2%}")
        if documentos:
            lineas += ["", "**Literatura de referencia (RAG):**"]
            for doc in documentos:
                lineas.append(f"- [{doc.fuente}] (relevancia {doc.score:.3f})")
                lineas.append(f"  {doc.texto}")
        else:
            lineas += ["", "_Nota: Base de conocimiento RAG en modo autónomo._"]
        lineas += ["", "_Requiere validación por radiólogo._"]
        return "\n".join(lineas)