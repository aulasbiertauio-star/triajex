"""Dashboard de triaje con Streamlit: imagen -> CNN -> RAG -> pre-reporte."""
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import get_settings
from cnn.model import ClasificadorColumna
from rag.qdrant_store import AlmacenRAG


@st.cache_resource
def cargar_cnn() -> ClasificadorColumna:
    cfg = get_settings()
    return ClasificadorColumna(cfg.model_path, cfg.num_classes, cfg.image_size)


@st.cache_resource
def cargar_rag() -> AlmacenRAG:
    cfg = get_settings()
    return AlmacenRAG(cfg.qdrant_host, cfg.qdrant_port,
                      cfg.qdrant_collection, cfg.embedding_model)


def superponer_gradcam(imagen: Image.Image, cam: np.ndarray) -> np.ndarray:
    """Superpone el Grad-CAM sobre la radiografía original."""
    img_np = np.array(imagen.convert("RGB"))
    cam_redim = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    calor = cv2.applyColorMap(np.uint8(255 * cam_redim), cv2.COLORMAP_JET)
    calor = cv2.cvtColor(calor, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.6, calor, 0.4, gamma=0)


def main() -> None:
    st.set_page_config(page_title="Triaje Radiografía de Columna",
                         layout="wide")
    st.title("🩻 TriajeX — Radiografías de Columna")

    with st.sidebar:
        try:
            cnn = cargar_cnn()
            st.success("✅ CNN cargada")
        except Exception as exc:
            st.error(f"❌ Error en CNN: {exc}")
            st.stop()
        try:
            rag = cargar_rag()
            st.success("✅ Conectado a Qdrant")
        except Exception as exc:
            st.warning(f"⚠️ RAG no disponible: {exc}")
            rag = None

    archivo = st.file_uploader("Suba la radiografía (PNG/JPG)",
                               type=["png", "jpg", "jpeg"])
    if archivo is not None:
        imagen = Image.open(archivo)
        col1, col2 = st.columns(2)
        col1.image(imagen, caption="Radiografía original",
                    use_container_width=True)
        try:
            with st.spinner("Analizando con la CNN…"):
                prediccion = cnn.predecir(imagen)
            col2.image(superponer_gradcam(imagen, prediccion.mapa_calor),
                       caption=f"Grad-CAM — {prediccion.etiqueta}",
                       use_container_width=True)
            st.subheader(f"Predicción: **{prediccion.etiqueta}** "
                         f"({prediccion.confianza:.2%})")
            st.bar_chart(prediccion.top_k)
            st.divider()
            if rag is not None:
                with st.spinner("Consultando literatura médica (RAG)…"):
                    docs = rag.consultar(prediccion.etiqueta,
                                         get_settings().top_k_rag)
                st.markdown(rag.generar_pre_reporte(prediccion, docs))
            else:
                st.info("RAG no configurado; solo hallazgo de la CNN.")
        except (TypeError, RuntimeError, ConnectionError) as exc:
            st.error(f"Error durante el análisis: {exc}")


if __name__ == "__main__":
    main()