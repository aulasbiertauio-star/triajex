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
            
            # --- 1. MÓDULO RAG Y TRANSPARENCIA ---
            if rag is not None:
                with st.spinner("Consultando literatura médica (RAG)…"):
                    docs = rag.consultar(prediccion.etiqueta,
                                         get_settings().top_k_rag)
                st.markdown(rag.generar_pre_reporte(prediccion, docs))
                
                precision = get_settings().model_accuracy
                st.info(
                    f"ℹ️ **Transparencia MLOps - Limitación de Muestra:**\n\n"
                    f"F1-Macro validado en desarrollo: **{precision}**. "
                    f"Debido a la limitación actual del dataset (70 imágenes), el modelo "
                    f"fue calibrado con un umbral de alta sensibilidad (Recall 95.8%) para "
                    f"minimizar falsos negativos. Herramienta estrictamente de apoyo; **requiere validación por médico radiólogo**."
                )
            else:
                st.info("RAG no configurado; solo hallazgo de la CNN.")
                
            st.divider()
            
            # --- 2. NUEVO MÓDULO: BIOMETRÍA INTERACTIVA (COBB) ---
            st.subheader("📐 Módulo de Biometría Asistida (Ángulo de Cobb)")
            st.write("Herramienta interactiva para validación humana de la curvatura espinal (Sugerencia de Mentores).")
            
            with st.expander("Abrir Calculadora Interactiva de Cobb", expanded=False):
                img_np_cobb = np.array(imagen.convert("RGB"))
                h_cobb, w_cobb, _ = img_np_cobb.shape
                
                st.info("Ajuste los controles para alinear las líneas de referencia con las vértebras límite superior e inferior.")
                
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Límite Superior (Vértebra Cervical/Torácica)**")
                    y_sup = st.slider("Altura de la línea superior", 0, h_cobb, int(h_cobb * 0.25))
                    ang_sup = st.slider("Inclinación línea superior (°)", -60, 60, 0)
                with cc2:
                    st.markdown("**Límite Inferior (Vértebra Lumbar)**")
                    y_inf = st.slider("Altura de la línea inferior", 0, h_cobb, int(h_cobb * 0.75))
                    ang_inf = st.slider("Inclinación línea inferior (°)", -60, 60, 0)
                    
                # Cálculos trigonométricos
                rad_sup = np.deg2rad(ang_sup)
                rad_inf = np.deg2rad(ang_inf)
                L = w_cobb // 2
                x_c = w_cobb // 2
                
                pt1_sup = (int(x_c - L * np.cos(rad_sup)), int(y_sup - L * np.sin(rad_sup)))
                pt2_sup = (int(x_c + L * np.cos(rad_sup)), int(y_sup + L * np.sin(rad_sup)))
                
                pt1_inf = (int(x_c - L * np.cos(rad_inf)), int(y_inf - L * np.sin(rad_inf)))
                pt2_inf = (int(x_c + L * np.cos(rad_inf)), int(y_inf + L * np.sin(rad_inf)))
                
                # Dibujar las líneas sobre la radiografía (OpenCV)
                img_dibujo = img_np_cobb.copy()
                cv2.line(img_dibujo, pt1_sup, pt2_sup, (0, 255, 0), 4)  # Verde
                cv2.line(img_dibujo, pt1_inf, pt2_inf, (255, 100, 0), 4) # Azul
                
                # Calcular Ángulo de Cobb (diferencia geométrica absoluta)
                cobb_angle = abs(ang_sup - ang_inf)
                
                if cobb_angle < 10:
                    sev = "Normal / Variación anatómica"
                elif cobb_angle <= 20:
                    sev = "Escoliosis Leve"
                elif cobb_angle <= 40:
                    sev = "Escoliosis Moderada"
                else:
                    sev = "Escoliosis Severa"
                    
                col_img, col_res = st.columns([2, 1])
                with col_img:
                    st.image(img_dibujo, caption="Alineación Trigonométrica Manual", use_container_width=True)
                with col_res:
                    st.metric(label="Ángulo de Cobb", value=f"{cobb_angle}°", 
                              delta=sev, delta_color="inverse" if cobb_angle >= 10 else "normal")
                    st.write("**Justificación:** El ángulo de Cobb es el estándar clínico. Esta calculadora empodera al radiólogo para validar rápidamente el hallazgo de la red neuronal (Grad-CAM) y tomar decisiones de derivación, sin salir de la plataforma.")

        except (TypeError, RuntimeError, ConnectionError) as exc:
            st.error(f"Error durante el análisis: {exc}")

if __name__ == "__main__":
    main()