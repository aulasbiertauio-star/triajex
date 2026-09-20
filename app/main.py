"""
Dashboard de triaje con Streamlit: imagen -> ensemble CNN -> RAG -> pre-reporte.

Cambios respecto a la versión anterior:
  - `cargar_cnn` ahora carga el ensemble ONNX real (5 modelos), no un ResNet50
    con pesos aleatorios.
  - El texto de transparencia usa las métricas de la corrida que se presenta,
    leídas de threshold.json, en vez de cifras escritas a mano y desfasadas.
  - Se elimina la afirmación "Recall 95.8%": esa cifra corresponde al umbral
    0,01, donde la especificidad cae a 0,23. El punto de operación real es
    0,22 con recall 0,75.
  - El mapa se llama CAM, no Grad-CAM, porque eso es lo que se calcula.
  - "Consultando literatura médica" pasa a decir lo que de verdad hay: una base
    de criterios escrita por el equipo, recuperada por TF-IDF.
"""

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
    return ClasificadorColumna(cfg.models_dir, cfg.num_classes,
                               cfg.image_size, cfg.threshold)


@st.cache_resource
def cargar_rag() -> AlmacenRAG:
    cfg = get_settings()
    return AlmacenRAG(cfg.qdrant_collection)


def superponer_cam(imagen: Image.Image, cam: np.ndarray) -> np.ndarray:
    """Superpone el mapa de atención sobre la radiografía original."""
    img_np = np.array(imagen.convert("RGB"))
    cam_redim = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    calor = cv2.applyColorMap(np.uint8(255 * cam_redim), cv2.COLORMAP_JET)
    calor = cv2.cvtColor(calor, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.6, calor, 0.4, gamma=0)


def bloque_cobb(imagen: Image.Image) -> None:
    """Calculadora manual del ángulo de Cobb. La mide el humano, no el modelo."""
    st.subheader("📐 Biometría asistida — ángulo de Cobb")
    st.write("Herramienta **manual**: las líneas las alinea la persona. "
             "El modelo no mide ángulos ni identifica vértebras.")

    with st.expander("Abrir calculadora interactiva de Cobb", expanded=False):
        img_np = np.array(imagen.convert("RGB"))
        h, w, _ = img_np.shape

        st.info("Ajuste los controles para alinear las líneas con las placas "
                "terminales de las vértebras límite, superior e inferior.")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Vértebra límite superior**")
            y_sup = st.slider("Altura de la línea superior", 0, h, int(h * 0.25))
            ang_sup = st.slider("Inclinación línea superior (°)", -60, 60, 0)
        with cc2:
            st.markdown("**Vértebra límite inferior**")
            y_inf = st.slider("Altura de la línea inferior", 0, h, int(h * 0.75))
            ang_inf = st.slider("Inclinación línea inferior (°)", -60, 60, 0)

        rad_sup, rad_inf = np.deg2rad(ang_sup), np.deg2rad(ang_inf)
        L, x_c = w // 2, w // 2

        pt1_sup = (int(x_c - L * np.cos(rad_sup)), int(y_sup - L * np.sin(rad_sup)))
        pt2_sup = (int(x_c + L * np.cos(rad_sup)), int(y_sup + L * np.sin(rad_sup)))
        pt1_inf = (int(x_c - L * np.cos(rad_inf)), int(y_inf - L * np.sin(rad_inf)))
        pt2_inf = (int(x_c + L * np.cos(rad_inf)), int(y_inf + L * np.sin(rad_inf)))

        dibujo = img_np.copy()
        cv2.line(dibujo, pt1_sup, pt2_sup, (0, 255, 0), 4)
        cv2.line(dibujo, pt1_inf, pt2_inf, (255, 100, 0), 4)

        cobb = abs(ang_sup - ang_inf)
        if cobb < 10:
            sev = "Por debajo del umbral de escoliosis (< 10°)"
        elif cobb <= 25:
            sev = "Curva leve"
        elif cobb <= 45:
            sev = "Curva moderada"
        else:
            sev = "Curva severa"

        col_img, col_res = st.columns([2, 1])
        with col_img:
            st.image(dibujo, caption="Alineación manual de las líneas de Cobb",
                     use_container_width=True)
        with col_res:
            st.metric(label="Ángulo de Cobb (medición manual)", value=f"{cobb}°")
            st.write(f"**{sev}**")
            st.caption(
                "Se considera escoliosis a partir de 10°. La medición del ángulo "
                "de Cobb tiene una variabilidad entre observadores del orden de "
                "5°, así que este valor se lee con ese margen. Los rangos son "
                "orientativos; la conducta la decide el especialista."
            )


def main() -> None:
    cfg = get_settings()
    st.set_page_config(page_title="Triaje Radiografía de Columna", layout="wide")
    st.title("🩻 TriajeX — Radiografías de Columna")
    st.caption("Preevaluación de escoliosis en proyección AP. Apoyo a la "
               "lectura, no diagnóstico.")

    with st.sidebar:
        try:
            cnn = cargar_cnn()
            st.success(f"✅ Ensemble cargado — {len(cnn.sesiones)} modelos")
            st.caption(f"Umbral operativo {cfg.threshold:.2f}")
        except Exception as exc:
            st.error(f"❌ No se pudo cargar el modelo: {exc}")
            st.stop()
        try:
            rag = cargar_rag()
            st.success(f"✅ Base de criterios — {rag.n_documentos} documentos")
        except Exception as exc:
            st.warning(f"⚠️ Base de criterios no disponible: {exc}")
            rag = None

        st.divider()
        st.markdown("**Desempeño (validación cruzada, n=%d)**" % cfg.n_desarrollo)
        st.markdown(
            f"- AUC **{cfg.auc_cv:.3f}**\n"
            f"- F1-macro **{cfg.f1_macro_cv:.3f}**\n"
            f"- Recall **{cfg.recall_cv:.2f}** · Especificidad **{cfg.especificidad_cv:.2f}**\n"
            f"- Control ciego (sin anatomía): AUC **{cfg.auc_control_ciego:.3f}**"
        )

    archivo = st.file_uploader("Suba la radiografía (PNG/JPG)",
                               type=["png", "jpg", "jpeg"])
    if archivo is None:
        return

    imagen = Image.open(archivo)
    col1, col2 = st.columns(2)
    col1.image(imagen, caption="Radiografía original", use_container_width=True)

    try:
        with st.spinner("Analizando con el ensemble…"):
            pred = cnn.predecir(imagen)

        col2.image(superponer_cam(imagen, pred.mapa_calor),
                   caption=f"Mapa de atención (CAM) — {pred.etiqueta}",
                   use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Hallazgo", pred.etiqueta)
        m2.metric("Probabilidad de escoliosis",
                  f"{pred.probabilidad_escoliosis:.1%}",
                  help=f"Se marca como escoliosis a partir de {pred.umbral:.2f}")
        m3.metric("Acuerdo entre modelos", pred.acuerdo,
                  help=f"Rango de los 5 modelos: {pred.dispersion}")

        st.bar_chart(pred.top_k)

        if pred.atencion_en_columna != pred.atencion_en_columna:
            st.caption(
                "El mapa de atención sale plano: ninguna región aportó evidencia "
                "positiva de escoliosis en esta imagen. Es lo esperable en un "
                "negativo con probabilidad muy baja."
            )
        else:
            frac = pred.atencion_en_columna
            st.caption(
                f"Atención dentro de la banda central donde discurre el raquis: "
                f"**{frac:.2f}** (la banda ocupa 0,40 del ancho). "
                + ("La atención está concentrada en la columna."
                   if frac > 0.50 else
                   "La atención cae mayormente fuera de la columna: es coherente "
                   "con el sesgo de adquisición documentado entre las dos clases.")
            )

        st.divider()

        # --- RAG y transparencia ---
        if rag is not None:
            with st.spinner("Recuperando criterios de informe…"):
                docs = rag.consultar(pred.etiqueta, cfg.top_k_rag)
            st.markdown(rag.generar_pre_reporte(pred, docs))
        else:
            st.info("Base de criterios no disponible; solo el hallazgo del modelo.")

        st.info(
            f"ℹ️ **Transparencia sobre el desempeño**\n\n"
            f"Validación cruzada de 5 particiones sobre {cfg.n_desarrollo} imágenes "
            f"de desarrollo ({cfg.n_positivos} positivas), agrupadas por paciente: "
            f"AUC **{cfg.auc_cv:.3f}**, F1-macro **{cfg.f1_macro_cv:.3f}**.\n\n"
            f"En el punto de operación elegido ({cfg.threshold:.2f}) el recall es "
            f"**{cfg.recall_cv:.2f}** y la especificidad **{cfg.especificidad_cv:.2f}**. "
            f"Con este dataset **ningún umbral alcanza recall 0,90**, así que la "
            f"herramienta no sirve como filtro de descarte.\n\n"
            f"Un clasificador que solo ve el tamaño, el brillo y el peso del archivo "
            f"—sin anatomía— alcanza AUC **{cfg.auc_control_ciego:.3f}**. El margen "
            f"atribuible a la anatomía es de unos "
            f"**{cfg.auc_cv - cfg.auc_control_ciego:+.3f}**.\n\n"
            f"Solo escoliosis, solo proyección AP. **Requiere validación por "
            f"médico radiólogo.**"
        )

        st.divider()
        bloque_cobb(imagen)

    except (TypeError, RuntimeError, ConnectionError, FileNotFoundError) as exc:
        st.error(f"Error durante el análisis: {exc}")


if __name__ == "__main__":
    main()
