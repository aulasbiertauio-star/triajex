"""
Módulo de visión: ensemble de 5 ResNet18 en ONNX + mapa de atención (CAM).

POR QUÉ CAMBIÓ ESTE ARCHIVO
---------------------------
La versión anterior montaba un ResNet50 con `Linear(2048, 2)` y softmax, y
después intentaba cargar `best_model.pt`, que pesa 42,7 MB — es decir, un
ResNet18. Un state_dict de ResNet18 no entra en un ResNet50 con `strict=True`:
o lanzaba excepción, o la ruta relativa no existía en Streamlit Cloud y el
modelo se quedaba con pesos de ImageNet y una capa final aleatoria. En los dos
casos la app no estaba sirviendo el modelo entrenado, y el sidebar mostraba
igualmente "✅ CNN cargada" porque ese mensaje solo comprueba que el
constructor no reviente.

Además el modelo real tiene UNA salida con sigmoide y un umbral calibrado de
0,22, no dos salidas con softmax y argmax. Con argmax sobre softmax, el punto
de operación pasa a ser 0,5 de facto y el recall se desploma.

QUÉ HACE AHORA
--------------
- Carga los 5 modelos de la validación cruzada exportados a ONNX e int8.
  Es exactamente el ensemble que se evaluó en el informe.
- Promedia cada modelo sobre la imagen y su reflejo horizontal (TTA), igual
  que en la evaluación.
- Decide con el umbral calibrado, no con 0,5.
- Sin PyTorch: onnxruntime consume ~150 MB de RAM frente a los ~600 MB que
  gasta torch solo al importarse. Es lo que hace que la app entre en el plan
  gratuito de Streamlit y arranque en segundos.

VERIFICACIÓN
------------
Los modelos int8 se compararon contra PyTorch sobre las 70 imágenes:
diferencia máxima de probabilidad 0,035 (media 0,004) y 0 de 70 decisiones
cambiadas al umbral 0,22. Deciden igual que el modelo evaluado.

El mapa de atención es un CAM de Zhou (suma de los mapas de la última capa
convolucional pesada por la capa lineal), no Grad-CAM: no necesita gradientes,
así que sale de la misma pasada hacia adelante y no hace falta torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ETIQUETAS: list[str] = ["Normal", "Escoliosis"]

# Banda central donde cae el raquis en una proyección AP bien encuadrada.
# Ocupa el 40 % del ancho: si la atención ronda 0,40 está repartida al azar.
BANDA_COLUMNA = (0.30, 0.70)


@dataclass(frozen=True)
class PrediccionCNN:
    """Resultado tipado de la inferencia."""
    etiqueta: str
    confianza: float                 # probabilidad de la etiqueta reportada
    mapa_calor: np.ndarray           # CAM normalizado [0, 1], shape (H, W)
    top_k: dict[str, float]

    # Campos añadidos: la interfaz anterior los ignora sin romperse.
    probabilidad_escoliosis: float = 0.0
    umbral: float = 0.22
    probabilidades_por_modelo: tuple[float, ...] = ()
    atencion_en_columna: float = float("nan")

    @property
    def acuerdo(self) -> str:
        """Cuántos de los 5 modelos coinciden con la decisión final."""
        if not self.probabilidades_por_modelo:
            return "—"
        positiva = self.probabilidad_escoliosis >= self.umbral
        n = sum(1 for p in self.probabilidades_por_modelo
                if (p >= self.umbral) == positiva)
        return f"{n} de {len(self.probabilidades_por_modelo)}"

    @property
    def dispersion(self) -> str:
        if not self.probabilidades_por_modelo:
            return "—"
        return (f"{min(self.probabilidades_por_modelo):.2f} – "
                f"{max(self.probabilidades_por_modelo):.2f}")


class ClasificadorColumna:
    """
    Ensemble de 5 modelos ONNX. Mantiene la misma interfaz que la versión
    anterior: `predecir(imagen_pil) -> PrediccionCNN`.
    """

    def __init__(self, models_dir: str | Path, num_clases: int = 2,
                 tamano_imagen: int = 224, umbral: float = 0.22) -> None:
        self.tamano = int(tamano_imagen)
        self.umbral = float(umbral)
        self.num_clases = int(num_clases)

        carpeta = Path(models_dir)
        if carpeta.is_file():          # tolera que llegue la ruta de un archivo
            carpeta = carpeta.parent

        rutas = sorted(carpeta.glob("fold*.onnx"))
        if not rutas:
            raise FileNotFoundError(
                f"No hay modelos fold*.onnx en {carpeta}. "
                f"Copia ahí los cinco archivos fold0..4.int8.onnx y threshold.json. "
                f"Sin modelos NO se arranca con pesos aleatorios: es preferible "
                f"que la app falle a que muestre predicciones inventadas."
            )

        opciones = ort.SessionOptions()
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opciones.log_severity_level = 3
        self.sesiones = [
            ort.InferenceSession(str(r), opciones, providers=["CPUExecutionProvider"])
            for r in rutas
        ]
        self.nombres = [r.name for r in rutas]

    # ------------------------------------------------------------------
    def _preprocesar(self, imagen_pil: Image.Image) -> np.ndarray:
        """
        Mismo preprocesado que el entrenamiento: RGB, Resize(224,224) sin
        conservar proporción, y normalización de ImageNet.
        """
        rgb = np.asarray(imagen_pil.convert("RGB"))
        rgb = cv2.resize(rgb, (self.tamano, self.tamano), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # (1,3,H,W)

    @staticmethod
    def _sigmoide(z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-z)))

    def _cam_normalizado(self, cam: np.ndarray) -> tuple[np.ndarray, float]:
        """ReLU + min-max, y la fracción de activación dentro de la banda."""
        pos = np.maximum(cam, 0.0)
        total = float(pos.sum())
        if total > 0:
            h, w = pos.shape
            x0, x1 = int(w * BANDA_COLUMNA[0]), int(np.ceil(w * BANDA_COLUMNA[1]))
            fraccion = float(pos[:, x0:x1].sum() / total)
        else:
            fraccion = float("nan")

        mapa = cv2.resize(pos, (self.tamano, self.tamano), interpolation=cv2.INTER_CUBIC)
        mapa = np.maximum(mapa, 0.0)
        rango = mapa.max() - mapa.min()
        mapa = (mapa - mapa.min()) / (rango + 1e-8)
        return mapa.astype(np.float32), fraccion

    # ------------------------------------------------------------------
    def predecir(self, imagen_pil: Image.Image) -> PrediccionCNN:
        if not isinstance(imagen_pil, Image.Image):
            raise TypeError("Se esperaba una imagen PIL.Image.Image.")

        x = self._preprocesar(imagen_pil)
        x_espejo = np.ascontiguousarray(x[:, :, :, ::-1])

        por_modelo: list[float] = []
        cam_suma: np.ndarray | None = None

        for sesion in self.sesiones:
            vistas = []
            for i, entrada in enumerate((x, x_espejo)):
                logit, cam = sesion.run(["logit", "cam"], {"input": entrada})
                vistas.append(self._sigmoide(float(logit.reshape(-1)[0])))
                if i == 0:                       # el CAM se toma sin espejar
                    mapa = cam.reshape(cam.shape[-2], cam.shape[-1])
                    cam_suma = mapa.copy() if cam_suma is None else cam_suma + mapa
            por_modelo.append(sum(vistas) / len(vistas))

        prob = float(np.mean(por_modelo))
        cam_medio = cam_suma / len(self.sesiones)
        mapa, fraccion = self._cam_normalizado(cam_medio)

        positiva = prob >= self.umbral
        etiqueta = ETIQUETAS[1] if positiva else ETIQUETAS[0]

        return PrediccionCNN(
            etiqueta=etiqueta,
            confianza=prob if positiva else 1.0 - prob,
            mapa_calor=mapa,
            top_k={"Escoliosis": prob, "Normal": 1.0 - prob},
            probabilidad_escoliosis=prob,
            umbral=self.umbral,
            probabilidades_por_modelo=tuple(por_modelo),
            atencion_en_columna=fraccion,
        )
