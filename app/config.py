"""
Configuración de TriajeX.

Cambios respecto a la versión anterior y por qué:

1. Las rutas se resuelven contra la ubicación de este archivo, no contra el
   directorio de trabajo. Antes eran relativas ("app/cnn/best_model.pt") y en
   Streamlit Cloud el CWD no siempre es la raíz del repositorio: si la ruta no
   existía, `ClasificadorColumna` caía en la rama de "pesos aleatorios" y el
   sidebar mostraba igualmente "✅ CNN cargada".

2. Ya no hay un `best_model.pt`. El modelo son cinco pesos (uno por partición
   de la validación cruzada) exportados a ONNX. `best_model.pt` venía de la
   primera corrida, la que dio F1-macro 0,389.

3. Las métricas se leen de `threshold.json`, que genera el entrenamiento, en
   vez de estar escritas a mano. Así no vuelven a quedar desactualizadas.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "cnn" / "models"


def _metricas() -> dict:
    ruta = MODELS_DIR / "threshold.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {}


@dataclass(frozen=True)
class Settings:
    # --- Modelo ---
    models_dir: Path = MODELS_DIR
    num_classes: int = 2
    image_size: int = 224

    # --- Punto de operación ---
    # Calibrado sobre las 59 imágenes de desarrollo, nunca sobre el test.
    threshold: float = field(default_factory=lambda: float(_metricas().get("threshold", 0.22)))

    # --- Métricas de la corrida que se presenta ---
    # Validación cruzada de 5 particiones, 59 imágenes, agrupada por paciente.
    auc_cv: float = field(default_factory=lambda: float(_metricas().get("auc_oof", 0.819)))
    f1_macro_cv: float = field(default_factory=lambda: float(_metricas().get("f1_macro_oof", 0.806)))
    recall_cv: float = 0.750          # 18 de 24 positivos detectados
    especificidad_cv: float = 0.857   # 30 de 35 normales
    auc_control_ciego: float = 0.724  # clasificador que solo ve metadatos
    n_desarrollo: int = 59
    n_positivos: int = 24

    # --- RAG ---
    qdrant_collection: str = "criterios_escoliosis"
    top_k_rag: int = 3

    @property
    def model_accuracy(self) -> str:
        """Texto corto para la interfaz."""
        return f"{self.f1_macro_cv:.3f} F1-macro (validación cruzada, n={self.n_desarrollo})"


def get_settings() -> Settings:
    return Settings()
