"""
Módulo de Visión por Computadora (CNN + Grad-CAM).
Estructura preparada para montar un modelo .pt preentrenado posteriormente.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


@dataclass(frozen=True)
class PrediccionCNN:
    """Resultado tipado de la inferencia de la CNN."""
    etiqueta: str
    confianza: float
    mapa_calor: np.ndarray  # Grad-CAM normalizado [0, 1], shape (H, W)
    top_k: dict[str, float]


class GradCAM:
    """Calcula el mapa de calor Grad-CAM sobre la última capa convolucional."""

    def __init__(self, modelo: torch.nn.Module, nombre_capa: str) -> None:
        self.modelo = modelo
        self._activaciones: Optional[torch.Tensor] = None
        self._gradientes: Optional[torch.Tensor] = None
        capa = dict(modelo.named_modules())[nombre_capa]
        capa.register_forward_hook(self._guardar_activacion)
        capa.register_full_backward_hook(self._guardar_gradiente)

    def _guardar_activacion(self, _mod: torch.nn.Module,
                            _ent: tuple, salida: torch.Tensor) -> None:
        """Hook forward: almacena las activaciones de la capa."""
        self._activaciones = salida.detach()

    def _guardar_gradiente(self, _mod: torch.nn.Module,
                           _grad_ent: tuple, grad_sal: tuple) -> None:
        """Hook backward: almacena los gradientes respecto a la capa."""
        self._gradientes = grad_sal[0].detach()

    def generar(self, imagen: torch.Tensor, clase: int) -> np.ndarray:
        """Genera el mapa Grad-CAM para la clase predicha."""
        assert self._activaciones is not None and self._gradientes is not None
        pesos = self._gradientes.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (pesos * self._activaciones).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=imagen.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


ETIQUETAS: list[str] = ["normal", "escoliosis", "espina_bifida", "fractura"]


class ClasificadorColumna:
    """
    Envoltorio del modelo CNN. NO entrena el modelo; solo monta la
    arquitectura base y carga los pesos si el archivo .pt existe.
    """

    def __init__(self, ruta_modelo: str, num_clases: int,
                 tamano_imagen: int) -> None:
        self.dispositivo = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_clases = num_clases

        # Arquitectura base con transfer-learning (ajustar al modelo final)
        self.modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.modelo.fc = torch.nn.Linear(self.modelo.fc.in_features, num_clases)
        self.modelo.to(self.dispositivo).eval()

        ruta = Path(ruta_modelo)
        if ruta.exists():
            try:
                estado = torch.load(ruta, map_location=self.dispositivo)
                self.modelo.load_state_dict(estado)
            except (RuntimeError, OSError) as exc:
                raise RuntimeError(
                    f"No se pudieron cargar los pesos desde {ruta}: {exc}"
                ) from exc
        else:
            import warnings
            warnings.warn(
                f"Modelo no encontrado en {ruta}. La inferencia usará "
                "pesos aleatorios (solo para pruebas de integración).")

        self.gradcam = GradCAM(self.modelo, nombre_capa="layer4")
        self.preproceso = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((tamano_imagen, tamano_imagen)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predecir(self, imagen_pil: Image.Image) -> PrediccionCNN:
        """Ejecuta la inferencia y calcula el Grad-CAM."""
        if not isinstance(imagen_pil, Image.Image):
            raise TypeError("Se esperaba una imagen PIL.Image.Image.")

        tensor = self.preproceso(imagen_pil).unsqueeze(0).to(self.dispositivo)

        # Forward con gradientes habilitados para Grad-CAM
        self.modelo.zero_grad()
        logits = self.modelo(tensor)
        probabilidades = F.softmax(logits, dim=1)
        confianzas, indices = torch.topk(probabilidades, k=self.num_clases)
        clase_predicha = int(indices[0, 0])

        logits[0, clase_predicha].backward()
        mapa = self.gradcam.generar(tensor, clase_predicha)

        return PrediccionCNN(
            etiqueta=ETIQUETAS[clase_predicha],
            confianza=float(confianzas[0, 0]),
            mapa_calor=mapa,
            top_k={ETIQUETAS[int(indices[0, i])]: float(confianzas[0, i])
                   for i in range(self.num_clases)},
        )