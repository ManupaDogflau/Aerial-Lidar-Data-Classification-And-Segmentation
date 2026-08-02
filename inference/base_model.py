from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class BaseSegmentationModel(ABC):
    """
    Interfaz común para cualquier modelo de segmentación LiDAR.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        num_classes: int = 2,
    ):

        self.checkpoint = checkpoint
        self.device = self._select_device(device)
        self.num_classes = num_classes

        self.model = None

    #################################################################
    # Métodos obligatorios
    #################################################################

    @abstractmethod
    def load(self):
        """
        Carga el modelo y los pesos.
        """
        pass

    @abstractmethod
    def preprocess(
        self,
        cloud: np.ndarray
    ):
        """
        Convierte una nube NxM en la entrada del modelo.

        Parameters
        ----------
        cloud : np.ndarray
            Nube de puntos.

        Returns
        -------
        object
            Entrada preparada para la red.
        """
        pass

    @abstractmethod
    def forward(
        self,
        inputs
    ):
        """
        Ejecuta únicamente la inferencia.
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        prediction
    ) -> np.ndarray:
        """
        Convierte la salida de la red en etiquetas.
        """
        pass

    #################################################################
    # API pública
    #################################################################

    def predict(
        self,
        cloud: np.ndarray
    ) -> np.ndarray:

        inputs = self.preprocess(cloud)

        prediction = self.forward(inputs)

        prediction = self.postprocess(prediction)

        return prediction

    #################################################################
    # Información del modelo
    #################################################################

    @property
    def name(self):

        return self.__class__.__name__

    #################################################################
    # Utilidades
    #################################################################

    def _select_device(
        self,
        device: str
    ):

        if device == "auto":

            try:

                import torch

                if torch.cuda.is_available():
                    return "cuda"

            except Exception:
                pass

            return "cpu"

        return device