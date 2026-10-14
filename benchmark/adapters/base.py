from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from PIL import Image


class BenchmarkModel(ABC):
    """
    Standard abstract interface for all benchmarked super-resolution models.
    Enforces identical inputs, scales, modalities, and execution telemetry across all models.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(
        self,
        input_image: Image.Image,
        scale: int,
        modality: str = "general"
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Executes super-resolution enhancement on input low-resolution image.
        Returns:
            (enhanced_image, execution_metadata)
            where execution_metadata contains:
              - inference_ms: float (measured execution time in milliseconds)
              - peak_memory_mb: Optional[float]
              - model_name: str
              - target_scale: int
              - used_modality: str
        """
        pass
