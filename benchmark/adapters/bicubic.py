import time
from typing import Tuple, Dict, Any
from PIL import Image
from .base import BenchmarkModel


class BicubicBaselineModel(BenchmarkModel):
    """
    Mandatory standard Bicubic interpolation baseline.
    Serves as the reference lower bound for resolution magnification.
    """

    def __init__(self, name: str = "bicubic"):
        super().__init__(name=name)

    def predict(
        self,
        input_image: Image.Image,
        scale: int,
        modality: str = "general"
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        target_w = max(1, input_image.width * scale)
        target_h = max(1, input_image.height * scale)

        start_time = time.perf_counter()
        # Execute bicubic upsampling
        output_image = input_image.resize((target_w, target_h), resample=Image.BICUBIC)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = {
            "model_name": self.name,
            "target_scale": scale,
            "used_modality": modality,
            "inference_ms": round(elapsed_ms, 3),
            "peak_memory_mb": 0.0,
            "output_dimensions": [target_w, target_h],
            "checkpoint_loaded": False,
        }

        return output_image, metadata
