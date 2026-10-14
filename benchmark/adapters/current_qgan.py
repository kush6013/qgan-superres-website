import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from .base import BenchmarkModel
from models.qgan_model import QGANGenerator, load_generator


MODALITY_CHECKPOINT_MAP = {
    "general": Path("models/qgan_generator.pt"),
    "brain_mri": Path("models/qgan_generator_mri.pt"),
    "chest_xray": Path("models/qgan_generator_xray.pt"),
    "ct_scan": Path("models/qgan_generator_ct.pt"),
}


class CurrentQGANModel(BenchmarkModel):
    """
    Adapter wrapping the project's current Quantum-Classical GAN Generator.
    Seamlessly handles modality checkpoint selection, hardware device management,
    runtime memory profiling, and graceful fallback when checkpoints are absent.
    """

    def __init__(
        self,
        name: str = "current_qgan",
        device: Optional[torch.device] = None,
        checkpoint_dir: Path | str = "models",
    ):
        super().__init__(name=name)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self._loaded_models: Dict[str, Optional[QGANGenerator]] = {}

    def _get_model_for_modality(self, modality: str) -> Tuple[Optional[QGANGenerator], str, bool]:
        mod = (modality or "general").lower().strip()
        if mod in self._loaded_models:
            gen = self._loaded_models[mod]
            return gen, mod, gen is not None

        # Resolve checkpoint path
        ckpt_filename = f"qgan_generator_{mod.split('_')[0]}.pt" if mod != "general" else "qgan_generator.pt"
        if self.checkpoint_dir == Path("models") and mod in MODALITY_CHECKPOINT_MAP:
            target_path = MODALITY_CHECKPOINT_MAP[mod]
        else:
            target_path = self.checkpoint_dir / ckpt_filename

        gen = load_generator(target_path, self.device)
        if gen is not None:
            self._loaded_models[mod] = gen
            return gen, mod, True

        # Fallback to general checkpoint
        general_path = self.checkpoint_dir / "qgan_generator.pt"
        gen = load_generator(general_path, self.device)
        if gen is not None:
            self._loaded_models[mod] = gen
            return gen, "general (fallback)", True

        # Neither found -> fallback to None
        self._loaded_models[mod] = None
        return None, "bicubic (fallback)", False

    def predict(
        self,
        input_image: Image.Image,
        scale: int,
        modality: str = "general"
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        target_w = max(1, input_image.width * scale)
        target_h = max(1, input_image.height * scale)

        model, used_mod, ckpt_loaded = self._get_model_for_modality(modality)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()

        if model is None:
            # Fallback bicubic when weights are not present
            output_image = input_image.resize((target_w, target_h), resample=Image.BICUBIC)
            fallback = True
        else:
            fallback = False
            # RGB conversion for 3-channel generator head
            work_rgb = input_image.convert("RGB")
            input_tensor = TF.to_tensor(work_rgb).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                output_tensor = model(input_tensor)

            if output_tensor.min() < 0 or output_tensor.max() > 1:
                output_tensor = (output_tensor + 1.0) / 2.0

            output_tensor = output_tensor.clamp(0, 1).cpu().squeeze(0)
            full_sr = TF.to_pil_image(output_tensor)

            if scale == 8:
                output_image = full_sr
            else:
                # Downscale 8x output to target scale (2x or 4x)
                output_image = full_sr.resize((target_w, target_h), resample=Image.LANCZOS)

            # Preserve grayscale if input was single-channel
            if input_image.mode in ("L", "1"):
                output_image = output_image.convert("L")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        peak_mem_mb = 0.0
        if torch.cuda.is_available():
            peak_mem_mb = round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2)

        metadata = {
            "model_name": self.name,
            "target_scale": scale,
            "used_modality": used_mod,
            "inference_ms": round(elapsed_ms, 3),
            "peak_memory_mb": peak_mem_mb,
            "output_dimensions": [output_image.width, output_image.height],
            "checkpoint_loaded": ckpt_loaded,
            "fallback_used": fallback,
        }

        return output_image, metadata
