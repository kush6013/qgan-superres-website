import hashlib
import io
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter


INTERPOLATION_MAP = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "box": Image.BOX,
    "lanczos": Image.LANCZOS,
}


@dataclass
class DegradationConfig:
    """
    Configuration for controlled synthetic degradation of medical images.
    All operations are deterministic given the specified random seed.
    
    DISCLAIMER:
    Synthetic degradation is a mathematical proxy for spatial and sensor resolution loss.
    It does not simulate the full physics of radiological acquisition (e.g., PSF, quantum mottle,
    scatter, CT reconstruction kernel filtering, or MRI k-space undersampling).
    """
    scale: int = 4  # Downsampling scale factor (2, 4, 8)
    downsampling: str = "bicubic"  # "bicubic", "bilinear", "nearest", "box", "lanczos"
    blur_sigma: float = 0.0  # Gaussian blur standard deviation before downsampling
    noise_type: str = "none"  # "none", "gaussian", "poisson", "speckle"
    noise_std: float = 0.0  # Standard deviation for additive Gaussian or speckle noise in [0, 255]
    jpeg_quality: Optional[int] = None  # None for lossless, or int 1-100 for compression artifacts
    seed: int = 12345

    def config_id(self) -> str:
        """Returns a deterministic SHA-256 hash identifying this exact degradation profile."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["config_id"] = self.config_id()
        data["is_clinical_acquisition"] = False
        data["synthetic_benchmark_assumption"] = True
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DegradationConfig":
        return cls(
            scale=int(data.get("scale", 4)),
            downsampling=str(data.get("downsampling", "bicubic")),
            blur_sigma=float(data.get("blur_sigma", 0.0)),
            noise_type=str(data.get("noise_type", "none")),
            noise_std=float(data.get("noise_std", 0.0)),
            jpeg_quality=data.get("jpeg_quality"),
            seed=int(data.get("seed", 12345)),
        )


class ControlledDegradationPipeline:
    """
    Executes reproducible, parameterized degradations on high-resolution ground-truth images.
    """

    @classmethod
    def degrade(cls, hr_image: Image.Image, config: DegradationConfig) -> Image.Image:
        """
        Degrades an HR PIL Image deterministically into a synthetic LR image.
        Order of operations:
          1. Optional Gaussian optical/sensor blur on HR.
          2. Spatial downsampling by factor of `config.scale`.
          3. Optional additive or multiplicative sensor noise on LR.
          4. Optional lossy JPEG compression simulation.
        """
        # Ensure working in RGB or L
        mode = hr_image.mode
        work_img = hr_image.copy()

        # 1. Blur
        if config.blur_sigma > 0.0:
            arr = np.asarray(work_img, dtype=np.float32)
            if arr.ndim == 2:
                blurred = gaussian_filter(arr, sigma=config.blur_sigma, mode="reflect")
            else:
                blurred = np.zeros_like(arr)
                for c in range(arr.shape[2]):
                    blurred[..., c] = gaussian_filter(arr[..., c], sigma=config.blur_sigma, mode="reflect")
            work_img = Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode=mode)

        # 2. Downsampling
        orig_w, orig_h = work_img.size
        lr_w = max(1, orig_w // config.scale)
        lr_h = max(1, orig_h // config.scale)
        resample_method = INTERPOLATION_MAP.get(config.downsampling.lower(), Image.BICUBIC)
        lr_img = work_img.resize((lr_w, lr_h), resample=resample_method)

        # 3. Noise simulation with fixed seed
        if config.noise_type != "none" and config.noise_std > 0.0:
            rng = np.random.RandomState(config.seed)
            arr_lr = np.asarray(lr_img, dtype=np.float32)

            if config.noise_type == "gaussian":
                noise = rng.normal(0, config.noise_std, arr_lr.shape)
                arr_lr = np.clip(arr_lr + noise, 0, 255)
            elif config.noise_type == "speckle":
                # Multiplicative noise common in ultrasound: I' = I + I * n
                noise = rng.normal(0, config.noise_std / 128.0, arr_lr.shape)
                arr_lr = np.clip(arr_lr + (arr_lr * noise), 0, 255)
            elif config.noise_type == "poisson":
                # Shot noise approximation
                scaled = np.clip(arr_lr, 0, 255) / 255.0
                vals = len(np.unique(scaled))
                vals = 2 ** np.ceil(np.log2(max(2, vals)))
                noisy = rng.poisson(scaled * vals) / float(vals) * 255.0
                arr_lr = np.clip(noisy, 0, 255)

            lr_img = Image.fromarray(arr_lr.astype(np.uint8), mode=mode)

        # 4. JPEG compression simulation
        if config.jpeg_quality is not None and 1 <= config.jpeg_quality <= 100:
            bio = io.BytesIO()
            # JPEG requires RGB or L
            save_img = lr_img if lr_img.mode in ("RGB", "L") else lr_img.convert("RGB")
            save_img.save(bio, format="JPEG", quality=config.jpeg_quality)
            bio.seek(0)
            lr_img = Image.open(bio).copy()
            if mode != lr_img.mode:
                lr_img = lr_img.convert(mode)

        return lr_img
