import math
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import lpips
except Exception:
    lpips = None

from metrics import calculate_psnr, calculate_ssim, calculate_laplacian_sharpness


class ResearchMetricsEvaluator:
    """
    Research-grade Image Quality Assessment (IQA) & Performance Evaluator.
    Computes mathematical fidelity (PSNR, SSIM), edge sharpness (Laplacian variance),
    and execution telemetry with strict numerical edge-case safeguards.
    """

    def __init__(self, use_lpips_if_available: bool = True):
        self.use_lpips = use_lpips_if_available and (lpips is not None)
        self._lpips_loss_fn = None
        if self.use_lpips:
            try:
                self._lpips_loss_fn = lpips.LPIPS(net="alex", verbose=False)
            except Exception:
                self._lpips_loss_fn = None
                self.use_lpips = False

    @staticmethod
    def _to_float32_array(img: Image.Image) -> np.ndarray:
        """Converts PIL image to float32 grayscale numpy array in range [0, 255]."""
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.float32)

    @classmethod
    def validate_finite_numerical_range(cls, arr: np.ndarray, name: str) -> None:
        """Validates that image array contains valid, finite numerical values."""
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values (NaN or Inf) detected in {name} array")
        if arr.size == 0:
            raise ValueError(f"Empty image array provided for {name}")

    def evaluate_sample(
        self,
        predicted_image: Image.Image,
        ground_truth_image: Image.Image,
        baseline_image: Optional[Image.Image] = None,
        reference_type: str = "synthetic_ground_truth",
    ) -> Dict[str, Any]:
        """
        Evaluates single super-resolved prediction against ground truth and baseline.
        Returns detailed metric record with explicit error handling.
        """
        result: Dict[str, Any] = {
            "reference_type": reference_type,
            "status": "success",
            "error": None,
        }

        try:
            # Spatial alignment: ground truth and prediction must match spatial dimensions
            pred_w, pred_h = predicted_image.size
            gt_w, gt_h = ground_truth_image.size

            if (pred_w, pred_h) != (gt_w, gt_h):
                # Align reference to prediction spatial dimensions using Lanczos resampler
                aligned_gt = ground_truth_image.resize((pred_w, pred_h), resample=Image.LANCZOS)
            else:
                aligned_gt = ground_truth_image

            arr_pred = self._to_float32_array(predicted_image)
            arr_gt = self._to_float32_array(aligned_gt)

            self.validate_finite_numerical_range(arr_pred, "predicted_image")
            self.validate_finite_numerical_range(arr_gt, "ground_truth_image")

            # 1. PSNR Calculation
            psnr_val = calculate_psnr(arr_pred, arr_gt)
            result["psnr"] = psnr_val

            # 2. SSIM Calculation (handles small or constant images safely)
            if min(arr_pred.shape) < 7:
                # Images smaller than standard 7x7 SSIM window cannot compute local covariance
                result["ssim"] = None
                result["ssim_warning"] = "Image dimensions smaller than 7x7; local SSIM window undefined"
            else:
                ssim_val = calculate_ssim(arr_pred, arr_gt)
                result["ssim"] = ssim_val

            # 3. Laplacian Edge Sharpness
            sharpness_val = calculate_laplacian_sharpness(arr_pred)
            result["sharpness"] = sharpness_val

            # 4. Baseline Relative Sharpness Gain
            if baseline_image is not None:
                if baseline_image.size != (pred_w, pred_h):
                    aligned_baseline = baseline_image.resize((pred_w, pred_h), resample=Image.BICUBIC)
                else:
                    aligned_baseline = baseline_image

                arr_base = self._to_float32_array(aligned_baseline)
                base_sharpness = calculate_laplacian_sharpness(arr_base)
                base_psnr = calculate_psnr(arr_base, arr_gt)
                base_ssim = calculate_ssim(arr_base, arr_gt) if min(arr_base.shape) >= 7 else None

                result["baseline_sharpness"] = base_sharpness
                result["baseline_psnr"] = base_psnr
                result["baseline_ssim"] = base_ssim

                if base_sharpness > 1e-4:
                    sharpness_gain = round(((sharpness_val - base_sharpness) / base_sharpness) * 100.0, 2)
                else:
                    sharpness_gain = 0.0
                result["sharpness_gain_pct"] = sharpness_gain
            else:
                result["baseline_sharpness"] = None
                result["baseline_psnr"] = None
                result["baseline_ssim"] = None
                result["sharpness_gain_pct"] = None

            # 5. Optional LPIPS Perceptual Metric
            if self.use_lpips and self._lpips_loss_fn is not None:
                try:
                    import torch
                    import torchvision.transforms.functional as TF
                    t_pred = TF.to_tensor(predicted_image.convert("RGB")).unsqueeze(0) * 2.0 - 1.0
                    t_gt = TF.to_tensor(aligned_gt.convert("RGB")).unsqueeze(0) * 2.0 - 1.0
                    with torch.no_grad():
                        lpips_score = float(self._lpips_loss_fn(t_pred, t_gt).item())
                    result["lpips"] = round(lpips_score, 4)
                except Exception as e:
                    result["lpips"] = None
                    result["lpips_status"] = f"Calculation failed: {e}"
            else:
                result["lpips"] = None
                result["lpips_status"] = "Optional LPIPS library not installed in environment"

            result["dimensions"] = [pred_w, pred_h]

        except Exception as exc:
            # Explicit failure representation — never silently convert to zero
            result["status"] = "failed"
            result["error"] = str(exc)
            result["psnr"] = None
            result["ssim"] = None
            result["sharpness"] = None

        return result
