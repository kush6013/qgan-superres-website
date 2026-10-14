import json
from pathlib import Path
from typing import List, Dict, Any


class FailureAnalyzer:
    """
    Identifies edge cases, outlier failures, and differential degradation patterns.
    Ensures research integrity by explicitly highlighting cases where deep models
    perform worse than baseline bicubic interpolation.
    """

    @classmethod
    def analyze(cls, records: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        """
        Analyzes records to extract:
          - Best performing samples by PSNR/SSIM
          - Worst performing samples
          - Largest gain over bicubic baseline
          - Largest degradation (cases where model PSNR < bicubic PSNR)
        """
        valid_records = [r for r in records if r.get("status") == "success" and r.get("psnr") is not None]
        model_records = [r for r in valid_records if r.get("model_name") != "bicubic"]

        # Sort by PSNR
        sorted_by_psnr = sorted(valid_records, key=lambda x: x["psnr"])
        worst_psnr = sorted_by_psnr[:top_k]
        best_psnr = sorted_by_psnr[-top_k:][::-1]

        # Sort by Delta PSNR over Bicubic
        degradations = []
        gains = []

        if model_records:
            sorted_by_delta = sorted(model_records, key=lambda x: x.get("delta_psnr_over_bicubic", 0.0))
            # Negative delta represents cases where the model performed WORSE than simple bicubic
            degradations = [r for r in sorted_by_delta if r.get("delta_psnr_over_bicubic", 0.0) < 0.0][:top_k]
            gains = sorted_by_delta[-top_k:][::-1]

        analysis = {
            "total_evaluated_records": len(valid_records),
            "worst_performing_by_psnr": [
                {
                    "sample_id": r["sample_id"],
                    "model": r["model_name"],
                    "scale": r["scale"],
                    "modality": r["modality"],
                    "psnr": r["psnr"],
                    "ssim": r["ssim"],
                }
                for r in worst_psnr
            ],
            "best_performing_by_psnr": [
                {
                    "sample_id": r["sample_id"],
                    "model": r["model_name"],
                    "scale": r["scale"],
                    "modality": r["modality"],
                    "psnr": r["psnr"],
                    "ssim": r["ssim"],
                }
                for r in best_psnr
            ],
            "largest_gain_over_bicubic": [
                {
                    "sample_id": r["sample_id"],
                    "model": r["model_name"],
                    "scale": r["scale"],
                    "modality": r["modality"],
                    "delta_psnr": r.get("delta_psnr_over_bicubic"),
                    "delta_ssim": r.get("delta_ssim_over_bicubic"),
                }
                for r in gains
            ],
            "largest_degradation_relative_to_bicubic": [
                {
                    "sample_id": r["sample_id"],
                    "model": r["model_name"],
                    "scale": r["scale"],
                    "modality": r["modality"],
                    "delta_psnr": r.get("delta_psnr_over_bicubic"),
                    "reason": "Model reconstruction fell below standard interpolation baseline.",
                }
                for r in degradations
            ],
        }

        return analysis

    @classmethod
    def save_report(cls, analysis: Dict[str, Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
