import statistics
from typing import List, Dict, Any, Optional
from collections import defaultdict


class BenchmarkAggregator:
    """
    Computes rigorous statistical aggregations across modalities, scales, and model architectures.
    Preserves paired comparisons and sample size counts (N) for downstream significance testing.
    """

    @staticmethod
    def _compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
        valid = [v for v in values if v is not None and isinstance(v, (int, float))]
        if not valid:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
            }
        n = len(valid)
        mean_val = round(statistics.mean(valid), 3)
        med_val = round(statistics.median(valid), 3)
        std_val = round(statistics.stdev(valid), 3) if n > 1 else 0.0
        return {
            "count": n,
            "mean": mean_val,
            "median": med_val,
            "std": std_val,
            "min": round(min(valid), 3),
            "max": round(max(valid), 3),
        }

    @classmethod
    def aggregate(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates raw benchmark records:
          1. Overall by Model
          2. By Model x Scale
          3. By Model x Modality
          4. By Model x Scale x Modality
        """
        by_model = defaultdict(list)
        by_model_scale = defaultdict(list)
        by_model_modality = defaultdict(list)
        by_matrix = defaultdict(list)

        for r in records:
            if r.get("status") != "success":
                continue
            m = r["model_name"]
            s = r["scale"]
            mod = r["modality"]

            by_model[m].append(r)
            by_model_scale[(m, s)].append(r)
            by_model_modality[(m, mod)].append(r)
            by_matrix[(m, s, mod)].append(r)

        def summarize_group(group_records: List[Dict[str, Any]]) -> Dict[str, Any]:
            psnrs = [r["psnr"] for r in group_records]
            ssims = [r["ssim"] for r in group_records]
            sharpnesses = [r["sharpness"] for r in group_records]
            times_ms = [r["inference_ms"] for r in group_records]
            delta_psnrs = [r.get("delta_psnr_over_bicubic") for r in group_records if r.get("delta_psnr_over_bicubic") is not None]
            delta_ssims = [r.get("delta_ssim_over_bicubic") for r in group_records if r.get("delta_ssim_over_bicubic") is not None]

            stats = {
                "sample_count_N": len(group_records),
                "psnr": cls._compute_stats(psnrs),
                "ssim": cls._compute_stats(ssims),
                "sharpness": cls._compute_stats(sharpnesses),
                "inference_ms": cls._compute_stats(times_ms),
                "delta_psnr_over_bicubic": cls._compute_stats(delta_psnrs),
                "delta_ssim_over_bicubic": cls._compute_stats(delta_ssims),
            }
            if stats["sample_count_N"] < 5:
                stats["statistical_power_warning"] = (
                    f"Sample size N={stats['sample_count_N']} is small. Standard deviations and "
                    "p-values must be interpreted with caution."
                )
            return stats

        summary: Dict[str, Any] = {
            "by_model": {},
            "by_model_scale": {},
            "by_model_modality": {},
            "by_matrix": {},
        }

        for model_name, recs in by_model.items():
            summary["by_model"][model_name] = summarize_group(recs)

        for (model_name, scale), recs in by_model_scale.items():
            key = f"{model_name}_{scale}x"
            summary["by_model_scale"][key] = summarize_group(recs)

        for (model_name, modality), recs in by_model_modality.items():
            key = f"{model_name}_{modality}"
            summary["by_model_modality"][key] = summarize_group(recs)

        for (model_name, scale, modality), recs in by_matrix.items():
            key = f"{model_name}_{scale}x_{modality}"
            summary["by_matrix"][key] = summarize_group(recs)

        return summary
