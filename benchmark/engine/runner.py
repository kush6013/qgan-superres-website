import csv
import json
import time
import uuid
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

from benchmark.core.config import BenchmarkConfig
from benchmark.core.dataset import BenchmarkDataset
from benchmark.core.splitting import SplitManifest
from benchmark.core.degradation import DegradationConfig
from benchmark.adapters.base import BenchmarkModel
from benchmark.adapters.bicubic import BicubicBaselineModel
from benchmark.adapters.current_qgan import CurrentQGANModel
from benchmark.metrics.evaluator import ResearchMetricsEvaluator


class BenchmarkRunner:
    """
    Orchestrates reproducible evaluation experiments across models, scales, and modalities.
    Collects sample-level telemetry and metric records, then serializes structured reports.
    """

    def __init__(self, config: BenchmarkConfig, dataset: Optional[BenchmarkDataset] = None):
        self.config = config
        self.dataset = dataset or BenchmarkDataset(config.dataset_dir)
        self.evaluator = ResearchMetricsEvaluator()
        self.experiment_id = f"{config.experiment_name}_{int(time.time())}"
        self.output_dir = Path(config.output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model adapters
        self.models: Dict[str, BenchmarkModel] = {}
        for m_name in config.models:
            if m_name.lower() == "bicubic":
                self.models["bicubic"] = BicubicBaselineModel()
            elif m_name.lower() in ("current_qgan", "current_model", "qgan"):
                self.models["current_qgan"] = CurrentQGANModel()
            else:
                raise ValueError(f"Unsupported model adapter name: '{m_name}'")

        # Mandatory bicubic baseline model must always be present for differential comparison
        if "bicubic" not in self.models:
            self.models["bicubic"] = BicubicBaselineModel()

    def _get_git_commit(self) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return "unknown_or_unversioned"

    def _get_system_telemetry(self) -> Dict[str, Any]:
        return {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "os": f"{platform.system()} {platform.release()}",
            "git_commit": self._get_git_commit(),
        }

    def run(self) -> Dict[str, Any]:
        """
        Executes the benchmark matrix.
        Returns full experiment dictionary including manifest, metadata, and results list.
        """
        all_samples = self.dataset.get_all_samples()

        # Filter by split if manifest exists
        if self.config.split_manifest_path and Path(self.config.split_manifest_path).exists():
            manifest = SplitManifest.load(Path(self.config.split_manifest_path))
            target_ids = set(manifest.get_split_samples(self.config.split_name))
            eval_samples = [s for s in all_samples if s.sample_id in target_ids]
        else:
            eval_samples = all_samples

        # Filter by configured modalities
        eval_samples = [s for s in eval_samples if s.modality.lower() in [m.lower() for m in self.config.modalities]]

        records: List[Dict[str, Any]] = []

        print(f"--- Launching Benchmark Experiment: {self.experiment_id} ---")
        print(f"Total Samples to Evaluate: {len(eval_samples)} across modalities: {self.config.modalities}")
        print(f"Scales: {self.config.scales} | Models: {list(self.models.keys())}")

        bicubic_model = self.models["bicubic"]

        for scale in self.config.scales:
            deg_config = DegradationConfig(
                scale=scale,
                downsampling=self.config.degradation.downsampling,
                blur_sigma=self.config.degradation.blur_sigma,
                noise_type=self.config.degradation.noise_type,
                noise_std=self.config.degradation.noise_std,
                jpeg_quality=self.config.degradation.jpeg_quality,
                seed=self.config.seed,
            )

            for sample in eval_samples:
                # 1. Obtain degraded input LR and ground-truth HR
                lr_img, gt_img = self.dataset.get_or_create_degraded_image(sample, deg_config)

                # 2. Run bicubic baseline first as anchor
                bicubic_pred, bicubic_meta = bicubic_model.predict(lr_img, scale=scale, modality=sample.modality)

                # 3. Evaluate each requested model
                for model_key, model_adapter in self.models.items():
                    if model_key == "bicubic":
                        pred_img = bicubic_pred
                        exec_meta = bicubic_meta
                    else:
                        pred_img, exec_meta = model_adapter.predict(lr_img, scale=scale, modality=sample.modality)

                    # 4. Compute metrics against ground truth
                    metric_data = self.evaluator.evaluate_sample(
                        predicted_image=pred_img,
                        ground_truth_image=gt_img,
                        baseline_image=bicubic_pred,
                        reference_type=sample.reference_type,
                    )

                    record = {
                        "experiment_id": self.experiment_id,
                        "sample_id": sample.sample_id,
                        "patient_id": sample.patient_id,
                        "model_name": model_adapter.name,
                        "modality": sample.modality,
                        "scale": scale,
                        "reference_type": sample.reference_type,
                        "psnr": metric_data.get("psnr"),
                        "ssim": metric_data.get("ssim"),
                        "lpips": metric_data.get("lpips"),
                        "sharpness": metric_data.get("sharpness"),
                        "baseline_psnr": metric_data.get("baseline_psnr"),
                        "baseline_ssim": metric_data.get("baseline_ssim"),
                        "sharpness_gain_pct": metric_data.get("sharpness_gain_pct"),
                        "inference_ms": exec_meta.get("inference_ms"),
                        "peak_memory_mb": exec_meta.get("peak_memory_mb"),
                        "dimensions": metric_data.get("dimensions"),
                        "checkpoint_loaded": exec_meta.get("checkpoint_loaded", False),
                        "fallback_used": exec_meta.get("fallback_used", False),
                        "degradation_config_id": deg_config.config_id(),
                        "status": metric_data.get("status"),
                        "error": metric_data.get("error"),
                    }

                    # Compute paired delta over bicubic if not bicubic itself
                    if model_key != "bicubic" and record["psnr"] is not None and record["baseline_psnr"] is not None:
                        record["delta_psnr_over_bicubic"] = round(record["psnr"] - record["baseline_psnr"], 3)
                    else:
                        record["delta_psnr_over_bicubic"] = 0.0

                    if model_key != "bicubic" and record["ssim"] is not None and record["baseline_ssim"] is not None:
                        record["delta_ssim_over_bicubic"] = round(record["ssim"] - record["baseline_ssim"], 4)
                    else:
                        record["delta_ssim_over_bicubic"] = 0.0

                    records.append(record)

        # Save structured results
        full_experiment = {
            "experiment_id": self.experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
            "system_telemetry": self._get_system_telemetry(),
            "total_records": len(records),
            "records": records,
        }

        self._save_results(full_experiment, records)
        return full_experiment

    def _save_results(self, full_experiment: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        """Serializes results to JSON and CSV formats."""
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_experiment, f, indent=2)

        csv_path = self.output_dir / "benchmark_results.csv"
        if records:
            fieldnames = list(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    row = dict(r)
                    if isinstance(row.get("dimensions"), list):
                        row["dimensions"] = "x".join(map(str, row["dimensions"]))
                    writer.writerow(row)

        print(f"Benchmark results successfully serialized to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
