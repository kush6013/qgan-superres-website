import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from .degradation import DegradationConfig


@dataclass
class BenchmarkConfig:
    """
    Standard experiment configuration for reproducible benchmarks.
    """
    experiment_name: str = "baseline_v1"
    seed: int = 12345
    scales: List[int] = field(default_factory=lambda: [2, 4, 8])
    modalities: List[str] = field(default_factory=lambda: ["general", "brain_mri", "chest_xray", "ct_scan"])
    models: List[str] = field(default_factory=lambda: ["bicubic", "current_qgan"])
    split_name: str = "test"
    dataset_dir: str = "data/benchmark"
    split_manifest_path: Optional[str] = "data/benchmark/splits/default_split.json"
    output_dir: str = "benchmark_runs"
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    description: str = "Standard multi-modality multi-scale benchmark run"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["degradation"] = self.degradation.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        deg_data = data.get("degradation", {})
        if isinstance(deg_data, dict):
            deg = DegradationConfig.from_dict(deg_data)
        else:
            deg = DegradationConfig()

        return cls(
            experiment_name=data.get("experiment_name", "baseline_v1"),
            seed=int(data.get("seed", 12345)),
            scales=list(data.get("scales", [2, 4, 8])),
            modalities=list(data.get("modalities", ["general", "brain_mri", "chest_xray", "ct_scan"])),
            models=list(data.get("models", ["bicubic", "current_qgan"])),
            split_name=str(data.get("split_name", "test")),
            dataset_dir=str(data.get("dataset_dir", "data/benchmark")),
            split_manifest_path=data.get("split_manifest_path"),
            output_dir=str(data.get("output_dir", "benchmark_runs")),
            degradation=deg,
            description=str(data.get("description", "")),
        )

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "BenchmarkConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
