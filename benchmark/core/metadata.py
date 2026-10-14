import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


SUSPICIOUS_PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(mr|mrs|ms|dr)\.?\s+[a-z]+", re.IGNORECASE),  # Direct names
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
]


@dataclass
class BenchmarkSampleMetadata:
    """
    Standard machine-readable metadata schema for a benchmark medical image sample.
    Adheres strictly to HIPAA Safe Harbor guidelines (zero raw PHI in public reports or filenames).
    """
    sample_id: str
    modality: str  # "general", "brain_mri", "chest_xray", "ct_scan"
    source_dataset: str
    original_filename: str  # Safe, sanitized basename
    original_dimensions: Tuple[int, int]  # (width, height)
    channel_count: int  # 1 for grayscale, 3 for RGB
    bit_depth: int  # 8, 12, 16
    original_format: str  # "PNG", "JPEG", "DICOM", "NIFTI"
    ground_truth_path: str
    split: str = "test"  # "train", "val", "test"
    reference_type: str = "synthetic_ground_truth"  # "ground_truth" or "synthetic_ground_truth"
    patient_id: Optional[str] = None  # De-identified/anonymized subject ID (e.g., "SUBJ_0042")
    degradation_config_id: Optional[str] = None
    checksum: Optional[str] = None  # SHA-256 hex digest of the ground-truth image

    def validate_phi_hygiene(self) -> bool:
        """
        Validates that no obvious Protected Health Information (PHI) is leaked in public fields.
        """
        fields_to_check = [
            self.sample_id,
            self.original_filename,
            self.patient_id or "",
            self.source_dataset,
        ]
        for field in fields_to_check:
            for pattern in SUSPICIOUS_PHI_PATTERNS:
                if pattern.search(field):
                    raise ValueError(f"Potential PHI detected in metadata field: '{field}'")
        return True

    @classmethod
    def compute_file_checksum(cls, file_path: Path) -> str:
        """Computes deterministic SHA-256 checksum for a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metadata to a standard dictionary."""
        self.validate_phi_hygiene()
        data = asdict(self)
        data["original_dimensions"] = list(self.original_dimensions)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSampleMetadata":
        """Deserializes metadata from a dictionary."""
        dims = data.get("original_dimensions", [256, 256])
        if isinstance(dims, list):
            dims = (int(dims[0]), int(dims[1]))
        return cls(
            sample_id=data["sample_id"],
            modality=data["modality"],
            source_dataset=data["source_dataset"],
            original_filename=data["original_filename"],
            original_dimensions=dims,
            channel_count=data.get("channel_count", 1),
            bit_depth=data.get("bit_depth", 8),
            original_format=data.get("original_format", "PNG"),
            ground_truth_path=data["ground_truth_path"],
            split=data.get("split", "test"),
            reference_type=data.get("reference_type", "synthetic_ground_truth"),
            patient_id=data.get("patient_id"),
            degradation_config_id=data.get("degradation_config_id"),
            checksum=data.get("checksum"),
        )

    def save(self, json_path: Path) -> None:
        """Saves metadata record as a JSON file."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, json_path: Path) -> "BenchmarkSampleMetadata":
        """Loads metadata record from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
