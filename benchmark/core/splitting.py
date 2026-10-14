import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from .metadata import BenchmarkSampleMetadata


@dataclass
class SplitManifest:
    """
    Manifest recording the exact partition of dataset samples across train, val, and test.
    Preserves audit trail to guarantee zero data leakage between development and evaluation.
    """
    manifest_id: str
    seed: int
    ratios: Dict[str, float]
    patient_level_grouping: bool
    train_sample_ids: List[str]
    val_sample_ids: List[str]
    test_sample_ids: List[str]
    counts: Dict[str, int]
    patient_counts: Dict[str, int]
    created_at: str
    audit_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitManifest":
        return cls(**data)

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "SplitManifest":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_split_samples(self, split_name: str) -> List[str]:
        split = split_name.lower()
        if split == "train":
            return self.train_sample_ids
        elif split in ("val", "validation"):
            return self.val_sample_ids
        elif split == "test":
            return self.test_sample_ids
        else:
            raise ValueError(f"Unknown split name: '{split_name}'. Expected train, val, or test.")


class DataSplitter:
    """
    Partitions benchmark samples into train, validation, and test splits with strict leakage safeguards.
    """

    @classmethod
    def split(
        cls,
        samples: List[BenchmarkSampleMetadata],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> SplitManifest:
        """
        Executes a leakage-safe split.
        If patient_id is present across all samples:
          Groups samples by patient_id, shuffles patients deterministically, and assigns all
          images belonging to each patient strictly into the same split.
        If patient_id is missing or incomplete:
          Falls back to sample-level deterministic splitting and generates an explicit audit notice.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        rng = random.Random(seed)

        has_patient_ids = all(s.patient_id is not None and len(s.patient_id.strip()) > 0 for s in samples)

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        patient_counts = {"train": 0, "val": 0, "test": 0}

        if has_patient_ids and len(samples) > 0:
            patient_level = True
            # Group samples by patient
            patient_map: Dict[str, List[str]] = defaultdict(list)
            for s in samples:
                patient_map[s.patient_id].append(s.sample_id)

            unique_patients = sorted(list(patient_map.keys()))
            rng.shuffle(unique_patients)

            n_patients = len(unique_patients)
            n_train_pat = int(round(train_ratio * n_patients))
            n_val_pat = int(round(val_ratio * n_patients))
            # Ensure at least 1 patient in test if we have enough patients
            if n_patients >= 3 and (n_patients - n_train_pat - n_val_pat) < 1:
                n_train_pat = max(1, n_train_pat - 1)

            train_pat = unique_patients[:n_train_pat]
            val_pat = unique_patients[n_train_pat : n_train_pat + n_val_pat]
            test_pat = unique_patients[n_train_pat + n_val_pat :]

            for p in train_pat:
                train_ids.extend(patient_map[p])
            for p in val_pat:
                val_ids.extend(patient_map[p])
            for p in test_pat:
                test_ids.extend(patient_map[p])

            patient_counts = {
                "train": len(train_pat),
                "val": len(val_pat),
                "test": len(test_pat),
            }
            audit_notes = (
                f"Patient-level grouping ENABLED. {n_patients} unique subjects partitioned. "
                "Zero inter-slice or inter-scan cross-contamination across splits."
            )
        else:
            patient_level = False
            sample_ids = sorted([s.sample_id for s in samples])
            rng.shuffle(sample_ids)

            n_total = len(sample_ids)
            n_train = int(round(train_ratio * n_total))
            n_val = int(round(val_ratio * n_total))
            if n_total >= 3 and (n_total - n_train - n_val) < 1:
                n_train = max(1, n_train - 1)

            train_ids = sample_ids[:n_train]
            val_ids = sample_ids[n_train : n_train + n_val]
            test_ids = sample_ids[n_train + n_val :]

            audit_notes = (
                "WARNING: Patient-level grouping DISABLED. Subject identifiers were unavailable in source data. "
                "Partition performed at individual image level. Potential patient-level correlation across splits cannot be ruled out."
            )

        # Sort sample IDs for deterministic order
        train_ids.sort()
        val_ids.sort()
        test_ids.sort()

        # Sanity check: zero overlap
        overlap_train_val = set(train_ids).intersection(val_ids)
        overlap_train_test = set(train_ids).intersection(test_ids)
        overlap_val_test = set(val_ids).intersection(test_ids)
        assert len(overlap_train_val) == 0, f"Leakage detected between train and val: {overlap_train_val}"
        assert len(overlap_train_test) == 0, f"Leakage detected between train and test: {overlap_train_test}"
        assert len(overlap_val_test) == 0, f"Leakage detected between val and test: {overlap_val_test}"

        manifest_id = f"manifest_seed{seed}_{len(samples)}samples"

        return SplitManifest(
            manifest_id=manifest_id,
            seed=seed,
            ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
            patient_level_grouping=patient_level,
            train_sample_ids=train_ids,
            val_sample_ids=val_ids,
            test_sample_ids=test_ids,
            counts={"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
            patient_counts=patient_counts,
            created_at=datetime.now(timezone.utc).isoformat(),
            audit_notes=audit_notes,
        )
