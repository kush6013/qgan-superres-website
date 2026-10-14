"""
NIH ChestX-ray14 dataset integration and processing pipeline for reproducible super-resolution benchmarking.

Handles:
- Acquisition audit and directory organization
- Inventory scanning (dimensions, format, corruption, patient identifiers, hash verification)
- Deterministic patient-level splitting with zero leakage
- Controlled pilot subset preparation
- Preprocessing and synthetic ground-truth registration with reproducible degradation (2x, 4x, 8x)
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark.core.dataset import BenchmarkDataset
from benchmark.core.degradation import DegradationConfig, ControlledDegradationPipeline
from benchmark.core.metadata import BenchmarkSampleMetadata
from benchmark.core.splitting import DataSplitter, SplitManifest


def extract_patient_id_from_filename(filename: str) -> str:
    """
    Extracts the official NIH ChestX-ray14 patient ID from filename.
    Format: <patient_id>_<scan_idx>.png (e.g. '00000001_000.png' -> '00000001').
    Falls back to regex match or base stem if format differs.
    """
    stem = Path(filename).stem
    match = re.match(r"^(\d+)_", stem)
    if match:
        return match.group(1)
    # Generic pattern: patient_123 or pat_123 or subj_123
    match_gen = re.search(r"\b(?:patient|pat|subj|subject)[_-]?([a-zA-Z0-9]+)", stem, re.IGNORECASE)
    if match_gen:
        return match_gen.group(1)
    # Default fallback to stem
    return stem


def compute_file_hash(path: Path) -> str:
    """Computes SHA-256 hash of a file for integrity and deduplication."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_and_inventory(
    source_dir: Path,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Inspects image files in source_dir and generates a comprehensive inventory report.
    Reports: counts, formats, dimensions, grayscale status, corruption, and patient groupings.
    """
    source_dir = Path(source_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    candidate_files = [p for p in source_dir.rglob("*") if p.suffix.lower() in image_extensions]

    records = []
    corrupted_count = 0
    unique_patients = set()
    dimensions_list = []
    channels_set = set()
    format_counts = {}

    for file_path in sorted(candidate_files):
        try:
            with Image.open(file_path) as img:
                w, h = img.size
                mode = img.mode
                fmt = img.format or file_path.suffix.lstrip(".").upper()
                dimensions_list.append((w, h))
                channels_set.add(mode)
                format_counts[fmt] = format_counts.get(fmt, 0) + 1

                patient_id = extract_patient_id_from_filename(file_path.name)
                unique_patients.add(patient_id)
                sha256 = compute_file_hash(file_path)

                records.append({
                    "file_name": file_path.name,
                    "rel_path": str(file_path.relative_to(source_dir)),
                    "file_size_bytes": file_path.stat().st_size,
                    "patient_id": patient_id,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "format": fmt,
                    "sha256": sha256,
                    "corrupted": False,
                })
        except Exception as e:
            corrupted_count += 1
            records.append({
                "file_name": file_path.name,
                "rel_path": str(file_path.relative_to(source_dir)),
                "corrupted": True,
                "error": str(e),
            })

    total_valid = len(dimensions_list)
    min_dim = [min(w for w, h in dimensions_list), min(h for w, h in dimensions_list)] if dimensions_list else [0, 0]
    max_dim = [max(w for w, h in dimensions_list), max(h for w, h in dimensions_list)] if dimensions_list else [0, 0]

    inventory_data = {
        "dataset_name": "NIH ChestX-ray14",
        "modality": "chest_xray",
        "source_directory": str(source_dir.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_files_scanned": len(candidate_files),
            "valid_images": total_valid,
            "corrupted_files": corrupted_count,
            "unique_patients": len(unique_patients),
            "formats": format_counts,
            "color_modes": list(channels_set),
            "min_dimensions": min_dim,
            "max_dimensions": max_dim,
        },
        "records": records,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(inventory_data, f, indent=2)

    return inventory_data


def generate_synthetic_chest_xray(
    patient_id: str,
    scan_idx: int,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None
) -> Image.Image:
    """
    Generates a realistic synthetic chest radiograph for reproducible offline pilot runs.
    Simulates thoracic cage, clavicles, lung fields, cardiac silhouette, and rib contours.
    """
    rng = np.random.default_rng(seed if seed is not None else int(patient_id) * 100 + scan_idx)

    # Base background: thoracic soft tissue
    img = Image.new("L", (width, height), color=40)
    draw = ImageDraw.Draw(img)

    # Thorax cavity ellipse
    draw.ellipse([int(width * 0.08), int(height * 0.05), int(width * 0.92), int(height * 0.95)], fill=15)

    # Left and Right Lung Fields (radiolucent = darker on negative radiograph)
    lung_y1, lung_y2 = int(height * 0.15), int(height * 0.78)
    # Right lung (anatomical right, viewer's left)
    draw.ellipse([int(width * 0.14), lung_y1, int(width * 0.44), lung_y2], fill=190)
    # Left lung (anatomical left, viewer's right)
    draw.ellipse([int(width * 0.56), lung_y1, int(width * 0.86), lung_y2], fill=185)

    # Mediastinum and Spine (dense bone = bright white / high intensity)
    draw.rectangle([int(width * 0.46), int(height * 0.08), int(width * 0.54), int(height * 0.88)], fill=70)

    # Cardiac Silhouette (dense heart muscle, shifted left)
    heart_bbox = [int(width * 0.38), int(height * 0.45), int(width * 0.65), int(height * 0.78)]
    draw.ellipse(heart_bbox, fill=85)

    # Rib cage contours (curved anterior and posterior ribs)
    for i in range(7):
        y_pos = int(height * (0.22 + i * 0.08))
        # Rib pairs
        draw.arc([int(width * 0.10), y_pos - 15, int(width * 0.46), y_pos + 25], start=190, end=350, fill=110, width=4)
        draw.arc([int(width * 0.54), y_pos - 15, int(width * 0.90), y_pos + 25], start=190, end=350, fill=110, width=4)

    # Clavicles
    draw.line([int(width * 0.18), int(height * 0.16), int(width * 0.46), int(height * 0.19)], fill=130, width=6)
    draw.line([int(width * 0.54), int(height * 0.19), int(width * 0.82), int(height * 0.16)], fill=130, width=6)

    # Add Gaussian smoothing to simulate anatomical soft-tissue transitions
    img = img.filter(ImageFilter.GaussianBlur(radius=3.0))

    # Add subtle Poisson/Gaussian radiologic quantum noise
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, 3.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")


class ChestXray14Pipeline:
    """
    Unified manager for NIH ChestX-ray14 dataset workflows.
    Maintains reproducible structure, patient-safe splitting, and benchmark ground truth registration.
    """

    def __init__(self, root_dir: Path | str = "data/benchmark/chest_xray14"):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "raw"
        self.processed_dir = self.root_dir / "processed"
        self.splits_dir = self.root_dir / "splits"
        self.metadata_dir = self.root_dir / "metadata"
        self.ground_truth_dir = self.root_dir / "ground_truth"
        self.degraded_dir = self.root_dir / "degraded"

        for d in [self.raw_dir, self.processed_dir, self.splits_dir, self.metadata_dir, self.ground_truth_dir, self.degraded_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def create_pilot_subset(
        self,
        num_patients: int = 12,
        scans_per_patient: int = 2,
        image_size: int = 256
    ) -> List[Path]:
        """
        Creates a controlled pilot subset with deterministic synthetic radiographs if no raw data exists.
        Guarantees multiple scans per patient to rigorously test patient-level grouping.
        """
        created_paths = []
        for p_idx in range(1, num_patients + 1):
            patient_id = f"{p_idx:08d}"
            for s_idx in range(scans_per_patient):
                filename = f"{patient_id}_{s_idx:03d}.png"
                target_path = self.raw_dir / filename
                if not target_path.exists():
                    img = generate_synthetic_chest_xray(
                        patient_id=patient_id,
                        scan_idx=s_idx,
                        width=image_size,
                        height=image_size,
                        seed=p_idx * 100 + s_idx
                    )
                    img.save(target_path, format="PNG")
                created_paths.append(target_path)
        return created_paths

    def generate_inventory(self) -> Dict[str, Any]:
        """Scans raw/ and saves chest_xray14_inventory.json."""
        inv_path = self.root_dir / "chest_xray14_inventory.json"
        return scan_and_inventory(self.raw_dir, inv_path)

    def create_patient_safe_split(
        self,
        inventory: Dict[str, Any],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> SplitManifest:
        """
        Builds a patient-level split ensuring 0% patient overlap across train/val/test.
        """
        samples: List[BenchmarkSampleMetadata] = []
        for r in inventory.get("records", []):
            if r.get("corrupted", False):
                continue
            sample_id = Path(r["file_name"]).stem
            meta = BenchmarkSampleMetadata(
                sample_id=sample_id,
                modality="chest_xray",
                source_dataset="NIH_ChestX-ray14",
                original_filename=r["file_name"],
                original_dimensions=(r["width"], r["height"]),
                channel_count=1 if r["mode"] == "L" else 3,
                bit_depth=8,
                original_format=r["format"],
                ground_truth_path=str(self.raw_dir / r["file_name"]),
                split="unassigned",
                patient_id=r["patient_id"],
                checksum=r["sha256"],
            )
            samples.append(meta)

        manifest = DataSplitter.split(
            samples=samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        manifest_path = self.splits_dir / "chest_xray14_patient_split.json"
        manifest.save(manifest_path)
        return manifest

    def register_benchmark_samples(
        self,
        manifest: SplitManifest,
        scales: List[int] = [2, 4, 8],
        degradation_pipeline: Optional[ControlledDegradationPipeline] = None
    ) -> List[BenchmarkSampleMetadata]:
        """
        Ingests the test partition into standardized ground-truth benchmark pairs with controlled degradations.
        """
        if degradation_pipeline is None:
            pipeline = ControlledDegradationPipeline()
        else:
            pipeline = degradation_pipeline

        registered = []
        for sample_id in manifest.test_sample_ids:
            raw_path = self.raw_dir / f"{sample_id}.png"
            if not raw_path.exists():
                candidates = list(self.raw_dir.glob(f"{sample_id}.*"))
                if candidates:
                    raw_path = candidates[0]
                else:
                    continue

            with Image.open(raw_path) as img:
                img_copy = img.copy()

            # Save to ground_truth_dir
            gt_path = self.ground_truth_dir / f"{sample_id}_gt.png"
            img_copy.save(gt_path, format="PNG")

            checksum = BenchmarkSampleMetadata.compute_file_checksum(gt_path)
            patient_id = extract_patient_id_from_filename(raw_path.name)

            meta = BenchmarkSampleMetadata(
                sample_id=sample_id,
                modality="chest_xray",
                source_dataset="NIH_ChestX-ray14",
                original_filename=raw_path.name,
                original_dimensions=img_copy.size,
                channel_count=1 if img_copy.mode == "L" else 3,
                bit_depth=8,
                original_format="PNG",
                ground_truth_path=str(gt_path),
                split="test",
                reference_type="synthetic_ground_truth",
                patient_id=patient_id,
                checksum=checksum,
            )

            # Pre-cache deterministic degradations for standard benchmark evaluation scales
            for s in scales:
                deg_config = DegradationConfig(scale=s)
                deg_dir = self.degraded_dir / deg_config.config_id()
                deg_dir.mkdir(parents=True, exist_ok=True)
                deg_path = deg_dir / f"{sample_id}_lr.png"
                if not deg_path.exists():
                    deg_img = pipeline.degrade(img_copy, deg_config)
                    deg_img.save(deg_path, format="PNG")

            meta_file = self.metadata_dir / f"{sample_id}.json"
            meta.save(meta_file)
            registered.append(meta)

        return registered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 Pipeline & Ingestion CLI")
    parser.add_argument("--root-dir", type=Path, default=Path("data/benchmark/chest_xray14"), help="Dataset root directory")
    parser.add_argument("--create-pilot", action="store_true", help="Generate controlled synthetic pilot dataset")
    parser.add_argument("--inventory", action="store_true", help="Generate inventory JSON")
    parser.add_argument("--split", action="store_true", help="Generate patient-safe train/val/test split")
    parser.add_argument("--register-benchmark", action="store_true", help="Generate benchmark ground-truth and degraded pairs")
    parser.add_argument("--num-patients", type=int, default=12, help="Number of patients for pilot subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    pipeline = ChestXray14Pipeline(root_dir=args.root_dir)

    if args.create_pilot:
        print(f"Creating pilot subset with {args.num_patients} patients in {pipeline.raw_dir}...")
        created = pipeline.create_pilot_subset(num_patients=args.num_patients)
        print(f"Created {len(created)} pilot images.")

    if args.inventory or args.create_pilot:
        print(f"Generating inventory for {pipeline.raw_dir}...")
        inv = pipeline.generate_inventory()
        print(f"Inventory saved. Total images: {inv['summary']['valid_images']}, Unique patients: {inv['summary']['unique_patients']}")

    if args.split or args.create_pilot:
        print("Generating patient-safe split manifest...")
        inv = pipeline.generate_inventory()
        split = pipeline.create_patient_safe_split(inv, seed=args.seed)
        print(f"Split created: Train={len(split.train_sample_ids)}, Val={len(split.val_sample_ids)}, Test={len(split.test_sample_ids)}")

    if args.register_benchmark or args.create_pilot:
        print("Registering ground-truth and degraded benchmark pairs...")
        manifest_path = pipeline.splits_dir / "chest_xray14_patient_split.json"
        manifest = SplitManifest.load(manifest_path)
        samples = pipeline.register_benchmark_samples(manifest, scales=[2, 4, 8])
        print(f"Registered {len(samples)} test samples with 2x, 4x, and 8x degradations.")


if __name__ == "__main__":
    main()
