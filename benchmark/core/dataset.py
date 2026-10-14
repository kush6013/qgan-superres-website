from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image, ImageDraw
import numpy as np

from .metadata import BenchmarkSampleMetadata
from .degradation import DegradationConfig, ControlledDegradationPipeline
from .splitting import SplitManifest


class BenchmarkDataset:
    """
    Manages benchmark data directories, sample registry, and ground-truth pairing.
    Maintains clean separation between raw, processed, ground-truth, and degraded assets.
    """

    def __init__(self, root_dir: Path | str = "data/benchmark"):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "raw"
        self.processed_dir = self.root_dir / "processed"
        self.ground_truth_dir = self.root_dir / "ground_truth"
        self.degraded_dir = self.root_dir / "degraded"
        self.splits_dir = self.root_dir / "splits"
        self.metadata_dir = self.root_dir / "metadata"

        # Ensure directory structure exists
        for d in [self.raw_dir, self.processed_dir, self.ground_truth_dir, self.degraded_dir, self.splits_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def register_ground_truth_sample(
        self,
        image: Image.Image,
        sample_id: str,
        modality: str,
        source_dataset: str = "curated_benchmark",
        patient_id: Optional[str] = None,
        bit_depth: int = 8,
        reference_type: str = "synthetic_ground_truth",
        file_format: str = "PNG",
    ) -> BenchmarkSampleMetadata:
        """
        Registers an HR ground-truth image into the benchmark repository.
        Saves image to ground_truth/, computes checksum, and writes metadata JSON.
        """
        gt_filename = f"{sample_id}_gt.{file_format.lower()}"
        gt_path = self.ground_truth_dir / gt_filename
        image.save(gt_path, format=file_format)

        checksum = BenchmarkSampleMetadata.compute_file_checksum(gt_path)
        channels = 1 if image.mode in ("L", "1") else 3

        meta = BenchmarkSampleMetadata(
            sample_id=sample_id,
            modality=modality,
            source_dataset=source_dataset,
            original_filename=gt_filename,
            original_dimensions=image.size,
            channel_count=channels,
            bit_depth=bit_depth,
            original_format=file_format.upper(),
            ground_truth_path=str(gt_path),
            split="test",
            reference_type=reference_type,
            patient_id=patient_id,
            checksum=checksum,
        )

        meta_path = self.metadata_dir / f"{sample_id}.json"
        meta.save(meta_path)
        return meta

    def get_all_samples(self) -> List[BenchmarkSampleMetadata]:
        """Loads all registered sample metadata records."""
        samples = []
        for p in sorted(self.metadata_dir.glob("*.json")):
            try:
                meta = BenchmarkSampleMetadata.load(p)
                samples.append(meta)
            except Exception as e:
                print(f"Warning: Could not load metadata at {p}: {e}")
        return samples

    def get_sample_by_id(self, sample_id: str) -> Optional[BenchmarkSampleMetadata]:
        meta_path = self.metadata_dir / f"{sample_id}.json"
        if not meta_path.exists():
            return None
        return BenchmarkSampleMetadata.load(meta_path)

    def get_or_create_degraded_image(
        self,
        sample: BenchmarkSampleMetadata,
        config: DegradationConfig,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Retrieves or generates the degraded LR image for a given sample and degradation config.
        Returns: (degraded_lr_image, ground_truth_hr_image)
        """
        gt_path = Path(sample.ground_truth_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file missing: {gt_path}")

        gt_img = Image.open(gt_path).convert("RGB" if sample.channel_count == 3 else "L")

        if sample.reference_type == "ground_truth":
            # Mode A: Paired LR already exists on disk
            lr_path = self.degraded_dir / f"{sample.sample_id}_lr_{config.scale}x.png"
            if lr_path.exists():
                lr_img = Image.open(lr_path).convert(gt_img.mode)
                return lr_img, gt_img

        # Mode B: Controlled synthetic degradation from HR
        deg_sub_dir = self.degraded_dir / config.config_id()
        deg_sub_dir.mkdir(parents=True, exist_ok=True)
        cached_lr_path = deg_sub_dir / f"{sample.sample_id}_lr.png"

        if cached_lr_path.exists():
            lr_img = Image.open(cached_lr_path).convert(gt_img.mode)
        else:
            lr_img = ControlledDegradationPipeline.degrade(gt_img, config)
            lr_img.save(cached_lr_path)

        return lr_img, gt_img

    def generate_synthetic_benchmark_fixtures(self, samples_per_modality: int = 2) -> List[BenchmarkSampleMetadata]:
        """
        Creates small deterministic synthetic medical patterns for test suites and baseline validation.
        Modalities: general, brain_mri, chest_xray, ct_scan.
        """
        modalities = ["general", "brain_mri", "chest_xray", "ct_scan"]
        created = []

        for m_idx, modality in enumerate(modalities):
            for i in range(samples_per_modality):
                sample_id = f"fixture_{modality}_{i+1}"
                patient_id = f"PAT_{modality[:3].upper()}_{m_idx+1}"

                # Generate a 256x256 medical-like geometric pattern
                img = Image.new("L", (256, 256), color=15)
                draw = ImageDraw.Draw(img)

                if modality == "brain_mri":
                    # Elliptical skull and inner cerebral structures
                    draw.ellipse([28, 18, 228, 238], outline=180, width=4, fill=40)
                    draw.ellipse([50, 40, 206, 216], outline=120, width=2, fill=60)
                    draw.ellipse([80, 80, 120, 160], outline=210, width=2, fill=90)
                    draw.ellipse([136, 80, 176, 160], outline=210, width=2, fill=90)
                elif modality == "chest_xray":
                    # Thoracic ribcage and lung fields
                    draw.rectangle([20, 20, 236, 236], outline=60, width=2, fill=25)
                    draw.ellipse([40, 50, 110, 200], outline=140, width=3, fill=8)
                    draw.ellipse([146, 50, 216, 200], outline=140, width=3, fill=8)
                    # Spine / mediastinum
                    draw.rectangle([115, 30, 141, 230], fill=160)
                elif modality == "ct_scan":
                    # Body contour with vertebral bone and soft tissue
                    draw.ellipse([25, 25, 231, 231], outline=160, width=3, fill=50)
                    draw.ellipse([110, 170, 146, 206], outline=250, width=4, fill=240)  # Dense bone
                    draw.ellipse([60, 70, 130, 140], outline=90, width=2, fill=75)   # Liver / organ
                else:
                    # General phantom
                    draw.rectangle([30, 30, 226, 226], outline=200, width=3, fill=50)
                    draw.line([30, 30, 226, 226], fill=220, width=2)
                    draw.ellipse([80, 80, 176, 176], fill=120, outline=240, width=3)

                meta = self.register_ground_truth_sample(
                    image=img,
                    sample_id=sample_id,
                    modality=modality,
                    source_dataset="synthetic_benchmark_fixtures",
                    patient_id=patient_id,
                    bit_depth=8,
                    reference_type="synthetic_ground_truth",
                    file_format="PNG",
                )
                created.append(meta)

        return created
