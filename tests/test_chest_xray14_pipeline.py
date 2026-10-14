"""
Automated test suite verifying the NIH ChestX-ray14 dataset workflow, inventory parser,
zero-leakage patient-level partitioning, and baseline benchmark integration.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from benchmark.core.metadata import BenchmarkSampleMetadata
from benchmark.core.degradation import DegradationConfig, ControlledDegradationPipeline
from benchmark.core.splitting import DataSplitter, SplitManifest
from benchmark.datasets.chest_xray14 import (
    ChestXray14Pipeline,
    extract_patient_id_from_filename,
    scan_and_inventory,
    generate_synthetic_chest_xray,
)


class TestChestXray14Workflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_cxr14_"))
        self.pipeline = ChestXray14Pipeline(root_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_patient_id_extraction(self):
        """Verify extraction of official NIH ChestX-ray14 patient identifiers from filenames."""
        self.assertEqual(extract_patient_id_from_filename("00000001_000.png"), "00000001")
        self.assertEqual(extract_patient_id_from_filename("00004523_012.png"), "00004523")
        self.assertEqual(extract_patient_id_from_filename("patient_9876_scan.jpg"), "9876")
        self.assertEqual(extract_patient_id_from_filename("random_scan.png"), "random_scan")

    def test_02_synthetic_xray_generation_determinism(self):
        """Verify that synthetic chest radiograph generation is reproducible given patient/scan index."""
        img1 = generate_synthetic_chest_xray(patient_id="00000042", scan_idx=0, width=128, height=128, seed=42)
        img2 = generate_synthetic_chest_xray(patient_id="00000042", scan_idx=0, width=128, height=128, seed=42)
        self.assertEqual(img1.size, (128, 128))
        self.assertEqual(img1.tobytes(), img2.tobytes())

    def test_03_inventory_scanning_and_corruption_handling(self):
        """Verify that scan_and_inventory correctly reports counts, dimensions, and corrupt files."""
        # Create 2 valid images and 1 corrupted file
        raw_dir = self.pipeline.raw_dir
        img1 = generate_synthetic_chest_xray(patient_id="00000001", scan_idx=0, width=64, height=64)
        img1.save(raw_dir / "00000001_000.png")
        img2 = generate_synthetic_chest_xray(patient_id="00000001", scan_idx=1, width=64, height=64)
        img2.save(raw_dir / "00000001_001.png")

        corrupt_file = raw_dir / "00000002_000.png"
        corrupt_file.write_bytes(b"NOT_A_VALID_IMAGE_HEADER")

        inv_path = self.test_dir / "inv.json"
        inv = scan_and_inventory(raw_dir, inv_path)

        self.assertEqual(inv["summary"]["total_files_scanned"], 3)
        self.assertEqual(inv["summary"]["valid_images"], 2)
        self.assertEqual(inv["summary"]["corrupted_files"], 1)
        self.assertEqual(inv["summary"]["unique_patients"], 1)
        self.assertTrue(inv_path.exists())

    def test_04_patient_level_zero_leakage_split(self):
        """Verify strictly zero patient overlap between train, val, and test splits."""
        self.pipeline.create_pilot_subset(num_patients=10, scans_per_patient=2, image_size=64)
        inv = self.pipeline.generate_inventory()
        manifest = self.pipeline.create_patient_safe_split(inv, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

        # Extract patient IDs in each partition
        train_patients = {extract_patient_id_from_filename(s) for s in manifest.train_sample_ids}
        val_patients = {extract_patient_id_from_filename(s) for s in manifest.val_sample_ids}
        test_patients = {extract_patient_id_from_filename(s) for s in manifest.test_sample_ids}

        # Verify disjoint sets
        self.assertEqual(train_patients.intersection(val_patients), set(), "Data leakage: Train and Val share patients!")
        self.assertEqual(train_patients.intersection(test_patients), set(), "Data leakage: Train and Test share patients!")
        self.assertEqual(val_patients.intersection(test_patients), set(), "Data leakage: Val and Test share patients!")

        # Verify all samples accounted for
        total_samples = len(manifest.train_sample_ids) + len(manifest.val_sample_ids) + len(manifest.test_sample_ids)
        self.assertEqual(total_samples, 20)

    def test_05_ground_truth_and_degradation_registration(self):
        """Verify test set ingestion and deterministic degradation caching."""
        self.pipeline.create_pilot_subset(num_patients=6, scans_per_patient=2, image_size=64)
        inv = self.pipeline.generate_inventory()
        manifest = self.pipeline.create_patient_safe_split(inv, seed=42)

        registered = self.pipeline.register_benchmark_samples(manifest, scales=[2, 4])
        self.assertGreater(len(registered), 0)

        for meta in registered:
            self.assertEqual(meta.modality, "chest_xray")
            self.assertEqual(meta.split, "test")
            self.assertTrue(Path(meta.ground_truth_path).exists())
            self.assertTrue(meta.validate_phi_hygiene())

            # Check cached degradations
            for s in [2, 4]:
                cfg = DegradationConfig(scale=s)
                lr_path = self.pipeline.degraded_dir / cfg.config_id() / f"{meta.sample_id}_lr.png"
                self.assertTrue(lr_path.exists(), f"Missing cached LR image: {lr_path}")


if __name__ == "__main__":
    unittest.main()
