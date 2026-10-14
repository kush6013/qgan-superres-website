import os
import sys
import unittest
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark.core.metadata import BenchmarkSampleMetadata
from benchmark.core.degradation import DegradationConfig, ControlledDegradationPipeline
from benchmark.core.splitting import DataSplitter, SplitManifest
from benchmark.core.config import BenchmarkConfig
from benchmark.core.dataset import BenchmarkDataset
from benchmark.adapters.bicubic import BicubicBaselineModel
from benchmark.adapters.current_qgan import CurrentQGANModel
from benchmark.metrics.evaluator import ResearchMetricsEvaluator
from benchmark.engine.runner import BenchmarkRunner
from benchmark.engine.aggregator import BenchmarkAggregator
from benchmark.engine.failure_analysis import FailureAnalyzer


class TestBenchmarkFramework(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metadata_integrity_and_phi_check(self):
        """Step 3: Test metadata schema, serialization, and PHI leak prevention."""
        meta = BenchmarkSampleMetadata(
            sample_id="sub_001_cxr_01",
            modality="chest_xray",
            source_dataset="test_repo",
            original_filename="sub_001_cxr_01_gt.png",
            original_dimensions=(256, 256),
            channel_count=1,
            bit_depth=8,
            original_format="PNG",
            ground_truth_path="/path/to/gt.png",
            patient_id="ANON_SUBJ_42",
        )
        d = meta.to_dict()
        self.assertEqual(d["sample_id"], "sub_001_cxr_01")
        self.assertEqual(d["patient_id"], "ANON_SUBJ_42")

        # Test deserialization
        loaded = BenchmarkSampleMetadata.from_dict(d)
        self.assertEqual(loaded.original_dimensions, (256, 256))

        # Test suspicious PHI detection
        bad_meta = BenchmarkSampleMetadata(
            sample_id="dr_smith_scan",
            modality="chest_xray",
            source_dataset="repo",
            original_filename="Dr. John Smith Chest.png",
            original_dimensions=(256, 256),
            channel_count=1,
            bit_depth=8,
            original_format="PNG",
            ground_truth_path="/path/to/gt.png",
        )
        with self.assertRaises(ValueError):
            bad_meta.to_dict()

    def test_split_reproducibility_and_patient_leakage_safeguard(self):
        """Step 4: Test patient-level grouping and zero overlap across train/val/test."""
        # Create 6 samples across 3 patients (2 slices per patient)
        samples = [
            BenchmarkSampleMetadata(
                sample_id=f"p1_s{i}",
                modality="brain_mri",
                source_dataset="ds",
                original_filename=f"p1_s{i}.png",
                original_dimensions=(256, 256),
                channel_count=1,
                bit_depth=8,
                original_format="PNG",
                ground_truth_path=f"gt_{i}.png",
                patient_id="PAT_001",
            )
            for i in [1, 2]
        ] + [
            BenchmarkSampleMetadata(
                sample_id=f"p2_s{i}",
                modality="brain_mri",
                source_dataset="ds",
                original_filename=f"p2_s{i}.png",
                original_dimensions=(256, 256),
                channel_count=1,
                bit_depth=8,
                original_format="PNG",
                ground_truth_path=f"gt_{i}.png",
                patient_id="PAT_002",
            )
            for i in [1, 2]
        ] + [
            BenchmarkSampleMetadata(
                sample_id=f"p3_s{i}",
                modality="brain_mri",
                source_dataset="ds",
                original_filename=f"p3_s{i}.png",
                original_dimensions=(256, 256),
                channel_count=1,
                bit_depth=8,
                original_format="PNG",
                ground_truth_path=f"gt_{i}.png",
                patient_id="PAT_003",
            )
            for i in [1, 2]
        ]

        split1 = DataSplitter.split(samples, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=123)
        split2 = DataSplitter.split(samples, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=123)

        # Reproducibility check
        self.assertEqual(split1.train_sample_ids, split2.train_sample_ids)
        self.assertEqual(split1.val_sample_ids, split2.val_sample_ids)
        self.assertEqual(split1.test_sample_ids, split2.test_sample_ids)

        # Patient-level grouping check: PAT_001 slices must be strictly in ONE split
        p1_slices = {"p1_s1", "p1_s2"}
        in_train = p1_slices.issubset(set(split1.train_sample_ids))
        in_val = p1_slices.issubset(set(split1.val_sample_ids))
        in_test = p1_slices.issubset(set(split1.test_sample_ids))
        self.assertTrue(in_train or in_val or in_test, "Patient 1 slices split across different partitions!")

        # Zero overlap check
        set_train = set(split1.train_sample_ids)
        set_val = set(split1.val_sample_ids)
        set_test = set(split1.test_sample_ids)
        self.assertEqual(len(set_train.intersection(set_val)), 0)
        self.assertEqual(len(set_train.intersection(set_test)), 0)
        self.assertEqual(len(set_val.intersection(set_test)), 0)

    def test_synthetic_degradation_determinism(self):
        """Step 6: Verify degradation pipeline is strictly deterministic with fixed seed."""
        arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")

        config1 = DegradationConfig(scale=4, downsampling="bicubic", blur_sigma=0.8, noise_type="gaussian", noise_std=5.0, seed=42)
        config2 = DegradationConfig(scale=4, downsampling="bicubic", blur_sigma=0.8, noise_type="gaussian", noise_std=5.0, seed=42)

        lr1 = ControlledDegradationPipeline.degrade(img, config1)
        lr2 = ControlledDegradationPipeline.degrade(img, config2)

        self.assertEqual(lr1.size, (32, 32))
        self.assertEqual(lr2.size, (32, 32))
        # Bitwise identical
        np.testing.assert_array_equal(np.asarray(lr1), np.asarray(lr2))

    def test_bicubic_baseline_and_adapter_interface(self):
        """Steps 7 & 8: Verify model adapters expose consistent prediction interface."""
        img = Image.new("L", (32, 32), color=100)
        bicubic = BicubicBaselineModel()
        out, meta = bicubic.predict(img, scale=4, modality="general")

        self.assertEqual(out.size, (128, 128))
        self.assertIn("inference_ms", meta)
        self.assertEqual(meta["target_scale"], 4)
        self.assertEqual(meta["model_name"], "bicubic")

        # Test current QGAN adapter with missing checkpoint (fallback path)
        qgan_adapter = CurrentQGANModel(checkpoint_dir=self.temp_dir)
        out_qgan, meta_qgan = qgan_adapter.predict(img, scale=4, modality="chest_xray")
        self.assertEqual(out_qgan.size, (128, 128))
        self.assertIn("inference_ms", meta_qgan)
        self.assertTrue(meta_qgan["fallback_used"])

    def test_metrics_evaluator_and_edge_cases(self):
        """Steps 9 & 10: Verify metrics handle constant, blank, small images, and NaNs safely."""
        evaluator = ResearchMetricsEvaluator(use_lpips_if_available=False)

        # 1. Normal comparison
        img_a = Image.new("L", (64, 64), color=100)
        img_b = Image.new("L", (64, 64), color=105)
        res = evaluator.evaluate_sample(img_a, img_b)
        self.assertEqual(res["status"], "success")
        self.assertIsNotNone(res["psnr"])
        self.assertIsNotNone(res["ssim"])
        self.assertIsNotNone(res["sharpness"])

        # 2. Identical images (MSE == 0 -> PSNR 100 dB, SSIM 1.0)
        res_same = evaluator.evaluate_sample(img_a, img_a)
        self.assertEqual(res_same["psnr"], 100.0)
        self.assertEqual(res_same["ssim"], 1.0)

        # 3. Blank/all-zero image
        blank = Image.new("L", (64, 64), color=0)
        res_blank = evaluator.evaluate_sample(blank, blank)
        self.assertEqual(res_blank["status"], "success")
        self.assertEqual(res_blank["sharpness"], 0.0)

        # 4. Tiny image (< 7x7)
        tiny1 = Image.new("L", (4, 4), color=50)
        tiny2 = Image.new("L", (4, 4), color=60)
        res_tiny = evaluator.evaluate_sample(tiny1, tiny2)
        self.assertEqual(res_tiny["status"], "success")
        self.assertIsNone(res_tiny["ssim"])  # Local window cannot be computed
        self.assertIn("ssim_warning", res_tiny)

        # 5. Non-finite array handling (must set status="failed", never silent 0)
        class CorruptImage(Image.Image):
            pass

        bad_arr = np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            evaluator.validate_finite_numerical_range(bad_arr, "test")

    def test_full_benchmark_run_serialization_and_aggregation(self):
        """Steps 11, 14, 17, 18: Test complete end-to-end benchmark execution and artifacts."""
        dataset = BenchmarkDataset(self.temp_dir)
        fixtures = dataset.generate_synthetic_benchmark_fixtures(samples_per_modality=1)
        self.assertEqual(len(fixtures), 4)

        config = BenchmarkConfig(
            experiment_name="test_experiment",
            seed=42,
            scales=[2],
            modalities=["chest_xray", "brain_mri"],
            models=["bicubic", "current_qgan"],
            dataset_dir=str(self.temp_dir),
            split_manifest_path=None,
            output_dir=str(Path(self.temp_dir) / "runs"),
        )

        runner = BenchmarkRunner(config, dataset)
        results = runner.run()

        self.assertIn("records", results)
        self.assertEqual(len(results["records"]), 4)  # 2 modalities x 1 sample x 1 scale x 2 models

        # Check serialization files exist
        json_path = runner.output_dir / "benchmark_results.json"
        csv_path = runner.output_dir / "benchmark_results.csv"
        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())

        # Test Aggregation
        agg = BenchmarkAggregator.aggregate(results["records"])
        self.assertIn("by_model", agg)
        self.assertIn("bicubic", agg["by_model"])

        # Test Failure Analysis
        fail_rep = FailureAnalyzer.analyze(results["records"])
        self.assertIn("worst_performing_by_psnr", fail_rep)
        self.assertIn("largest_gain_over_bicubic", fail_rep)

    def test_unknown_modality_handling(self):
        """Step 19: Unknown modality should fall back safely without crashing."""
        img = Image.new("L", (32, 32), color=120)
        qgan_adapter = CurrentQGANModel(checkpoint_dir=self.temp_dir)
        out, meta = qgan_adapter.predict(img, scale=2, modality="ultrasound_unknown")
        self.assertEqual(out.size, (64, 64))
        self.assertIn("fallback", meta["used_modality"])

    def test_split_without_patient_ids_records_audit_warning(self):
        """Step 4 & 19: Missing patient IDs must trigger sample-level split with explicit audit warning."""
        samples = [
            BenchmarkSampleMetadata(
                sample_id=f"img_{i}",
                modality="general",
                source_dataset="ds",
                original_filename=f"img_{i}.png",
                original_dimensions=(256, 256),
                channel_count=1,
                bit_depth=8,
                original_format="PNG",
                ground_truth_path="gt.png",
                patient_id=None,  # No patient ID available
            )
            for i in range(10)
        ]
        manifest = DataSplitter.split(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        self.assertFalse(manifest.patient_level_grouping)
        self.assertIn("WARNING", manifest.audit_notes)
        self.assertEqual(manifest.counts["train"] + manifest.counts["val"] + manifest.counts["test"], 10)

    def test_visualizer_side_by_side_grid(self):
        """Step 16: Test side-by-side visual comparison grid generator."""
        from benchmark.engine.visualizer import BenchmarkVisualizer
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        out_path = Path(self.temp_dir) / "test_grid.png"
        grid = BenchmarkVisualizer.create_side_by_side_grid(
            lr_image=img,
            bicubic_image=img,
            model_image=img,
            ground_truth_image=img,
            output_path=out_path,
        )
        self.assertTrue(out_path.exists())
        self.assertGreater(grid.width, 64 * 4)


if __name__ == "__main__":
    unittest.main()
