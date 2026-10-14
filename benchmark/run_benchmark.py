#!/usr/bin/env python3
"""CLI driver for running reproducible AI research benchmarks."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.core.config import BenchmarkConfig
from benchmark.core.dataset import BenchmarkDataset
from benchmark.core.splitting import DataSplitter
from benchmark.engine.runner import BenchmarkRunner
from benchmark.engine.aggregator import BenchmarkAggregator
from benchmark.engine.failure_analysis import FailureAnalyzer
from benchmark.engine.visualizer import BenchmarkVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="QGAN Medical Image Super-Resolution Research Benchmarking Engine")
    parser.add_argument("--config", type=Path, default=None, help="Path to benchmark JSON configuration file")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/benchmark"), help="Root directory for benchmark data")
    parser.add_argument("--create-fixtures", action="store_true", help="Generate synthetic test fixtures for development/testing")
    parser.add_argument("--create-split", action="store_true", help="Generate reproducible train/val/test split manifest")
    parser.add_argument("--split-name", type=str, default="test", help="Split to evaluate: test, val, or train")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4, 8], help="Super-resolution magnification scales (e.g., 2 4 8)")
    parser.add_argument("--models", nargs="+", type=str, default=["bicubic", "current_qgan"], help="Models to benchmark")
    parser.add_argument("--modalities", nargs="+", type=str, default=["general", "brain_mri", "chest_xray", "ct_scan"], help="Modalities")
    parser.add_argument("--visualize-samples", type=int, default=2, help="Number of representative side-by-side comparison images to save")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = BenchmarkDataset(args.dataset_dir)

    if args.create_fixtures:
        print("Generating synthetic benchmark fixtures across modalities...")
        fixtures = dataset.generate_synthetic_benchmark_fixtures(samples_per_modality=2)
        print(f"Created {len(fixtures)} synthetic fixtures in {args.dataset_dir}.")

    all_samples = dataset.get_all_samples()
    if not all_samples:
        print(f"No benchmark samples found in {args.dataset_dir}/metadata. Run with --create-fixtures to create test samples.")
        sys.exit(1)

    split_manifest_path = args.dataset_dir / "splits" / "default_split.json"
    if args.create_split or not split_manifest_path.exists():
        print("Generating deterministic train/val/test split manifest...")
        manifest = DataSplitter.split(all_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
        manifest.save(split_manifest_path)
        print(f"Saved split manifest to {split_manifest_path}")

    # Build configuration
    if args.config and args.config.exists():
        config = BenchmarkConfig.load(args.config)
    else:
        config = BenchmarkConfig(
            experiment_name="research_benchmark_gate",
            seed=12345,
            scales=args.scales,
            modalities=args.modalities,
            models=args.models,
            split_name=args.split_name,
            dataset_dir=str(args.dataset_dir),
            split_manifest_path=str(split_manifest_path),
            output_dir="benchmark_runs",
        )

    # Execute benchmark run
    runner = BenchmarkRunner(config, dataset)
    experiment_results = runner.run()

    records = experiment_results.get("records", [])
    if not records:
        print("No evaluation records generated.")
        return

    # Run Aggregations
    aggregation = BenchmarkAggregator.aggregate(records)
    agg_path = runner.output_dir / "aggregated_summary.json"
    import json
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregation, f, indent=2)

    # Run Failure Analysis
    failure_report = FailureAnalyzer.analyze(records)
    fail_path = runner.output_dir / "failure_analysis.json"
    FailureAnalyzer.save_report(failure_report, fail_path)

    # Render Visual Summary
    summary_md = BenchmarkVisualizer.generate_markdown_summary_table(aggregation)
    print("\n" + summary_md)

    # Generate representative visual comparison grids
    if args.visualize_samples > 0:
        print(f"Generating {args.visualize_samples} representative visual comparison grids...")
        from PIL import Image
        sample_ids = list({r["sample_id"] for r in records})[:args.visualize_samples]
        for idx, sid in enumerate(sample_ids):
            meta = dataset.get_sample_by_id(sid)
            if not meta:
                continue
            deg_cfg = runner.config.degradation
            lr_img, gt_img = dataset.get_or_create_degraded_image(meta, deg_cfg)
            bicubic_model = runner.models["bicubic"]
            bicubic_pred, _ = bicubic_model.predict(lr_img, scale=4, modality=meta.modality)
            qgan_model = runner.models.get("current_qgan", bicubic_model)
            qgan_pred, _ = qgan_model.predict(lr_img, scale=4, modality=meta.modality)

            out_img_path = runner.output_dir / f"visual_comparison_{sid}_4x.png"
            BenchmarkVisualizer.create_side_by_side_grid(
                lr_image=lr_img,
                bicubic_image=bicubic_pred,
                model_image=qgan_pred,
                ground_truth_image=gt_img,
                output_path=out_img_path,
            )
            print(f"  Saved comparison grid: {out_img_path}")

    print("\nBENCHMARK RUN COMPLETED SUCCESSFULLY.")
    print(f"Experiment Directory: {runner.output_dir}")


if __name__ == "__main__":
    main()
