"""Create the directory layout required for the medical classifier dataset.

This script creates:
  data/medical/train/medical
  data/medical/train/nonmedical
  data/medical/val/medical
  data/medical/val/nonmedical

Usage:
  ./venv/bin/python setup_medical_dataset.py

Optional flags:
  --no-val        Skip validation folder creation.
  --root ROOT     Use a different root directory than data/medical.
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create medical classifier dataset directories")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/medical"),
        help="Dataset root directory (default: data/medical)",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Do not create validation directories",
    )
    return parser


def create_dirs(root: Path, no_val: bool = False) -> None:
    train_medical = root / "train" / "medical"
    train_nonmedical = root / "train" / "nonmedical"
    train_medical.mkdir(parents=True, exist_ok=True)
    train_nonmedical.mkdir(parents=True, exist_ok=True)
    print(f"Created: {train_medical}")
    print(f"Created: {train_nonmedical}")

    if not no_val:
        val_medical = root / "val" / "medical"
        val_nonmedical = root / "val" / "nonmedical"
        val_medical.mkdir(parents=True, exist_ok=True)
        val_nonmedical.mkdir(parents=True, exist_ok=True)
        print(f"Created: {val_medical}")
        print(f"Created: {val_nonmedical}")

    print("\nDataset structure is ready. Add medical scan images under 'medical' and non-medical images under 'nonmedical'.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    create_dirs(args.root, args.no_val)


if __name__ == "__main__":
    main()
