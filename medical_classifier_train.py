"""Train a dataset-based medical scan classifier.

Expected dataset layout:
  data/medical/train/medical/
  data/medical/train/nonmedical/
  data/medical/val/medical/  (optional)
  data/medical/val/nonmedical/  (optional)

Usage:
  ./venv/bin/python medical_classifier_train.py --train-dir data/medical/train --save-path models/medical_classifier.pt --epochs 10
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.medical_classifier import BinaryResNet18, save_classifier


class MedicalScanDataset(Dataset):
    def __init__(
        self,
        positive_dir: Path,
        negative_dir: Path,
        image_size: int = 224,
    ):
        self.samples = []
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        for path in sorted(Path(positive_dir).rglob("*.png")) + sorted(Path(positive_dir).rglob("*.jpg")) + sorted(Path(positive_dir).rglob("*.jpeg")):
            self.samples.append((path, 1))
        for path in sorted(Path(negative_dir).rglob("*.png")) + sorted(Path(negative_dir).rglob("*.jpg")) + sorted(Path(negative_dir).rglob("*.jpeg")):
            self.samples.append((path, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor([label], dtype=torch.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a dataset-based medical image classifier")
    parser.add_argument("--train-dir", type=Path, default=Path("data/medical/train"), help="Training data root directory")
    parser.add_argument("--val-dir", type=Path, default=None, help="Validation data root directory")
    parser.add_argument("--save-path", type=Path, default=Path("models/medical_classifier.pt"), help="Path to save the classifier weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    return parser


def get_dataset(root_dir: Path, image_size: int, train: bool = True):
    positive_dir = root_dir / "medical"
    negative_dir = root_dir / "nonmedical"
    if not positive_dir.exists() or not negative_dir.exists():
        raise SystemExit(
            f"Missing dataset folders. Expected '{positive_dir}' and '{negative_dir}'."
        )
    dataset = MedicalScanDataset(positive_dir, negative_dir, image_size=image_size)
    if len(dataset) == 0:
        raise SystemExit(
            f"No images found in '{positive_dir}' and '{negative_dir}'. "
            "Add PNG/JPG/JPEG images to these folders before training."
        )
    if train:
        return dataset
    return dataset


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).squeeze(1)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds.float() == targets).sum().item()
            total += inputs.size(0)
    return total_loss / total if total else 0.0, correct / total if total else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = args.train_dir
    if not train_dir.exists():
        raise SystemExit(f"Training directory not found: {train_dir}")

    train_dataset = get_dataset(train_dir, args.image_size, train=True)
    if len(train_dataset) == 0:
        raise SystemExit(
            f"Training dataset is empty. Put images in '{train_dir / 'medical'}' and '{train_dir / 'nonmedical'}'."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = get_dataset(args.val_dir, args.image_size, train=False)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            print(f"Warning: validation dataset is empty. Skipping validation for '{args.val_dir}'.")

    model = BinaryResNet18(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Train loss={loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} finished. Training loss: {epoch_loss:.4f}")

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")

        save_classifier(model, args.save_path)
        print(f"Saved classifier to {args.save_path}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
