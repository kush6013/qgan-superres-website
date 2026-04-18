"""QGAN training helper.

Place your high-resolution training images in `data/hr/`.
The script loads PNG/JPG/JPEG images recursively from that folder.
It downscales each image by the configured scale factor and trains a generator
that can sharpen / super-resolve low-resolution inputs.

Usage:
    ./venv/bin/python qgan_train.py --data-dir data/hr --epochs 20 --batch-size 8
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.qgan_model import QGANGenerator, QGANDiscriminator, save_generator


class SRDataset(Dataset):
    def __init__(
        self,
        hr_dir: Path,
        hr_size: int = 256,
        scale: int = 8,
        augment: bool = False,
        rotation: int = 10,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.files = [p for p in hr_dir.rglob("*.png")] + [p for p in hr_dir.rglob("*.jpg")] + [p for p in hr_dir.rglob("*.jpeg")]
        self.hr_size = hr_size
        self.scale = scale
        self.augment = augment
        self.hr_transform = transforms.Compose([
            transforms.CenterCrop(hr_size),
            transforms.ToTensor(),
        ])
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop(hr_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(rotation, interpolation=Image.BICUBIC),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")

        if min(image.size) < self.hr_size:
            image = transforms.Resize(self.hr_size, interpolation=Image.BICUBIC)(image)

        if self.augment:
            hr = self.augment_transform(image)
        else:
            hr = self.hr_transform(image)

        lr_image = transforms.Resize(self.hr_size // self.scale, interpolation=Image.BICUBIC)(transforms.ToPILImage()(hr))
        lr_image = transforms.Resize(self.hr_size, interpolation=Image.BICUBIC)(lr_image)
        lr = transforms.ToTensor()(lr_image)
        return lr, hr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a QGAN super-resolution model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hr"), help="High-resolution training images directory")
    parser.add_argument("--save-path", type=Path, default=Path("models/qgan_generator.pt"), help="Path to save the trained generator")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hr-size", type=int, default=256, help="High-resolution crop size")
    parser.add_argument("--scale", type=int, default=8, help="Upscale factor")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--rotation", type=int, default=10, help="Max rotation degrees for augmentation")
    parser.add_argument("--brightness", type=float, default=0.1, help="Brightness jitter for augmentation")
    parser.add_argument("--contrast", type=float, default=0.1, help="Contrast jitter for augmentation")
    parser.add_argument("--saturation", type=float, default=0.1, help="Saturation jitter for augmentation")
    parser.add_argument("--hue", type=float, default=0.05, help="Hue jitter for augmentation")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument(
        "--use-quantum-refiner",
        action="store_true",
        help="Enable the optional PennyLane-based hybrid refinement block in the generator",
    )
    return parser


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.data_dir.exists():
        raise SystemExit(f"Training data directory not found: {args.data_dir}")

    dataset = SRDataset(
        args.data_dir,
        hr_size=args.hr_size,
        scale=args.scale,
        augment=args.augment,
        rotation=args.rotation,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    generator = QGANGenerator(use_quantum_refiner=args.use_quantum_refiner).to(device)
    discriminator = QGANDiscriminator().to(device)

    adversarial_loss = nn.BCELoss()
    content_loss = nn.L1Loss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        for batch_idx, (lr, hr) in enumerate(loader, start=1):
            lr = lr.to(device)
            hr = hr.to(device)

            # Train discriminator
            optimizer_d.zero_grad()
            sr = generator(lr)
            real_labels = torch.ones((lr.size(0), 1), device=device)
            fake_labels = torch.zeros((lr.size(0), 1), device=device)
            real_pred = discriminator(hr)
            fake_pred = discriminator(sr.detach())
            d_loss = (adversarial_loss(real_pred, real_labels) + adversarial_loss(fake_pred, fake_labels)) * 0.5
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            fake_pred = discriminator(sr)
            g_adv = adversarial_loss(fake_pred, real_labels)
            g_content = content_loss(sr, hr)
            g_loss = g_content + 1e-3 * g_adv
            g_loss.backward()
            optimizer_g.step()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(loader)} | D_loss={d_loss.item():.4f} | G_loss={g_loss.item():.4f}")

        save_generator(generator, args.save_path)
        print(f"Saved generator to {args.save_path} after epoch {epoch}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
