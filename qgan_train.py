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
        lr = transforms.ToTensor()(lr_image)
        return lr, hr


class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity (SSIM) Loss for preserving anatomical edges."""
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.register_buffer("window", self._create_window(window_size, self.channel))

    def _gaussian_kernel(self, size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, size: int, channel: int) -> torch.Tensor:
        k = self._gaussian_kernel(size)
        w2d = k.unsqueeze(1).mm(k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        return w2d.expand(channel, 1, size, size).contiguous()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        if channel != self.channel:
            self.channel = channel
            self.window = self._create_window(self.window_size, channel).to(img1.device)

        w = self.window.to(img1.device)
        padding = self.window_size // 2

        mu1 = nn.functional.conv2d(img1, w, padding=padding, groups=channel)
        mu2 = nn.functional.conv2d(img2, w, padding=padding, groups=channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = nn.functional.conv2d(img1 * img1, w, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, w, padding=padding, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, w, padding=padding, groups=channel) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        return 1.0 - ssim_map.mean()


class TotalVariationLoss(nn.Module):
    """Total Variation (TV) loss to suppress high-frequency noise and checkboard artifacts."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (b * c * h * w)


class PerceptualLoss(nn.Module):
    """Anatomical edge & high-frequency perceptual feature loss (with optional VGG support)."""
    def __init__(self, device: torch.device, use_vgg: bool = False):
        super().__init__()
        self.device = device
        self.vgg = None
        if use_vgg:
            try:
                from torchvision.models import vgg19, VGG19_Weights
                vgg_model = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval().to(device)
                for param in vgg_model.parameters():
                    param.requires_grad = False
                self.vgg = vgg_model
            except Exception:
                self.vgg = None

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(1, 3, 1, 1).to(device)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(1, 3, 1, 1).to(device)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.criterion = nn.L1Loss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if self.vgg is not None:
            mean = torch.tensor([0.485, 0.456, 0.406], device=sr.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=sr.device).view(1, 3, 1, 1)
            sr_norm = (sr - mean) / std
            hr_norm = (hr - mean) / std
            return self.criterion(self.vgg(sr_norm), self.vgg(hr_norm))
        else:
            sr_gx = nn.functional.conv2d(sr, self.sobel_x, padding=1)
            hr_gx = nn.functional.conv2d(hr, self.sobel_x, padding=1)
            sr_gy = nn.functional.conv2d(sr, self.sobel_y, padding=1)
            hr_gy = nn.functional.conv2d(hr, self.sobel_y, padding=1)
            return self.criterion(sr_gx, hr_gx) + self.criterion(sr_gy, hr_gy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a QGAN super-resolution model with advanced medical loss functions")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hr"), help="High-resolution training images directory")
    parser.add_argument("--save-path", type=Path, default=None, help="Path to save the trained generator (auto-selected by modality if not set)")
    parser.add_argument("--modality", type=str, default="general", choices=["general", "brain_mri", "chest_xray", "ct_scan"], help="Target medical scan modality")
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
    parser.add_argument("--use-quantum-refiner", action="store_true", help="Enable PennyLane hybrid quantum refinement block")
    # Advanced Loss Weights
    parser.add_argument("--w-l1", type=float, default=1.0, help="L1 Content Loss weight")
    parser.add_argument("--w-ssim", type=float, default=0.15, help="SSIM Structural Loss weight")
    parser.add_argument("--w-perceptual", type=float, default=0.05, help="VGG Perceptual Feature Loss weight")
    parser.add_argument("--w-tv", type=float, default=1e-5, help="Total Variation Denoising Loss weight")
    parser.add_argument("--w-adv", type=float, default=1e-3, help="Adversarial Generator Loss weight")
    return parser


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.data_dir.exists():
        raise SystemExit(f"Training data directory not found: {args.data_dir}")

    # Determine save path based on modality
    if args.save_path is None:
        if args.modality == "general":
            save_path = Path("models/qgan_generator.pt")
        else:
            save_path = Path(f"models/qgan_generator_{args.modality}.pt")
    else:
        save_path = args.save_path

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

    # Losses
    adversarial_loss = nn.BCELoss()
    content_loss = nn.L1Loss()
    ssim_loss = SSIMLoss().to(device)
    perceptual_loss = PerceptualLoss(device)
    tv_loss = TotalVariationLoss()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    print(f"--- Starting QGAN Training [Modality: {args.modality.upper()}] ---")
    print(f"Loss formulation: {args.w_l1}*L1 + {args.w_ssim}*SSIM + {args.w_perceptual}*VGG + {args.w_tv}*TV + {args.w_adv}*GAN")
    print(f"Target Checkpoint: {save_path} on {device}")

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

            # Train generator with composite multi-task objective
            optimizer_g.zero_grad()
            fake_pred = discriminator(sr)
            g_adv = adversarial_loss(fake_pred, real_labels)
            g_l1 = content_loss(sr, hr)
            g_ssim = ssim_loss(sr, hr) if args.w_ssim > 0 else torch.tensor(0.0, device=device)
            g_perc = perceptual_loss(sr, hr) if args.w_perceptual > 0 else torch.tensor(0.0, device=device)
            g_tv = tv_loss(sr) if args.w_tv > 0 else torch.tensor(0.0, device=device)

            g_loss = (
                args.w_l1 * g_l1
                + args.w_ssim * g_ssim
                + args.w_perceptual * g_perc
                + args.w_tv * g_tv
                + args.w_adv * g_adv
            )
            g_loss.backward()
            optimizer_g.step()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(loader)} | D={d_loss.item():.4f} | G_total={g_loss.item():.4f} (L1={g_l1.item():.4f}, SSIM={g_ssim.item():.4f})")

        save_generator(generator, save_path)
        print(f"Saved generator to {save_path} after epoch {epoch}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
