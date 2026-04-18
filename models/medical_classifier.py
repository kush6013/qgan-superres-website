from pathlib import Path
from PIL import Image, ImageStat
import torch
import torch.nn as nn
from torchvision import models, transforms

REJECT_KEYWORDS = {
    "dog",
    "cat",
    "bird",
    "flower",
    "fruit",
    "tree",
    "mountain",
    "lake",
    "ocean",
    "beach",
    "vehicle",
    "car",
    "truck",
    "airplane",
    "plane",
    "ship",
    "bicycle",
    "motorcycle",
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "horse",
    "cow",
    "pig",
    "sheep",
    "elephant",
    "lion",
    "tiger",
    "bear",
    "fish",
    "insect",
    "butterfly",
    "spider",
    "snake",
    "apple",
    "banana",
    "orange",
    "grape",
    "pineapple",
    "chair",
    "sofa",
    "bench",
    "desk",
    "building",
    "skyscraper",
    "bridge",
    "road",
    "street",
    "window",
    "door",
    "handbag",
    "shoe",
    "hat",
    "umbrella",
    "camera",
    "bottle",
    "glass",
    "keyboard",
    "computer",
    "phone",
    "television",
    "guitar",
    "piano",
    "drum",
}


class BinaryResNet18(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def save_classifier(model: nn.Module, model_path: Path) -> None:
    torch.save(model.state_dict(), model_path)


def load_classifier(model_path: Path, device: torch.device) -> nn.Module:
    model = BinaryResNet18(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


class MedicalImageClassifier:
    def __init__(self, device: torch.device | None = None, checkpoint_path: Path | None = None):
        self.device = device or torch.device("cpu")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.custom_model = None
        self.custom_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.fallback_model = None
        self.fallback_transform = None
        self.labels = []
        self._load_fallback_model()
        self._load_custom_model()

    def _load_fallback_model(self):
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.fallback_model = models.resnet18(weights=weights).to(self.device).eval()
            self.fallback_transform = weights.transforms()
            self.labels = weights.meta["categories"]
        except Exception:
            self.fallback_model = models.resnet18(pretrained=True).to(self.device).eval()
            self.fallback_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.labels = [f"class_{i}" for i in range(1000)]

    def _load_custom_model(self):
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                self.custom_model = load_classifier(self.checkpoint_path, self.device)
            except Exception:
                self.custom_model = None

    def _predict_imagenet_labels(self, image: Image.Image) -> list[str]:
        tensor = self.fallback_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.fallback_model(tensor)
            probabilities = torch.softmax(logits, dim=-1)
            topk = torch.topk(probabilities, k=5, dim=-1)
            indices = topk.indices[0].cpu().tolist()
        return [self.labels[idx] for idx in indices]

    def _predict_custom(self, image: Image.Image) -> bool:
        if self.custom_model is None:
            return True
        tensor = self.custom_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.custom_model(tensor).squeeze(1)
            score = torch.sigmoid(logits).item()
        return score >= 0.5

    def _is_obvious_nonmedical(self, image: Image.Image) -> bool:
        labels = self._predict_imagenet_labels(image)
        for label in labels:
            normalized = label.lower()
            for keyword in REJECT_KEYWORDS:
                if keyword in normalized:
                    return True
        return False

    def _has_color_distribution_of_natural_photo(self, image: Image.Image) -> bool:
        hsv = image.convert("HSV")
        _, saturation, _ = hsv.split()
        histogram = saturation.histogram()
        total_pixels = sum(histogram) or 1
        bright_pixels = sum(histogram[int(0.35 * 255) :])
        return (bright_pixels / total_pixels) > 0.35

    def _has_scan_grayscale_profile(self, image: Image.Image) -> bool:
        r, g, b = image.split()
        stat_r = ImageStat.Stat(r)
        stat_g = ImageStat.Stat(g)
        stat_b = ImageStat.Stat(b)
        avg_r, avg_g, avg_b = stat_r.mean[0], stat_g.mean[0], stat_b.mean[0]
        return abs(avg_r - avg_g) + abs(avg_r - avg_b) + abs(avg_g - avg_b) < 50

    def is_medical_scan(self, image: Image.Image) -> bool:
        if image.width < 128 or image.height < 128:
            return False

        if self.custom_model is not None:
            return self._predict_custom(image)

        if self._is_obvious_nonmedical(image):
            return False

        if self._has_color_distribution_of_natural_photo(image) and not self._has_scan_grayscale_profile(image):
            return False

        return True
