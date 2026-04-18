import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

try:
    import pennylane as qml
except Exception:  # pragma: no cover - optional dependency at runtime
    qml = None


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual


class QuantumRefinementBlock(nn.Module):
    """Optional PennyLane-powered channel refinement for hybrid experimentation."""

    def __init__(self, channels: int, num_qubits: int = 4, num_layers: int = 2):
        super().__init__()
        if qml is None:
            raise RuntimeError("PennyLane is required to use QuantumRefinementBlock")

        self.channels = channels
        self.num_qubits = num_qubits
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_qubits),
            nn.Tanh(),
        )

        device = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(device, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        weight_shapes = {"weights": (num_layers, num_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.project = nn.Sequential(
            nn.Linear(num_qubits, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder(x)
        refined = self.quantum_layer(pooled)
        gates = self.project(refined).view(x.size(0), self.channels, 1, 1)
        return x + (x * gates)


class QGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_residuals: int = 8,
        use_quantum_refiner: bool = False,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])
        self.body_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )

        self.quantum_refiner = (
            QuantumRefinementBlock(64)
            if use_quantum_refiner and qml is not None
            else nn.Identity()
        )
        self.uses_quantum_refiner = use_quantum_refiner and qml is not None
        self.tail = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body(head)
        body = self.body_conv(body)
        x = head + body
        x = self.quantum_refiner(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x


class QGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_generator(model_path: Path, device: torch.device) -> Optional[QGANGenerator]:
    if not model_path.exists():
        return None

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, nn.Module):
        generator = checkpoint.to(device).eval()
    else:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            uses_quantum_refiner = any(
                key.startswith("quantum_refiner.") for key in state_dict.keys()
            )
            generator = QGANGenerator(use_quantum_refiner=uses_quantum_refiner).to(device)
            generator.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Unsupported checkpoint format for QGAN generator")
        generator.eval()

    return generator


def save_generator(model: QGANGenerator, model_path: Path) -> None:
    torch.save(model.state_dict(), model_path)
