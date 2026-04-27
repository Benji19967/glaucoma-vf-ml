from typing import NamedTuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSet(NamedTuple):
    annotated_image: torch.Tensor


class LabelSet(NamedTuple):
    grid: torch.Tensor
    image_name: str


class Batch(NamedTuple):
    X: FeatureSet
    y: LabelSet


class ModelOutput(NamedTuple):
    pred_grids: torch.Tensor


class DummyVFModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr

        # 1. Feature Extractor (CNN)
        # Input: (B, 3, 432, 432)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> 216x216
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 108x108
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 54x54
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 27x27
            nn.Flatten(),  # -> 32 * 27 * 27 = 23,328
        )

        # 2. Prediction Head
        # Projects CNN features to the 61x61 grid (3721 values)
        self.head = nn.Sequential(
            nn.Linear(32 * 27 * 27, 3721),
            nn.Sigmoid(),  # Use Sigmoid if your dB values are normalized [0, 1]
        )

    def forward(self, X: FeatureSet):
        features = self.feature_extractor(X.annotated_image)
        out = self.head(features)
        # Reshape back to the 61x61 grid
        return ModelOutput(pred_grids=out.view(-1, 1, 61, 61))

    def training_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)
        loss = F.mse_loss(out.pred_grids, batch.y.grid)
        self.log("train/mse_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)
        loss = F.mse_loss(out.pred_grids, batch.y.grid)
        self.log("val/mse_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx) -> ModelOutput:  # type: ignore
        batch, out = self._shared_step(batch)
        loss = F.mse_loss(out.pred_grids, batch.y.grid)
        self.log("test/mse_loss", loss, prog_bar=True)

        return out

    def _shared_step(self, batch) -> tuple[Batch, ModelOutput]:
        features = FeatureSet(**batch["X"])
        labels = LabelSet(**batch["y"])
        batch = Batch(X=features, y=labels)

        out = self(batch.X)

        return batch, out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
