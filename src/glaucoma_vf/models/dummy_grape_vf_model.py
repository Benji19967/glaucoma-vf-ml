import sys
from typing import NamedTuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSet(NamedTuple):
    grids: torch.Tensor


class LabelSet(NamedTuple):
    grids: torch.Tensor


class Batch(NamedTuple):
    X: FeatureSet
    y: LabelSet


class ModelOutput(NamedTuple):
    pred_grids: torch.Tensor


class DummyVFModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Encoder: Downsample slightly
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 61x61 -> 30x30
        )

        # Decoder: Upsample back to 61x61
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=0
            ),
            # ConvTranspose output will be 59x59, so we pad to reach 61x61
            nn.ReplicationPad2d((1, 1, 1, 1)),
        )

    def forward(self, X: FeatureSet):
        # X.grids shape: (Batch, 1, 61, 61)
        z = self.encoder(X.grids)
        out = self.decoder(z)
        return ModelOutput(pred_grids=out)

    def training_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)
        loss = F.mse_loss(out.pred_grids, batch.y.grids)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)
        loss = F.mse_loss(out.pred_grids, batch.y.grids)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)

        return out

    def _shared_step(self, batch) -> tuple[Batch, ModelOutput]:
        features = FeatureSet(**batch["X"])
        labels = LabelSet(**batch["y"])
        batch = Batch(X=features, y=labels)

        out = self(batch.X)

        return batch, out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
