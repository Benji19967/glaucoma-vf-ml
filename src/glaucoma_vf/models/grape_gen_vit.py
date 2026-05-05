from typing import NamedTuple

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSet(NamedTuple):
    image: torch.Tensor


class LabelSet(NamedTuple):
    grid: torch.Tensor
    image_name: str


class Batch(NamedTuple):
    X: FeatureSet
    y: LabelSet


class ModelOutput(NamedTuple):
    pred_grids: torch.Tensor


class GenViT(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr

        # Encoder: Extracts features from Optic Disc
        self.encoder = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        )

        # Decoder: Generative upsampling to reach 61x61
        # We start from the latent vector and "expand" spatially
        self.decoder = nn.Sequential(
            nn.Linear(192, 1024),
            nn.Unflatten(1, (64, 4, 4)),  # Turn vector into a 4x4 feature map
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),  # 9x9
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),  # 19x19
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2),  # 39x39
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=1),  # Adjusting to 61x61
            nn.AdaptiveAvgPool2d((61, 61)),
        )

    def forward(self, X: FeatureSet):
        latent = self.encoder(X.image)
        out = self.decoder(latent).squeeze(1)  # [Batch, 61, 61]
        return ModelOutput(pred_grids=out)

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
