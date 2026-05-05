from typing import NamedTuple

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class HybViT(L.LightningModule):
    def __init__(self, lr=1e-4):  # 52 points for VF grid
        super().__init__()
        self.lr = lr

        # Load a Hybrid ViT (e.g., R50+ViT which uses a ResNet50 backbone)
        # 'vit_tiny_r_s16_p8_224' uses a small CNN + Transformer
        self.encoder = timm.create_model(
            "vit_tiny_r_s16_p8_224", num_classes=0, pretrained=True
        )

        # Mapping Transformer output to 61x61 grid
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.num_features, 1024),  # type: ignore
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 61 * 61),
        )

        self.loss_fn = nn.SmoothL1Loss()  # Better than MSE for grid regression

    def forward(self, X: FeatureSet):
        features = self.encoder(X.image)  # [Batch, 192]
        out = self.decoder(features)  # [Batch, 3721]
        out = out.view(-1, 61, 61)
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

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mse_loss",
            },
        }
