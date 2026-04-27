from typing import NamedTuple

import lightning as L
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


class EfficientNetModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr

        # Load pretrained weights (trained on 1M+ images)
        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        # Freeze early layers (optional, but helps with small datasets)
        # for param in self.backbone.features[:4].parameters():
        #     param.requires_grad = False

        # Replace the classifier head
        # EfficientNet-B0 output features are 1280
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Essential for small datasets
            nn.Linear(512, 3721),  # 61 * 61 grid
        )
        self.backbone.classifier = self.head

    def forward(self, X: FeatureSet):
        # Reshape back to the 61x61 grid
        out = self.backbone(X.image).view(-1, 1, 61, 61)
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
