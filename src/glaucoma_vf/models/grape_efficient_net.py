from typing import Any, NamedTuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


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

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        Called before a batch is transferred to the device.
        We apply data augmentation here only during training.
        """
        if self.trainer.training:
            batch_t = self._get_typed_batch(batch)

            # Get random parameters for affine transform
            # (degrees, translate, scale, shear)
            params = transforms.RandomAffine.get_params(
                degrees=[-10, 10],
                translate=[0.1, 0.1],
                scale_ranges=[1.0, 1.0],
                shears=[0, 0],
                img_size=batch_t.X.image.shape[-2:],  # type: ignore
            )

            batch["X"]["image"] = transforms.functional.affine(batch_t.X.image, *params)  # type: ignore
            batch["y"]["grid"] = transforms.functional.affine(batch_t.y.grid, *params)  # type: ignore

        return batch

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
        batch = self._get_typed_batch(batch)

        out = self(batch.X)

        return batch, out

    def _get_typed_batch(self, batch) -> Batch:
        features = FeatureSet(**batch["X"])
        labels = LabelSet(**batch["y"])
        batch = Batch(X=features, y=labels)

        return batch

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # We want to minimize validation loss
            factor=0.5,  # Reduce LR by half when performance stalls
            patience=5,  # Number of epochs to wait before reducing
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mse_loss",  # This must match the key used in self.log()
                "interval": "epoch",
                "frequency": 1,
            },
        }
