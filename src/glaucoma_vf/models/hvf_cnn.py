import sys
from typing import NamedTuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, F1Score, MeanSquaredError, MetricCollection


class FeatureSet(NamedTuple):
    grids: torch.Tensor
    age: torch.Tensor
    years_since_first: torch.Tensor
    years_since_last: torch.Tensor


class LabelSet(NamedTuple):
    category: torch.Tensor
    mtd: torch.Tensor
    grids: torch.Tensor


class Batch(NamedTuple):
    X: FeatureSet
    y: LabelSet


class ModelOutput(NamedTuple):
    curr_category: torch.Tensor
    next_mtd: torch.Tensor
    next_hvf: torch.Tensor


class HVFSystem(L.LightningModule):
    def __init__(self, backbone, num_classes=3, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = backbone

        num_features_with_temporal = 32 * 8 * 9
        num_features_with_temporal = num_features_with_temporal + 3

        # 1. Classifier Head (Mild/Moderate/Severe)
        self.classifier = nn.Linear(num_features_with_temporal, num_classes)

        # # 2. Forecast: Class (predicting future stage)
        # self.forecast_class = nn.Linear(num_features_with_temporal, num_classes)

        # 3. Forecast: Mean Deviation (Regression)
        self.forecast_md = nn.Linear(num_features_with_temporal, 1)

        # 4. Forecast: Full HVF (8x9 Grid Reconstruction)
        self.forecast_hvf = nn.Sequential(
            nn.Linear(num_features_with_temporal, 256),
            nn.ReLU(),
            nn.Linear(256, 72),  # 8x9 = 72 pixels
            nn.Unflatten(1, (1, 8, 9)),
        )

        # Metrics for Classification (Current & Forecast)
        metrics_cls = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=3),
                "f1": F1Score(task="multiclass", num_classes=3),
            }
        )

        # Separate collections for each "head" to avoid naming collisions
        self.val_cls_metrics = metrics_cls.clone(prefix="val/curr_")
        self.val_md_metrics = MeanSquaredError()
        self.val_hvf_metrics = MeanSquaredError()

    def forward(self, X: FeatureSet):
        # 1. Extract spatial features: Shape [Batch, Num Features]
        spatial_features = self.backbone(X.grids)

        # 2. Concatenate temporal features: Shape [Batch, Num Features + 3]
        combined = torch.cat(
            [
                spatial_features,
                X.age.unsqueeze(1),
                X.years_since_first.unsqueeze(1),
                X.years_since_last.unsqueeze(1),
            ],
            dim=1,
        )

        return ModelOutput(
            curr_category=self.classifier(combined),
            next_mtd=self.forecast_md(combined),
            next_hvf=self.forecast_hvf(combined),
        )

    def training_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)

        # Calculate individual losses
        # loss_cls = F.cross_entropy(out["curr_category"], y_class)
        loss_md = F.mse_loss(out.next_mtd.squeeze(), batch.y.mtd)
        loss_hvf = F.mse_loss(out.next_hvf, batch.y.grids)

        # Weighted Total Loss
        # total_loss = loss_cls + (0.5 * loss_md) + (2.0 * loss_hvf)
        total_loss = loss_md + loss_hvf

        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)

        # 1. Compute Losses (for monitoring convergence)
        # loss_curr = F.cross_entropy(out["curr_category"], y_class)
        loss_md = F.mse_loss(out.next_mtd.squeeze(), batch.y.mtd)
        loss_hvf = F.mse_loss(out.next_hvf, batch.y.grids)

        # total_val_loss = loss_curr + loss_md + loss_hvf
        total_val_loss = loss_md + loss_hvf

        # 2. Update Metrics (No need to log every step, just update)
        self.val_cls_metrics(out.curr_category, batch.y.category)
        self.val_md_metrics(out.next_mtd.squeeze(), batch.y.mtd)
        self.val_hvf_metrics(out.next_hvf, batch.y.grids)

        # 3. Log Scalars
        self.log_dict(
            {
                "val/total_loss": total_val_loss,
                "val/md_mse": self.val_md_metrics.compute(),
                "val/hvf_mse": self.val_hvf_metrics.compute(),
            },
            prog_bar=True,
        )
        self.log_dict(self.val_cls_metrics.compute())
        return total_val_loss

    def on_validation_epoch_end(self):
        # Compute and log all metrics at once
        self.log_dict(self.val_cls_metrics.compute())
        self.log("val/md_mse", self.val_md_metrics.compute())
        self.log("val/hvf_mse", self.val_hvf_metrics.compute())

        # Reset for next epoch
        self.val_cls_metrics.reset()

    def test_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)

        return ModelOutput(
            curr_category=out.curr_category.argmax(dim=1),
            next_mtd=out.next_mtd.squeeze(),
            next_hvf=out.next_hvf,
        )

    def _shared_step(self, batch) -> tuple[Batch, ModelOutput]:
        features = FeatureSet(**batch["X"])
        labels = LabelSet(**batch["y"])
        batch = Batch(X=features, y=labels)

        out = self(batch.X)

        return batch, out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore
