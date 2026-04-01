import sys

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, F1Score, MeanSquaredError, MetricCollection


class HVFSystem(L.LightningModule):
    def __init__(self, backbone, num_classes=3, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = backbone

        # 1. Classifier Head (Mild/Moderate/Severe)
        self.classifier = nn.Linear(32 * 8 * 9, num_classes)

        # # 2. Forecast: Class (predicting future stage)
        # self.forecast_class = nn.Linear(32 * 8 * 9, num_classes)

        # 3. Forecast: Mean Deviation (Regression)
        self.forecast_md = nn.Linear(32 * 8 * 9, 1)

        # 4. Forecast: Full HVF (8x9 Grid Reconstruction)
        self.forecast_hvf = nn.Sequential(
            nn.Linear(32 * 8 * 9, 256),
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

    def forward(self, x):
        features = self.backbone(x)
        return {
            "current_class": self.classifier(features),
            "next_md": self.forecast_md(features),
            "next_hvf": self.forecast_hvf(features),
        }

    def training_step(self, batch, batch_idx):
        x_grid, y_class, y_mtd, y_grid = batch
        out = self(x_grid)

        # Calculate individual losses
        loss_cls = F.cross_entropy(out["current_class"], y_class)
        loss_md = F.mse_loss(out["next_md"].squeeze(), y_mtd)
        loss_hvf = F.mse_loss(out["next_hvf"], y_grid)

        # Weighted Total Loss
        total_loss = loss_cls + (0.5 * loss_md) + (2.0 * loss_hvf)

        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x_grid, y_class, y_mtd, y_grid = batch

        out = self(x_grid)

        # 1. Compute Losses (for monitoring convergence)
        loss_curr = F.cross_entropy(out["current_class"], y_class)
        loss_md = F.mse_loss(out["next_md"].squeeze(), y_mtd)
        loss_hvf = F.mse_loss(out["next_hvf"], y_grid)

        total_val_loss = loss_curr + loss_md + loss_hvf

        # 2. Update Metrics (No need to log every step, just update)
        self.val_cls_metrics(out["current_class"], y_class)
        self.val_md_metrics(out["next_md"].squeeze(), y_mtd)
        self.val_hvf_metrics(out["next_hvf"], y_grid)

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
        x_grid, y_class, y_mtd, y_grid = batch
        out = self(x_grid)

        # We return the predictions for the Callback
        return {
            "x": x_grid,
            "y_class": y_class,
            "y_mtd": y_mtd,
            "y_grid": y_grid,  # Actual future HVF
            "pred_class": out["current_class"].argmax(dim=1),
            "pred_mtd": out["next_md"].squeeze(),
            "pred_grid": out["next_hvf"],  # Predicted future HVF
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore
