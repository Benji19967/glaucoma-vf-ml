from typing import NamedTuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MeanSquaredError,
    MetricCollection,
)


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
    def __init__(self, backbone, num_classes, learning_rate):
        super().__init__()

        # Saves hyperparameters (params passed in init) to self.hparams
        # for easy access and logging. They are set in train.yaml and
        # passed via the CLI.
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

        # weight inversely proportional to class frequency because of imbalanced data affecting classifier accuracy
        # Mild: 13015, Moderate: 4722, Severe: 3778
        # class_weights = torch.tensor([1.0, 2.76, 3.44], device=self.device) removed for focal loss

        # ---cross entropy---
        # loss_cls = F.cross_entropy(out.curr_category, batch.y.category.long(), weight=class_weights)
        # loss_cls = F.cross_entropy(out.curr_category, batch.y.category.long())
        # ---

        # ---focal loss---
        loss_cls = self._focal_loss(out.curr_category, batch.y.category.long())
        # ---

        loss_md = F.mse_loss(out.next_mtd.squeeze(), batch.y.mtd)
        loss_hvf = F.mse_loss(out.next_hvf, batch.y.grids)

        # Weighted Total Loss
        total_loss = loss_cls + (0.5 * loss_md) + (2.0 * loss_hvf)
        # total_loss = loss_md + loss_hvf

        self.log("train/total_loss", total_loss)
        self.log("train/cls_loss", loss_cls)
        self.log("train/md_loss", loss_md)
        self.log("train/hvf_loss", loss_hvf)
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch, out = self._shared_step(batch)

        # 1. Compute Losses (for monitoring convergence)
        # loss_curr = F.cross_entropy(out["curr_category"], y_class)

        # same
        # Weight inversely proportional to class frequency
        # Mild: 13015, Moderate: 4722, Severe: 3778
        # class_weights = torch.tensor([1.0, 2.76, 3.44], device=self.device) removed for focal loss
        # loss_cls = F.cross_entropy(out.curr_category, batch.y.category.long(), weight=class_weights)
        # loss_cls = F.cross_entropy(out.curr_category, batch.y.category.long())

        # ---focal loss---
        loss_cls = self._focal_loss(out.curr_category, batch.y.category.long())
        # ---

        loss_md = F.mse_loss(out.next_mtd.squeeze(), batch.y.mtd)
        loss_hvf = F.mse_loss(out.next_hvf, batch.y.grids)

        # total_val_loss = loss_curr + loss_md + loss_hvf
        # total_val_loss = loss_md + loss_hvf
        total_val_loss = loss_cls + (0.5 * loss_md) + (2.0 * loss_hvf)

        # 2. Update Metrics (No need to log every step, just update)
        self.val_cls_metrics(out.curr_category, batch.y.category)
        self.val_md_metrics(out.next_mtd.squeeze(), batch.y.mtd)
        self.val_hvf_metrics(out.next_hvf, batch.y.grids)

        # 3. Log Scalars
        self.log_dict(
            {
                "val/total_loss": total_val_loss,
                "val/cls_loss": loss_cls,
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

    def test_step(self, batch, batch_idx):  # added the actual accuracy for the testing
        batch, out = self._shared_step(batch)

        # Regression errors
        loss_md = F.mse_loss(out.next_mtd.squeeze(), batch.y.mtd)
        loss_hvf = F.mse_loss(out.next_hvf, batch.y.grids)
        self.log("test/md_mse", loss_md)
        self.log("test/hvf_mse", loss_hvf)

        labels = batch.y.category.long()

        # classifier
        preds = out.curr_category.argmax(dim=1)
        # labels = batch.y.category.long()
        # acc = (preds == batch.y.category.long()).float().mean()
        acc = (preds == labels).float().mean()

        self.log("test/md_mse", loss_md)
        self.log("test/hvf_mse", loss_hvf)
        self.log("test/cls_acc", acc, prog_bar=True)

        # accuracy per class, because since most cases are Mild, the model says it performs good but is actually bad at detecting other classes (aka. accuracy paradox)
        for cls in range(3):
            mask = labels == cls
            if mask.sum() > 0:
                cls_acc = (preds[mask] == labels[mask]).float().mean()
                self.log(f"test/cls_acc_{cls}", cls_acc)

        return ModelOutput(
            # curr_category=out.curr_category.argmax(dim=1),
            curr_category=preds,
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

    def _focal_loss(
        self, logits, targets, gamma=2.0
    ):  # focal loss replaces cross entropy loss:  dynamically adjusts based on how confident the model is. If the model easily predicts a "Mild" case (high confidence), it reduces the loss to near zero
        class_weights = torch.tensor([1.0, 2.76, 3.44], device=self.device)
        ce_loss = F.cross_entropy(
            logits, targets, weight=class_weights, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()
