import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class HVFClassifier(L.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # HVF 24-2 is represented as an 8x9 grid
        # Input: (Batch, 1, 8, 9)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32 * 8 * 9, num_classes)

        # 1. Initialize the metric
        # "macro" --> balanced accuracy rather than std accuracy.
        # This is important for imbalanced datasets like ours.
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")

    def forward(self, x):
        # x shape: (batch, 1, 8, 9), (batch, channels, height, width)
        # print(x.shape)
        return self.fc(self.conv_layers(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 2. Update the metric with predictions
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)

        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": acc,
                "val_balanced_acc": self.val_acc.compute(),
            },
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 2. Update the metric with predictions
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)

        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": acc,
                "test_balanced_acc": self.test_acc.compute(),
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
