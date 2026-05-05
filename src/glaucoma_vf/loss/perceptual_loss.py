import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = nn.Sequential(*list(vgg[:16])).eval()  # type: ignore
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # 1. Ensure 4D: [Batch, 1, 61, 61]
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 2. Upsample FIRST to (224, 224)
        # This fixes the "spatial dimensions" mismatch error
        pred_up = F.interpolate(
            pred, size=(224, 224), mode="bilinear", align_corners=False
        )
        target_up = F.interpolate(
            target, size=(224, 224), mode="bilinear", align_corners=False
        )

        # 3. Repeat channel to get RGB: [Batch, 3, 224, 224]
        pred_rgb = pred_up.repeat(1, 3, 1, 1)
        target_rgb = target_up.repeat(1, 3, 1, 1)

        # 4. Extract and compare features
        pred_features = self.layers(pred_rgb)
        target_features = self.layers(target_rgb)

        return F.mse_loss(pred_features, target_features)
