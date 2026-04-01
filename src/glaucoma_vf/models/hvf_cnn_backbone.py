from torch import nn


class HVFCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # HVF 24-2 is represented as an 8x9 grid
        # Input: (Batch, 1, 8, 9)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.conv(x)
