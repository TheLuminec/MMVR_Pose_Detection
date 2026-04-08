from torch import nn
import torch

# Develop CNN-based encoder-decoder with residual and attention blocks
# Decoder: 2 convolutional layers reconstructing spatial detail
# 1×1 convolution projects features to 17 keypoint heatmaps
# Soft-Argmax layer provides differentiable coordinate prediction
# Output: 17 keypoint (x, y) coordinates per frame
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.convh = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.convh2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.convv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.convv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.dense = nn.Linear(64 * 256 * 128, 17 * 2)

    def forward(self, hori, vert):
        x_hori = self.convh(hori)
        x_hori = self.convh2(x_hori)
        x_vert = self.convv(vert)
        x_vert = self.convv2(x_vert)

        x = torch.cat((x_hori, x_vert), dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense(x)
        x = x.view(x.size(0), 17, 2)  # Reshape to (batch_size, 17, 2)

        return x
        
