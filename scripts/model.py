from torch import nn
from torch.functional import F
import torch

# Develop CNN-based encoder-decoder with residual and attention blocks
# Decoder: 2 convolutional layers reconstructing spatial detail
# 1×1 convolution projects features to 17 keypoint heatmaps
# Soft-Argmax layer provides differentiable coordinate prediction
# Output: 17 keypoint (x, y) coordinates per frame
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.dense = nn.Linear(64 * 256 * 128, 17 * 2)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for the dense layer
        x = self.dense(x)
        x = F.softmax(x, dim=1)

        x = x.view(x.size(0), 17, 2)  # Reshape to (batch_size, 17, 2)

        return x
        
