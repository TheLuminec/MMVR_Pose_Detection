import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class SoftArgmax2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is (B, C, H, W)
        b, c, h, w = x.shape
        
        x = x.view(b, c, h * w)
        x_softmax = F.softmax(x, dim=2)
        x_softmax = x_softmax.view(b, c, h, w)
        
        # Y grid corresponds to W (cols)
        y_grid = torch.linspace(0, 1, steps=w, device=x.device).view(1, 1, 1, w).expand(b, c, h, w)
        # X grid corresponds to H (rows)
        x_grid = torch.linspace(0, 1, steps=h, device=x.device).view(1, 1, h, 1).expand(b, c, h, w)
        
        # Expected value
        expected_y = torch.sum(x_softmax * y_grid, dim=(2, 3)) # (B, C) - this is width
        expected_x = torch.sum(x_softmax * x_grid, dim=(2, 3)) # (B, C) - this is height
        
        # Stack as (B, C, 2)
        coords = torch.stack([expected_x, expected_y], dim=2)
        return coords

# Develop CNN-based encoder-decoder with residual and attention blocks
# Decoder: 2 convolutional layers reconstructing spatial detail
# 1x1 convolution projects features to 17 keypoint heatmaps
# Soft-Argmax layer provides differentiable coordinate prediction
# Output: 17 keypoint (x, y) coordinates per frame
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (Input is 2 channel radar: hori & vert, size 256x128)
        self.in_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),  # -> 128x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                   # -> 64x32
        )
        
        self.layer1 = BasicBlock(64, 128, stride=2)   # -> 32x16
        self.layer2 = BasicBlock(128, 256, stride=2)  # -> 16x8
        
        # Self-Attention
        self.attention = CBAMBlock(256)
        
        # Decoder (2 transposes)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # -> 32x16
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> 64x32
        
        self.heatmap_proj = nn.Conv2d(64, 17, kernel_size=1)
        self.soft_argmax = SoftArgmax2d()

    def forward(self, x):
        # Encoder
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.attention(x)
        
        # Decoder
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        
        # Project to 17 keypoint heatmaps
        x = self.heatmap_proj(x)
        
        # Produce (x, y) coordinates normalized to [0, 1]
        coords = self.soft_argmax(x)
        
        # Scale according to [480, 640] where 480 is x(height) and 640 is y(width)
        scale = torch.tensor([480.0, 640.0], device=coords.device)
        coords = coords * scale

        return coords
        
