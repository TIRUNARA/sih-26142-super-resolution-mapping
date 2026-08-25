"""
SRM-Net: Deep Residual Attention Super-Resolution & Sub-Pixel Mapping Model
Tailored for Multi-Spectral Medium Resolution Satellite Imagery (e.g. Sentinel-2 / Landsat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel Attention Mechanism for Satellite Band Weighting"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualAttentionBlock(nn.Module):
    """Residual Block with Channel Attention for feature extraction"""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.ca(res)
        return x + res

class SatelliteSRMNet(nn.Module):
    """
    Satellite Super Resolution Mapping Network (SRM-Net)
    Upscales medium-resolution (e.g. 10m/30m) satellite input to 4x spatial resolution.
    """
    def __init__(self, in_channels: int = 4, num_features: int = 64, num_blocks: int = 6, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Shallow Feature Extraction
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Deep Feature Extraction Stack
        self.body = nn.Sequential(*[ResidualAttentionBlock(num_features) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Sub-Pixel Upsampling (PixelShuffle)
        upsample_layers = []
        if scale_factor == 4:
            for _ in range(2):
                upsample_layers.extend([
                    nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU()
                ])
        else:
            upsample_layers.extend([
                nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale_factor),
                nn.PReLU()
            ])
            
        self.upsample = nn.Sequential(*upsample_layers)
        
        # High-Resolution Output Reconstruction
        self.tail = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save low-res input for global skip connection
        lr_bicubic = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        
        head_feat = self.head(x)
        body_feat = self.body(head_feat)
        body_feat = self.body_conv(body_feat) + head_feat
        
        up_feat = self.upsample(body_feat)
        res_hr = self.tail(up_feat)
        
        return torch.clamp(lr_bicubic + res_hr, 0.0, 1.0)


if __name__ == "__main__":
    # Quick sanity test
    model = SatelliteSRMNet(in_channels=4, scale_factor=4)
    dummy_input = torch.randn(1, 4, 32, 32)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Super-Resolved Output shape: {output.shape}")
