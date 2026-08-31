import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (RCAN) for Multi-Spectral Band Relational Weighting.
    Learns spatial-spectral relationships between RGB and Near-Infrared (NIR) bands.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True)
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RCABlock(nn.Module):
    """
    Residual Channel Attention Block for deep feature extraction without spatial blur.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(RCABlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            ChannelAttention(in_channels, reduction=reduction)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SpatialAttractionFilter(nn.Module):
    """
    Sub-Pixel Spatial Attraction & Spectral Confidence Filter.
    Enforces contiguous land-cover feature bounds and removes AI artifacts.
    """
    def __init__(self, num_bands: int = 4):
        super(SpatialAttractionFilter, self).__init__()
        self.conv = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1)
        self.gate = nn.Sigmoid()

    def forward(self, sr_tensor: torch.Tensor) -> torch.Tensor:
        attraction_weights = self.gate(self.conv(sr_tensor))
        return sr_tensor * attraction_weights


class SatelliteSRMNet(nn.Module):
    """
    Deep Learning Super Resolution Mapping Network (SRM-Net).
    Upscales multi-spectral satellite tiles (RGB + NIR) by 4x using Spatial-Spectral Relational Fitting.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, scale_factor: int = 4, num_features: int = 64):
        super(SatelliteSRMNet, self).__init__()
        self.scale_factor = scale_factor
        
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        self.trunk = nn.Sequential(
            RCABlock(num_features),
            RCABlock(num_features),
            RCABlock(num_features),
            RCABlock(num_features)
        )
        self.mid_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )
        
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        self.spatial_filter = SpatialAttractionFilter(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow_feats = self.head(x)
        deep_feats = self.trunk(shallow_feats)
        deep_feats = self.mid_conv(deep_feats) + shallow_feats
        
        upscaled_feats = self.upsample(deep_feats)
        sr_output = self.tail(upscaled_feats)
        
        filtered_output = self.spatial_filter(sr_output)
        return filtered_output
