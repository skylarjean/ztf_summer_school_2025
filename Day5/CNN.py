import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from torchvision.models import DenseNet

class FeatureInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim*2)
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        projected = self.proj(x)  # [B, dim*2]
        gate = self.gate(projected)  # [B, dim]
        return x * gate  # Element-wise multiplication [B, dim] * [B, dim]

class CoordCNNJointTower(nn.Module):
    def __init__(self):
        super().__init__()
        # Coordinate processor
        self.coord_net = nn.Sequential(
            nn.Linear(2, 64),  # sgscore1, sgscore2
            FeatureInteraction(64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.SiLU()
        )
        
        # CNN backbone (DenseNet)
        self.cnn = DenseNetImageTower()
        
        # Cross-modality fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),  # CNN feats + coord feats
            FeatureInteraction(256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.SiLU()
        )

    def forward(self, coords, image):
        # Process coordinates
        coord_feats = self.coord_net(coords)  # [B, 128]
        
        # Process image
        img_feats = self.cnn(image) if image is not None else torch.zeros(coords.shape[0], 256, device=coords.device)  # [B, 256]
        
        # Early fusion
        joint_feats = self.fusion(torch.cat([img_feats, coord_feats], dim=1))
        return joint_feats  # [B, 256]



class DenseNetImageTower(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize pre-trained DenseNet-121 (remove classifier)
        self.densenet = DenseNet(
            growth_rate=32,
            block_config=(6, 12, 24, 16),  # DenseNet-121 configuration
            num_init_features=64,
            bn_size=4,
            drop_rate=0.0
        )
        
        # Remove original classifier
        self.densenet.classifier = nn.Identity()
        
        # Projection to match your 256D output
        self.proj = nn.Sequential(
            nn.Linear(1024, 256),  # DenseNet-121 final features: 1024
            nn.LayerNorm(256),
            nn.SiLU()
        )

    def forward(self, x):
        features = self.densenet(x)
        return self.proj(features)


class DeepCNNImageTower(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Added padding
        )
        
        # Conv Blocks with proper dimension handling
        self.block1 = self._make_conv_block(64, 128, downsample=True)
        self.block2 = self._make_conv_block(128, 256, downsample=True)
        self.block3 = self._make_conv_block(256, 512, downsample=True)
        
        # Attention and projection
        self.attention = ChannelAttention(512)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )

    def _make_conv_block(self, in_channels, out_channels, downsample):
        stride = 2 if downsample else 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)  # [B, 64, H/4, W/4]
        
        # Block 1 with dimension matching
        residual = F.avg_pool2d(x, kernel_size=2, stride=2) if x.size(-1) > 1 else x
        x = self.block1(x)
        x = x + residual if x.shape == residual.shape else x
        
        # Block 2
        residual = F.avg_pool2d(x, kernel_size=2, stride=2) if x.size(-1) > 1 else x
        x = self.block2(x)
        x = x + residual if x.shape == residual.shape else x
        
        # Block 3
        residual = F.avg_pool2d(x, kernel_size=2, stride=2) if x.size(-1) > 1 else x
        x = self.block3(x)
        x = x + residual if x.shape == residual.shape else x
        
        # Final processing
        x = self.attention(x)
        return self.proj(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)