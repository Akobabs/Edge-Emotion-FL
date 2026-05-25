import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise + pointwise conv — PyTorch equivalent of Keras SeparableConv2D."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionModule(nn.Module):
    """
    Single XCEPTION module: residual 1×1 stride-2 shortcut + [SepConv→BN→ReLU→SepConv→BN→MaxPool].
    Spatial dimension is halved by the stride-2 MaxPool + matching stride-2 shortcut.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sep1 = SeparableConv2d(in_channels, out_channels)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.sep2 = SeparableConv2d(out_channels, out_channels)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.sep1(x)), inplace=True)
        x = self.bn2(self.sep2(x))
        x = self.pool(x)
        return F.relu(x + residual, inplace=True)


class EmotionCNN(nn.Module):
    """
    PyTorch port of mini_XCEPTION (oarriaga/face_classification, ~65-66% on FER-2013).
    Input: (B, 1, 64, 64)  |  Output logits: (B, 7)

    Architecture (spatial dims with 64×64 input):
      Base:    Conv(1→8, valid)  → Conv(8→8, valid)           → 60×60
      Block 1: XceptionModule(8→16)   stride-2 MaxPool         → 30×30
      Block 2: XceptionModule(16→32)  stride-2 MaxPool         → 15×15
      Block 3: XceptionModule(32→64)  stride-2 MaxPool         →  8×8
      Block 4: XceptionModule(64→128) stride-2 MaxPool         →  4×4
      Head:    Conv(128→7, same) → GlobalAvgPool               →  7
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # Two valid-padded (no padding) 3×3 convs reduce 64→62→60
        self.base = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.block1 = XceptionModule(8,   16)
        self.block2 = XceptionModule(16,  32)
        self.block3 = XceptionModule(32,  64)
        self.block4 = XceptionModule(64, 128)
        self.conv_pred = nn.Conv2d(128, num_classes, 3, padding=1)
        self.gap       = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_pred(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)


def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EmotionCNN()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {get_model_size(model):,}")
