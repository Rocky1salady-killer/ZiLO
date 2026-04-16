import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ultralytics.utils import LOGGER
from timm.layers import SqueezeExcite
from .transformer import TransformerBlock
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, mn_conv, conv_ln, ln_conv


__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "MobileNetC2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "MobileNetV3_BLOCK",
    "MobileNetV4_BLOCK",
    "EfficientMobileNetBlock",
    "InvertedBottleneck",
    "RepViTBlock",
    "UltraRepViTBlock",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MobileNetC2f(nn.Module):
    """C2f variant using EfficientMobileNetBlock for lightweight feature extraction."""

    def __init__(self, c1, c2, n=3, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            EfficientMobileNetBlock(self.c, self.c, act="SI") for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.identity = nn.BatchNorm2d(c2) if self.add else nn.Identity()

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return self.identity(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def adjust_channels(channels: int, width_mult: float):
    return _make_divisible(channels * width_mult, 8)


class InvertedBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        super().__init__()
        c_mid = e if e is not None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else []
        features.extend([
            mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
            nn.Conv2d(c_mid, c2, 1),
            nn.BatchNorm2d(c2),
        ])
        self.layers = nn.Sequential(*features)

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV3_BLOCK(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        super().__init__()
        c_mid = e if e is not None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else []
        features.extend([
            mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
            nn.Conv2d(c_mid, c2, 1),
            nn.BatchNorm2d(c2),
        ])
        self.layers = nn.Sequential(*features)

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class ECAModule(nn.Module):
    """Efficient Channel Attention (ECA) module."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class SqueezeExcitation_mbnet(nn.Module):
    """Squeeze-and-Excitation block with hard-sigmoid activation."""

    def __init__(self, c_in, c_red):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_red),
            nn.Linear(c_red, c_in),
            nn.ReLU(),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).view(x.size(0), -1)
        y = self.fc(y).view(x.size(0), x.size(1), 1, 1)
        return x * y


class EfficientMobileNetBlock(nn.Module):
    """
    Efficient mobile inverted bottleneck block with adaptive channel expansion,
    adaptive attention (SE / ECA), and residual connections with optional projection.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (int): Depthwise kernel size.
        e: Unused; channel expansion is determined adaptively.
        pw (bool): Whether to include the pointwise expansion conv.
        act (str): Activation type for mn_conv layers.
        stride (int): Depthwise convolution stride.
    """

    def __init__(self, c1, c2, k=3, e=None, pw=True, act='SI', stride=1, *unused):
        super().__init__()

        # Adaptive channel expansion: wider for small inputs, narrower for large ones
        c_mid = max(c1, int(c1 * 4 * (1 - c1 / 320)))

        # Main branch
        feats = []
        if pw or c1 != c_mid:
            feats.append(mn_conv(c1, c_mid, k=1, s=1, g=1, act=act))
        feats.append(mn_conv(c_mid, c_mid, k=k, s=stride, g=c_mid, act=act))

        # Adaptive attention: SE for mid-range channels, ECA for large channels
        if c_mid > 8:
            if c_mid > 24:
                feats.append(ECAModule(c_mid))
            else:
                feats.append(SqueezeExcitation_mbnet(c_mid, max(2, c_mid // 8)))

        feats += [nn.Conv2d(c_mid, c2, 1, bias=False), nn.BatchNorm2d(c2)]
        self.layers = nn.Sequential(*feats)

        # Residual connection: identity if c1 == c2, learned projection otherwise
        self.enable_res = (stride == 1)
        if self.enable_res and c1 != c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, 1, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        y = self.layers(x)
        if self.enable_res:
            if hasattr(self, 'shortcut'):
                x = self.shortcut(x)
            y = x + y
        return y


def make_divisible(value, divisor=8, min_value=None, round_down_protect=True):
    """Ensures that all layers have channels that are divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class SqueezeExcitation_effnet(nn.Module):
    """Squeeze-and-Excitation block with SiLU activation."""

    def __init__(self, c_in, c_red):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_red),
            nn.Linear(c_red, c_in),
            nn.SiLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).view(x.size(0), -1)
        y = self.fc(y).view(x.size(0), x.size(1), 1, 1)
        return x * y


class MobileNetV4_BLOCK(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        super().__init__()
        c_mid = e if e is not None else c1 * 4
        c_mid = make_divisible(c_mid, 8)

        if act == "RE":
            activation = nn.ReLU6(inplace=True)
        elif act == "HS":
            activation = nn.Hardswish(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        start_dw = True
        middle_dw = True
        middle_dw_downsample = False

        if not pw:
            start_dw = False
            middle_dw = True
            middle_dw_downsample = True

        layers = []

        if start_dw:
            dw_stride = stride if not middle_dw_downsample else 1
            layers.append(nn.Conv2d(c1, c1, kernel_size=k, stride=dw_stride,
                                    padding=(k - 1) // 2, groups=c1, bias=False))
            layers.append(nn.BatchNorm2d(c1))

        layers.append(nn.Conv2d(c1, c_mid, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(c_mid))
        layers.append(activation)

        if middle_dw:
            mid_stride = stride if middle_dw_downsample else 1
            layers.append(nn.Conv2d(c_mid, c_mid, kernel_size=k,
                                    stride=mid_stride, padding=(k - 1) // 2,
                                    groups=c_mid, bias=False))
            layers.append(nn.BatchNorm2d(c_mid))
            layers.append(activation)

        layers.append(nn.Conv2d(c_mid, c2, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(c2))

        self.layers = nn.Sequential(*layers)
        self.residual = c1 == c2 and stride == 1

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class efficient_det(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, stride=1, act=None, pw=True):
        super().__init__()
        c_mid = e if e is not None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act="SI")] if pw else []
        features.extend([
            mn_conv(c_mid, c_mid, k, stride, g=c_mid, act="SI"),
            SqueezeExcitation_effnet(c_mid, c_mid // 24),
            nn.Conv2d(c_mid, c2, 1),
            nn.BatchNorm2d(c2),
        ])
        self.layers = nn.Sequential(*features)

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class classic_mobilenet_v3(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, stride=1, attn=True, act=None, pw=True):
        super().__init__()
        c_mid = e if e is not None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else []
        features.extend([
            mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
            SqueezeExcitation_mbnet(c_mid, c_mid // 4) if attn else nn.Identity(),
            nn.Conv2d(c_mid, c2, 1),
            nn.BatchNorm2d(c2),
        ])
        self.layers = nn.Sequential(*features)

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=None, mode="row"):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
        self.mode = mode

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        if self.mode == "row":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
        else:
            shape = [x.shape[0], x.shape[1]] + [1] * (x.ndim - 2)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNext_block(nn.Module):
    def __init__(self, c1, c2, c_mid, depth_prob=0.0):
        super().__init__()
        self.c_in = conv_ln(c1, c1)
        self.linear_in = nn.Linear(c1, c_mid)
        self.act = nn.GELU(approximate="none")
        self.linear_out = nn.Linear(c_mid, c1)
        self.stochastic_depth = StochasticDepth(depth_prob)
        self.skip = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, kernel_size=1)

    def forward(self, x):
        y = self.c_in(x)
        y = y.permute(0, 2, 3, 1)
        y = self.act(self.linear_in(y))
        y = self.linear_out(y)
        y = y.permute(0, 3, 1, 2)
        x = self.stochastic_depth(y) + x
        return x


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        conv = nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)

        self.add_module("c", conv)
        self.add_module("bn", bn)
        self.c = conv
        self.bn = bn

    @torch.no_grad()
    def fuse_self(self):
        c, bn = self.c, self.bn
        w = bn.weight / (bn.running_var + bn.eps).sqrt()
        fused_weight = c.weight * w[:, None, None, None]
        fused_bias = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps).sqrt()

        fused = nn.Conv2d(
            c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding,
            c.dilation, c.groups, bias=True, device=c.weight.device
        )
        fused.weight.data.copy_(fused_weight)
        fused.bias.data.copy_(fused_bias)
        return fused


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop)
            return x + self.m(x) * mask.div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse_self(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse_self()
            assert m.groups == m.in_channels
            identity = torch.eye(m.out_channels, m.in_channels).view(m.out_channels, m.in_channels, 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            identity = torch.eye(m.out_channels, m.in_channels).view(m.out_channels, m.in_channels, 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.dim = ed
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def fuse_self(self):
        conv = self.conv.fuse_self()
        conv1 = self.conv1
        identity = torch.eye(conv1.weight.size(0), conv1.weight.size(1)).view(
            conv1.weight.size(0), conv1.weight.size(1), 1, 1
        )
        conv1_w = F.pad(conv1.weight, [1, 1, 1, 1])
        identity = F.pad(identity, [1, 1, 1, 1]).to(conv1_w.device)

        conv.weight.data += conv1_w + identity
        conv.bias.data += conv1.bias

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps).sqrt()
        conv.weight.data *= w[:, None, None, None]
        conv.bias.data = bn.bias + (conv.bias.data - bn.running_mean) * w
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, kernel_size=3, oup=None, use_se=False, act="RE", stride=1, pw=True, hidden_dim=None):
        super().__init__()
        oup = oup or inp
        hidden_dim = hidden_dim or 2 * inp
        self.identity = (stride == 1 and inp == oup)

        act_map = {
            "RE": nn.ReLU6(inplace=True),
            "SI": nn.GELU(),
            "HS": nn.Hardswish()
        }
        if act not in act_map:
            raise ValueError(f"Unsupported activation: {act}")
        activation = act_map[act]

        if stride == 2 or not self.identity:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, 1, 1, 0)
            )
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity()
            )

        self.channel_mixer = Residual(nn.Sequential(
            Conv2d_BN(oup, hidden_dim, 1, 1, 0),
            activation,
            Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
        ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

    @torch.no_grad()
    def fuse(self):
        for i, layer in enumerate(self.token_mixer):
            if hasattr(layer, "fuse_self"):
                self.token_mixer[i] = layer.fuse_self()
        if hasattr(self.channel_mixer, "fuse_self"):
            self.channel_mixer = self.channel_mixer.fuse_self()
        return self


class LightweightAttention(nn.Module):
    """Combined channel and spatial attention module."""

    def __init__(self, channels: int, reduction: float = 0.25):
        super().__init__()
        reduced_channels = max(8, int(channels * reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        self.spatial = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        channel_att = self.avg_pool(x).view(b, c)
        channel_att = self.fc(channel_att).view(b, c, 1, 1)
        spatial_att = self.sigmoid(self.spatial(x))
        return x * channel_att * spatial_att


class UltraRepViTBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        kernel_size: int = 3,
        oup: Optional[int] = None,
        use_se: bool = False,
        act: str = "RE",
        stride: int = 1,
        pw: bool = True,
        hidden_dim: Optional[int] = None,
        depth_multiplier: int = 1,
        use_attention: str = "se"
    ):
        super().__init__()
        oup = oup or inp
        hidden_dim = hidden_dim or 2 * inp
        self.identity = (stride == 1 and inp == oup)

        act_map = {
            "RE": nn.ReLU6(inplace=True),
            "SI": nn.GELU(),
            "HS": nn.Hardswish(),
            "SW": nn.SiLU(),
            "MI": nn.Mish()
        }
        if act not in act_map:
            raise ValueError(f"Unsupported activation: {act}")
        activation = act_map[act]

        if stride == 2 or not self.identity:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                LightweightAttention(inp, 0.25) if use_attention == "lightweight" else
                SqueezeExcite(inp, 0.25) if use_se and use_attention == "se" else nn.Identity(),
                Conv2d_BN(inp, oup, 1, 1, 0)
            )
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                LightweightAttention(inp, 0.25) if use_attention == "lightweight" else
                SqueezeExcite(inp, 0.25) if use_se and use_attention == "se" else nn.Identity()
            )

        channel_layers = []
        for _ in range(depth_multiplier):
            channel_layers.append(
                Residual(nn.Sequential(
                    Conv2d_BN(oup, hidden_dim, 1, 1, 0),
                    activation,
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))
            )
        self.channel_mixer = nn.Sequential(*channel_layers)

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

    @torch.no_grad()
    def fuse_self(self):
        for i, layer in enumerate(self.token_mixer):
            if hasattr(layer, "fuse_self"):
                self.token_mixer[i] = layer.fuse_self()
        for i, layer in enumerate(self.channel_mixer):
            if hasattr(layer, "fuse_self"):
                self.channel_mixer[i] = layer.fuse_self()
        return self
