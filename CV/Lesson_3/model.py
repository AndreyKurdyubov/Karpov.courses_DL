# !pip install fastparquet

from typing import Union

import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, squeeze_rate: int) -> None:
        super().__init__()
        
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // squeeze_rate, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // squeeze_rate, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se_block(x)
        return scale * x
        

class MBConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, exp_channels: int, kernel_size: Union[int, tuple[int, int]],
                 padding: Union[int, tuple[int, int]], stride: Union[int, tuple[int, int]], non_linearity: str = 'RE',
                 se_block: bool = True, squeeze_rate: int = 16) -> None:
        super().__init__()

        self.non_linearity_bank = {'RE': nn.ReLU6, 'HS': nn.Hardswish}

        self.use_skip_connection = stride != 2 and in_channels == out_channels

        self.layers = []

        if exp_channels != in_channels:
            self.layers.append(nn.Conv2d(in_channels, exp_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(exp_channels))
            self.layers.append(self.non_linearity_bank[non_linearity]())
        
        self.layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding, groups=exp_channels))
        self.layers.append(nn.BatchNorm2d(exp_channels))
        self.layers.append(self.non_linearity_bank[non_linearity]())
        
        if se_block:
            self.layers.append(SqueezeExcitation(exp_channels, squeeze_rate))
        
        self.layers.append(nn.Conv2d(exp_channels, out_channels, kernel_size=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        # Если пространственный размер входного тензора не меняется, то прибавляем скип
        if self.use_skip_connection:
            out = out + x
        return out


class MobileNetV3(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        self.feature_extractor = nn.Sequential(
            # bneck  in  out  exp k  p  s     NE    se
            MBConv2d(16, 16, 16, 3, 1, 1, 'RE', True),
            MBConv2d(16, 24, 64, 3, 1, 2, 'RE', False),
            MBConv2d(24, 24, 72, 3, 1, 1, 'HS', True),
            MBConv2d(24, 32, 72, 3, 1, 1, 'HS', True),
            MBConv2d(32, 64, 96, 3, 1, 2, 'HS', True),
            MBConv2d(64, 64, 128, 3, 1, 1, 'HS', True),
            MBConv2d(64, 128, 128, 3, 1, 1, 'HS', True),
            MBConv2d(128, 128, 256, 3, 1, 1, 'HS', True),
            MBConv2d(128, 128, 256, 3, 1, 1, 'HS', True),
            nn.Conv2d(128, 512, 1),
            nn.BatchNorm2d(512),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Hardswish(),
            nn.Linear(1024, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.classifier(x)
        return x