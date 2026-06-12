from typing import Optional, Union, Any

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BasicBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BasicDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BasicUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BasicSkip(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([x, skip], dim=1))

# Решение автора
class UNet(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 filters: list[int],
                 conv_block: Optional[nn.Module] = None,
                 bottleneck_block: Optional[nn.Module] = None,
                 downsampling_block: Optional[nn.Module] = None,
                 upsampling_block: Optional[nn.Module] = None,
                 skip_block: Optional[nn.Module] = None,
                 block_kwargs: dict[str, dict[str, Any]] = {}):
        super().__init__()

        self.n_blocks = len(filters)
        self.blocks = {
            'conv': conv_block or BasicBlock,
            'bottleneck': bottleneck_block or BasicBottleneck,
            'down': downsampling_block or BasicDown,
            'up': upsampling_block or BasicUp,
            'skip': skip_block or BasicSkip
        }
        self.kwargs = {
            'conv': block_kwargs.get('conv', {}),
            'bottleneck': block_kwargs.get('bottleneck', {}),
            'down': block_kwargs.get('down', {}),
            'up': block_kwargs.get('up', {}),
            'skip': block_kwargs.get('skip', {})
        }
        
        self.conv2d_in = nn.Conv2d(in_channels, filters[0], 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            self.blocks['conv'](filters[i-1], **self.kwargs['conv']) 
            for i in range(1, self.n_blocks)
        ])
        
        self.downsampling_blocks = nn.ModuleList([
            self.blocks['down'](filters[i-1], filters[i], **self.kwargs['down'])
            for i in range(1, self.n_blocks)
        ])
        
        # Bottleneck
        self.bottleneck = self.blocks['bottleneck'](filters[-1], **self.kwargs['bottleneck'])
        
        # Decoder
        self.upsampling_blocks = nn.ModuleList([
            self.blocks['up'](filters[i], filters[i-1], **self.kwargs['up'])
            for i in range(self.n_blocks-1, 0, -1)
        ])
        
        self.skip_blocks = nn.ModuleList([
            self.blocks['skip'](filters[i-1], **self.kwargs['skip'])
            for i in range(self.n_blocks-1, 0, -1)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            self.blocks['conv'](filters[i-1], **self.kwargs['conv'])
            for i in range(self.n_blocks-1, 0, -1)
        ])
        
        self.conv2d_out = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_in(x)
        
        # Encoder path with skip connections
        skips = []
        for encode, down in zip(self.encoder_blocks, self.downsampling_blocks):
            x = encode(x)
            skips.append(x)
            x = down(x)
        
        x = self.bottleneck(x)
        
        # Decoder path
        for up, skip_connection, decode in zip(
            self.upsampling_blocks,
            self.skip_blocks,
            self.decoder_blocks
        ):
            x = up(x)
            x = skip_connection(x, skips.pop())
            x = decode(x)
        
        return self.conv2d_out(x)

# адаптированное решение автора из 2-го урока
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
        

class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, exp_channels: int = None, kernel_size: Union[int, tuple[int, int]] = 3,
                 padding: Union[int, tuple[int, int]] = 1, stride: Union[int, tuple[int, int]] = 1, non_linearity: str = 'RE',
                 se_block: bool = True, squeeze_rate: int = 16) -> None:
        super().__init__()

        self.non_linearity_bank = {'RE': nn.ReLU6, 'HS': nn.Hardswish}

        self.use_skip_connection = stride != 2 and in_channels == out_channels

        self.layers = []

        if out_channels is None:
            out_channels = in_channels

        if exp_channels is None:
            exp_channels = 4 * in_channels

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

# Решение автора
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAMSkip(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(2 * channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return self.conv(x)

# Решение автора
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv_block(x))
        
class ResBottleneck(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        
        self.res_block1 = ResBlock(channels)
        self.res_block2 = ResBlock(channels)
        
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return self.relu(x + residual)
    

# Initialize model
config = {
    'in_channels': 3,
    'out_channels': 3,
    'filters': [16, 32, 64, 128],
    'conv_block': MBConv,
    'bottleneck_block': ResBottleneck,
    'skip_block': CBAMSkip
}