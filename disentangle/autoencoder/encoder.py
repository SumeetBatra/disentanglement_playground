import torch
import torch.nn as nn

from typing import Tuple, Dict, Any


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class QuantizedEncoder(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 num_latents: int,
                 amp: bool = False):
        super(QuantizedEncoder, self).__init__()
        self.obs_shape = obs_shape

        self.conv = nn.Sequential(
                    Conv2DBlock(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
                    Conv2DBlock(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                    Conv2DBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                    Conv2DBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                    nn.Flatten(),
                )

        self.dense = nn.Sequential(
            nn.Linear(in_features=4096, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU()
        )

        self.head = nn.Linear(in_features=256, out_features=num_latents)

        self.amp = amp

    def encode(self, rgbd: torch.Tensor, state: torch.Tensor = None):
        x = self.conv(rgbd)
        features = self.dense(x)
        pre_z = self.head(features)
        outs = {
            'pre_z': pre_z,
            'features': features,
        }
        return outs

    def forward(self, rgbd: torch.Tensor, state: torch.Tensor = None):
        with torch.autocast(device_type="cuda", enabled=self.amp):
            outs = self.encode(rgbd, state)
        return outs