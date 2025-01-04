import torch
import torch.nn as nn

from typing import Tuple


class Conv2DTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DTransposeBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Decoder(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 transition_shape: Tuple[int, ...],
                 num_latents: int):
        super().__init__()
        self.obs_shape = obs_shape
        self.network = nn.Sequential(
            nn.Linear(num_latents, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=transition_shape),
            Conv2DTransposeBlock(256, 128, kernel_size=4, stride=2, padding=1),
            Conv2DTransposeBlock(128, 64, kernel_size=4, stride=2, padding=1),
            Conv2DTransposeBlock(64, 32, kernel_size=4, stride=2, padding=1),
            Conv2DTransposeBlock(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z_hat: torch.Tensor):
        '''
        :param z: posterior
        :param deterministic: deterministics
        '''
        x_hat_logits = self.network(z_hat)
        out = {'x_hat_logits': x_hat_logits}

        return out