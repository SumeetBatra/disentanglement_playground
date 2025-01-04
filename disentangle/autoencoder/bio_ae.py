import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from disentangle.autoencoder.encoder import Encoder
from disentangle.autoencoder.decoder import Decoder
from typing import Dict


class BioAE(nn.Module):
    def __init__(self, obs_shape, num_latents, lambdas: Dict[str, float]):
        super(BioAE, self).__init__()
        self.num_latents = num_latents
        self.encoder = Encoder(obs_shape, num_latents * 2)
        self.decoder = Decoder(obs_shape, transition_shape=(256, 4, 4), num_latents=num_latents)
        self.lambdas = lambdas
        self.continuous_latent = True

    def encode(self, x):
        mu, logvar = torch.chunk(self.encoder(x)['pre_z'], chunks=2, dim=-1)
        # enforce nonnegativity hard constraint
        mu = F.relu(mu)
        return mu, logvar

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    @staticmethod
    def reparameterize(inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn((batch, dim), device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    @staticmethod
    def ent_reg(hidden):
        ent_reg_ = [(0.5 * hid ** 2).sum(dim=1).mean(0) for hid in hidden]
        return torch.stack(ent_reg_).sum()

    @staticmethod
    def nonneg_loss(hidden):
        nonneg = [F.relu(-hid).sum(dim=1).mean(0) for hid in hidden]
        return torch.stack(nonneg).sum()

    @staticmethod
    def weight_reg(weights, exclude=('abcdefghijklmnopqrstuvwxyz',)):
        exclude_ = ('bias',) + exclude
        weights_reg_ = 0
        for name, p in weights:
            if np.all([y not in name for y in exclude_]):
                weights_reg_ += p.pow(2).sum()
        return weights_reg_

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu
        logits = self.decode(z)['x_hat_logits']
        return logits, z, mu, logvar

    def batched_loss(self, batch):
        logits, z, mu, logvar = self.forward(batch['x'])

        bce_loss = F.binary_cross_entropy_with_logits(logits, batch['x'], reduction='none').sum(dim=(1, 2, 3)).mean()
        bce_loss *= self.lambdas['recon']

        entropy_loss = (0.5 * mu ** 2).sum(dim=-1)
        entropy_loss = self.lambdas['ent'] * entropy_loss.mean()

        nonneg_loss = F.relu(-mu).sum(dim=-1)
        nonneg_loss = self.lambdas['nonneg'] * nonneg_loss.mean()

        weight_reg = self.weight_reg(self.named_parameters(), exclude=('conv', 'encoder'))
        weight_loss = self.lambdas['weight'] * weight_reg

        total_loss = bce_loss + entropy_loss + weight_loss + nonneg_loss

        metrics = {
            'loss': total_loss.item(),
            'binary_cross_entropy_loss': bce_loss.item(),
            'latent_nonneg_loss': nonneg_loss.item(),
            'latent_entropy_loss': entropy_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }
        outs = {
            'x_hat_logits': logits,
            'z_hat': z
        }
        aux = {
            'metrics': metrics,
            'outs': outs,
        }
        return total_loss, aux





