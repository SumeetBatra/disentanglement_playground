import torch
import torch.nn as nn
import torch.nn.functional as F

from disentangle.autoencoder.quantized_ae import Encoder, Decoder
from typing import Tuple, Dict, List


class BetaVAE(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], num_latents: int, lambdas: Dict[str, float]):
        super(BetaVAE, self).__init__()
        self.num_latents = num_latents
        self.encoder = Encoder(obs_shape, 2 * num_latents)
        self.decoder = Decoder(obs_shape, transition_shape=(256, 4, 4), num_latents=num_latents)
        self.lambdas = lambdas
        self.continuous_latent = True

    def forward(self, x):
        mu, log_var = self.encoder(x)['pre_z'].chunk(2, dim=-1)
        z = mu + log_var.exp() * torch.randn_like(log_var)
        x_hat = self.decoder(z)['x_hat_logits']
        return x_hat, z, mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)['pre_z'].chunk(2, dim=-1)
        z = mu + log_var.exp() * torch.randn_like(log_var)
        return z

    def decode(self, z):
        return self.decoder(z)['x_hat_logits']

    def batched_loss(self, batch):
        x_hat, z_hat, mu, log_var = self(batch['x'])
        kl_div_loss = -0.5 * (1. + log_var - mu.pow(2) - log_var.exp())
        kl_div_loss = self.lambdas['kl_div'] * kl_div_loss.mean()
        recon_loss = F.binary_cross_entropy_with_logits(x_hat, target=batch['x'], reduction='none').sum((1, 2, 3)).mean()
        recon_loss = recon_loss * self.lambdas['recon']

        total_loss = kl_div_loss + recon_loss
        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_div_loss': kl_div_loss.item(),
        }

        outs = {
            'x_hat_logits': x_hat,
            'z_hat': z_hat,
            'mu': mu,
            'log_var': log_var,
        }
        aux = {
            'metrics': metrics,
            'outs': outs
        }
        return total_loss, aux


