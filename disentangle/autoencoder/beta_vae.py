import torch
import torch.nn as nn

from disentangle.autoencoder.quantized_ae import QuantizedEncoder, QuantizedDecoder


class BetaVAE(nn.Module):
    def __init__(self, obs_shape, transition_shape, num_latents):
        super(BetaVAE, self).__init__()
        self.encoder = QuantizedEncoder(obs_shape, 2 * num_latents)
        self.decoder = QuantizedDecoder(obs_shape, transition_shape, num_latents)

    def forward(self, x):
        mu, log_var = self.encoder(x)['pre_z'].chunk(2, dim=-1)
        z = mu + log_var.exp() * torch.randn_like(log_var)
        x_hat = self.decoder(z)['x_hat_logits']

        return x_hat, mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)['pre_z'].chunk(2, dim=-1)
        z = mu + log_var.exp() * torch.randn_like(log_var)
        return z

    def decode(self, z):
        return self.decoder(z)['x_hat_logits']
