import torch
import torch.nn as nn
import torch.nn.functional as F

from disentangle.autoencoder.encoder import QuantizedEncoder
from disentangle.autoencoder.decoder import QuantizedDecoder
from disentangle.latents.quantized import AssociativeQuantizedLatent


class AssociativeAutoEncoder(nn.Module):
    def __init__(self, obs_shape):
        super(AssociativeAutoEncoder, self).__init__()
        self.num_latents = 12
        self.values_per_latent = 20
        self.obs_shape = obs_shape
        self.encoder = QuantizedEncoder(obs_shape, num_latents=self.num_latents)
        self.decoder = QuantizedDecoder(obs_shape, transition_shape=(256, 4, 4), num_latents=self.num_latents)
        self.latent = AssociativeQuantizedLatent(self.num_latents, self.values_per_latent, quantity=256, emb_dim=128)

        self.lambdas = {
            'binary_cross_entropy': 1.0,
            'quantization': 0.01,
            'commitment': 0.01,
            'hopfield': 0.01,
            'l2': 0.1,
            'l1': 0.0
        }

    def forward(self, x):
        outs_enc = self.encoder(x)
        outs_latent = self.latent(outs_enc['pre_z'])
        outs_dec = self.decoder(outs_latent['z_hat'])
        outs = {**outs_enc, **outs_latent, **outs_dec}
        return outs

    def batched_loss(self, batch):
        outs = self(batch['x'])
        bce_loss = F.binary_cross_entropy_with_logits(outs['x_hat_logits'], target=batch['x'])
        # cov_loss = self.latent.decorrelate(outs['z_hopfield'])
        cov_loss = torch.Tensor([0]).cuda()
        quantization_loss = (outs['z_continuous'].detach() - outs['z_quantized']).pow(2).mean(1)
        commitment_loss = (outs['z_continuous'] - outs['z_quantized'].detach()).pow(2).mean(1)

        # hopfield_lhs = (outs['z_hopfield'].detach() - outs['z_hat']).pow(2).mean(1)
        hopfield_rhs = (outs['z_hopfield'] - outs['z_hat']).pow(2).mean(1)
        hopfield_loss = hopfield_rhs

        total_loss = self.lambdas['binary_cross_entropy'] * bce_loss + \
                     self.lambdas['quantization'] * quantization_loss + \
                     self.lambdas['commitment'] * commitment_loss + \
                     self.lambdas['hopfield'] * hopfield_loss
        total_loss = total_loss.mean()

        metrics = {
            'loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'cov_loss': cov_loss,
            'hopfield_loss': hopfield_loss.mean().item(),
            'quantization_loss': quantization_loss.mean().item(),
            'commitment_loss': commitment_loss.mean().item(),
        }

        aux = {
            'metrics': metrics,
            'outs': outs
        }

        return total_loss, aux