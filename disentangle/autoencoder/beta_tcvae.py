import torch
import torch.nn.functional as F

from disentangle.autoencoder.beta_vae import BetaVAE
from typing import Tuple, Dict


def gaussian_log_density(samples: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor):
    '''Borrowed from https://github.com/google-research/disentanglement_lib/'''
    device = samples.device
    normalization = torch.log(torch.Tensor([2.0 * torch.pi]).to(device))
    inv_sigma = torch.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def total_correlation(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    '''
    Also borrowed from https://github.com/google-research/disentanglement_lib/
    compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]
    '''

    # compute log( q(z(x_j) | x_i)) for every sample in the batch, which produces a tensor of
    # shape (batch_size x batch_size x num_latents)
    log_qz_prob = gaussian_log_density(
        z[:, None, :],
        mu[None, :, :],
        logvar[None, :, :]
    )
    log_qz_product = log_qz_prob.logsumexp(dim=1).sum(dim=1)
    log_qz = log_qz_prob.sum(dim=2).logsumexp(dim=1)
    return (log_qz - log_qz_product).mean()


class BetaTCVAE(BetaVAE):
    def __init__(self, obs_shape: Tuple[int, ...], num_latents: int, lambdas: Dict[str, float]):
        super(BetaTCVAE, self).__init__(obs_shape, num_latents, lambdas)

    def batched_loss(self, batch):
        x_hat, z_hat, mu, log_var = self(batch['x'])

        kl_div_loss = -0.5 * (1. + log_var - mu.pow(2) - log_var.exp())
        kl_div_loss = self.lambdas['kl_div'] * kl_div_loss.mean()
        recon_loss = F.binary_cross_entropy_with_logits(x_hat, target=batch['x'], reduction='none').sum((1, 2, 3)).mean()
        recon_loss = recon_loss * self.lambdas['recon']
        tc_loss = total_correlation(z_hat, mu, log_var)
        tc_loss = self.lambdas['tc'] * tc_loss
        total_loss = kl_div_loss + recon_loss + tc_loss

        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_div_loss': kl_div_loss.item(),
            'tc_loss': tc_loss.item(),
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

