import torch
import wandb
import einops
import torch.nn.functional as F
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model, preprocessing


def peak_signal_to_noise_ratio(x_hat_logits, x_true):
    mse = torch.mean(torch.square(F.sigmoid(x_hat_logits) - x_true))
    max_value = torch.Tensor([1.]).to(mse.device)
    return 20 * torch.log10(max_value) - 10 * torch.log10(mse)


def discretize_unique(z):
    """

    :param z: (num_samples, num_dims)
    :return:
    """
    ret = np.zeros_like(z, dtype=np.int32)
    for i in range(z.shape[1]):
        unique, labels = np.unique(z[:, i], return_inverse=True)
        ret[:, i] = labels
    return ret


def compute_entropy(z):
    """

    :param z: (num_samples, num_dims)
    :return:
    """
    ret = np.zeros(z.shape[1])
    if z.dtype in [np.int32, np.int64, torch.int32, torch.int64]:
        for i in range(z.shape[1]):
            ret[i] = sklearn.metrics.mutual_info_score(z[:, i], z[:, i])
    else:
        for i in range(z.shape[1]):
            ret[i] = sklearn.feature_selection.mutual_info_regression(z[:, i][:, None], z[:, i])
    return ret


def log_reconstruction_metrics(aux, step, use_wandb: bool = False):
    num_samples = 16
    true = einops.rearrange(aux['x_true'][:num_samples], 'b c h w -> h (b w) c')
    predicted = einops.rearrange(F.sigmoid(aux['x_hat_logits'][:num_samples]), 'b c h w -> h (b w) c')
    absolute_diff = torch.abs(true - predicted)
    image = torch.cat((true, predicted, absolute_diff), dim=0)
    psnr = peak_signal_to_noise_ratio(aux['x_hat_logits'], aux['x_true'])

    if use_wandb:
        wandb.log({
            'reconstructions': wandb.Image(image.detach().cpu().numpy()),
            'ae/psnr': psnr.mean().item()
        }, step=step)


