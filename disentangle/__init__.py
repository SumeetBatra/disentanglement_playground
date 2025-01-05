import torch
import yaml

from disentangle.autoencoder.quantized_ae import QuantizedAutoEncoder
from disentangle.autoencoder.bio_ae import BioAE
from disentangle.autoencoder.beta_vae import BetaVAE
from disentangle.autoencoder.beta_tcvae import BetaTCVAE
from disentangle.autoencoder.slot_ae import SlotAutoEncoder


def model_factory(model_key: str):
    if model_key == 'qlae':
        with open('./disentangle/configs/qlae.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        model = QuantizedAutoEncoder(obs_shape=(3, 64, 64),
                                     num_latents=cfg['model']['num_latents'],
                                     values_per_latent=cfg['model']['values_per_latent'],
                                     lambdas=cfg['model']['lambdas'],)
        model_optim = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()),
                                        lr=cfg['model']['model_lr'], weight_decay=cfg['model']['weight_decay'])
        latent_optim = torch.optim.Adam(model.latent.parameters(), lr=cfg['model']['latent_lr'])
    elif model_key == 'bio_ae':
        with open('./disentangle/configs/bio_ae.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        model = BioAE(obs_shape=(3, 64, 64), num_latents=cfg['model']['num_latents'], lambdas=cfg['model']['lambdas'])
        model_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        latent_optim = None
    elif model_key == 'beta_vae':
        with open('./disentangle/configs/beta_vae.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        model = BetaVAE(obs_shape=(3, 64, 64), num_latents=cfg['model']['num_latents'], lambdas=cfg['model']['lambdas'])
        model_optim = torch.optim.Adam(model.parameters(), lr=cfg['model']['model_lr'])
        latent_optim = None
    elif model_key == 'slot_ae':
        with open('./disentangle/configs/slot_ae.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        model = SlotAutoEncoder(obs_shape=(3, 64, 64),
                                slot_dim=cfg['model']['slot_dim'],
                                num_slots=cfg['model']['num_latents'],
                                adaptive=cfg['model']['adaptive'],
                                lambdas=cfg['model']['lambdas'])
        model_optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.1)
        latent_optim = None
    elif model_key == 'beta_tcvae':
        with open('./disentangle/configs/beta_tcvae.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        model = BetaTCVAE(obs_shape=(3, 64, 64),
                          num_latents=cfg['model']['num_latents'],
                          lambdas=cfg['model']['lambdas'])
        model_optim = torch.optim.Adam(model.parameters(), lr=cfg['model']['model_lr'])
        latent_optim = None
    else:
        raise ValueError(f'Unknown model type {model_key}')

    return model, model_optim, latent_optim
