import torch

from disentangle.autoencoder.quantized_ae import QuantizedAutoEncoder
from disentangle.autoencoder.bio_ae import BioAE


def model_factory(model_key: str):
    if model_key == 'qlae':
        model = QuantizedAutoEncoder(obs_shape=(3, 64, 64),
                                     num_latents=10,
                                     values_per_latent=10)
        optimizer = torch.optim.Adam([
            {'params': list(model.encoder.parameters()) + list(model.decoder.parameters()), 'lr': 1e-3, 'weight_decay': 0.1},
            {'params': model.latent.parameters(), 'lr': 1e-3}
        ])
    elif model_key == 'bio_ae':
        model = BioAE(obs_shape=(3, 64, 64), num_latents=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        raise ValueError(f'Unknown model type {model_key}')

    return model, optimizer
