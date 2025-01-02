import numpy as np
import torch
import argparse
import wandb
import einops
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from tqdm import tqdm
from disentangle.datasets.shapes3d import get_datasets
from disentangle.autoencoder.quantized_ae import QuantizedAutoEncoder
from disentangle.autoencoder.bio_ae import BioAE
from disentangle.utils.metrics import *
from pathlib import Path
from distutils.util import strtobool
from typing import Dict, Any
from common.utils import _setup_wandb


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bio_ae', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_project', type=str, default='sim2real')
    parser.add_argument('--wandb_group', type=str, default='disentangle')
    parser.add_argument('--wandb_run_name', type=str, default='qlae_shapes3D')
    parser.add_argument('--wandb_tag', type=str, default='QLAE')

    return vars(parser.parse_args())


def evaluate(val_set, model, step: int, use_wandb: bool = False):
    loss, bce_loss, quant_loss, commit_loss = [], [], [], []
    final_aux = {}
    for batch in iter(val_set):
        batch['x'] = torch.from_numpy(batch['x']).to('cuda')
        _, aux = model.batched_loss(batch)
        aux['x_true'] = batch['x']
        aux['z_true'] = batch['z']
        aux['z_hat'] = aux['outs']['z_hat']
        aux['x_hat_logits'] = aux['outs']['x_hat_logits']
        # auxs.append(aux)
        loss.append(aux['metrics']['loss'])
        bce_loss.append(aux['metrics']['binary_cross_entropy_loss'])
        quant_loss.append(aux['metrics']['quantization_loss'])
        commit_loss.append(aux['metrics']['commitment_loss'])

        final_aux = aux

    if use_wandb:
        wandb.log({
            'eval/total_loss': sum(loss) / len(loss),
            'eval/bce_loss': sum(bce_loss) / len(bce_loss),
            'eval/quantization_loss': sum(quant_loss) / len(quant_loss),
            'eval/commitment_loss': sum(commit_loss) / len(commit_loss)
        }, step=step)
    log_reconstruction_metrics(final_aux, step, use_wandb=use_wandb)


def evaluate_bio_ae(val_set, model, step: int, use_wandb: bool = False):
    loss, bce_loss, nonneg_loss, entropy_loss = [], [], [], []
    final_aux = {}
    for batch in iter(val_set):
        batch['x'] = torch.from_numpy(batch['x']).to('cuda')
        _, aux = model.batched_loss(batch)
        aux['x_true'] = batch['x']
        aux['x_hat_logits'] = aux['outs']['x_hat_logits']
        loss.append(aux['metrics']['loss'])
        bce_loss.append(aux['metrics']['binary_cross_entropy_loss'])
        # nonneg_loss.append(aux['metrics']['latent_nonneg_loss'])
        entropy_loss.append(aux['metrics']['latent_entropy_loss'])

        final_aux = aux

    if use_wandb:
        wandb.log({
            'eval/total_loss': sum(loss) / len(loss),
            'eval/bce_loss': sum(bce_loss) / len(bce_loss),
            # 'eval/latent_nonneg_loss': sum(nonneg_loss) / len(nonneg_loss),
            'eval/latent_entropy_loss': sum(entropy_loss) / len(entropy_loss),
        }, step=step)
    log_reconstruction_metrics(final_aux, step, use_wandb=use_wandb)

    if use_wandb:
        num_samples = 16
        num_perturbations = 16
        # generate perturbations to the latents and see what happens
        for i_latent in range(model.num_latents):
            print(f'Interpolating {num_perturbations} samples for latent {i_latent}')
            latent_mins = aux['outs']['z_hat'].min(dim=0).values
            latent_maxs = aux['outs']['z_hat'].max(dim=0).values

            latent_perturbed = torch.tile(aux['outs']['z_hat'][:num_samples],
                                          (num_perturbations, 1, 1))  # (num_perturbations, num_samples, num_latents)
            latent_perturbed[:, :, i_latent] = torch.linspace(latent_mins[i_latent], latent_maxs[i_latent],
                                                              num_perturbations)[:, None]

            x = []
            with torch.no_grad():
                for i_perturbation in range(num_perturbations):
                    x.append(
                        F.sigmoid(
                            model.decoder(latent_perturbed[i_perturbation])['x_hat_logits']
                        )
                    )
                x = torch.stack(x)
                image = einops.rearrange(x, 'v n c h w -> (n h) (v w) c')
            wandb.log({f'latent {i_latent} perturbations': wandb.Image(image.detach().cpu().numpy()), 'step': step})


def train():
    args = parse_command_line()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup wandb
    if args['use_wandb']:
        _setup_wandb(args, spec={})

    # load the shapes3D dataset
    dataset_cfg = {'seed': 0,
                   'possible_dirs': ['/home/sumeet/latent_quantization/data'],
                   'batch_size': 256,
                   'num_val_data': 10_000}
    dataset_metadata, train_set, val_set = get_datasets(dataset_cfg)

    # construct the model
    if args['bio_ae']:
        model = BioAE(obs_shape=(3, 64, 64), num_latents=10)
        model = model.to(device)
        ae_optim = Adam(model.parameters(), lr=1e-3)
        latent_optim = None
    else:
        model = QuantizedAutoEncoder(obs_shape=(3, 64, 64), num_latents=10, values_per_latent=10)
        model = model.to(device)
        # optimizer for encoder and decoder
        weight_decay = model.lambdas.get('l2', 0.0)
        ae_params = (list(model.encoder.parameters()) + list(model.decoder.parameters()))
        ae_optim = AdamW(ae_params, lr=1e-3, weight_decay=weight_decay)
        # optimizer for latent space
        latent_optim = Adam(model.latent.parameters(), lr=1e-3)

    def save_checkpoint(fp: str):
        torch.save({
            'model': model.state_dict(),
            'ae_optim': ae_optim.state_dict(),
            'latent_optim': latent_optim.state_dict() if not args['bio_ae'] else None,
        }, fp)

    def load_checkpoint(fp: str):
        checkpoint = torch.load(fp)
        model.load_state_dict(checkpoint['model'])
        ae_optim.load_state_dict(checkpoint['ae_optim'])
        latent_optim.load_state_dict(checkpoint['latent_optim']) if not args['bio_ae'] else None

    if args['resume'] is not None:
        cp = torch.load(args['resume'])
        model.load_state_dict(cp['model'])
        ae_optim.load_state_dict(cp['ae_optim'])

    # exp dir
    exp_dir = Path('./results/disentangle_shape3D/checkpoints')
    exp_dir.mkdir(exist_ok=True, parents=True)

    num_steps = 2e5
    checkpoint_n_steps = 5000
    eval_n_steps = 5000
    for step, batch in enumerate(tqdm(train_set, total=num_steps)):
        if step > num_steps:
            break

        if (step + 1) % checkpoint_n_steps == 0:
            cp_path = exp_dir / f'checkpoint_{step}.pt'
            save_checkpoint(str(cp_path))

        batch['x'] = torch.from_numpy(batch['x']).to(device)
        loss, aux = model.batched_loss(batch)
        ae_optim.zero_grad()
        if not args['bio_ae']:
            latent_optim.zero_grad()
        loss.backward()
        ae_optim.step()
        if not args['bio_ae']:
            latent_optim.step()

        if step == 0 or (step + 1) % eval_n_steps == 0 or ((step + 1 < eval_n_steps) and (step + 1) % (eval_n_steps // 10) == 0):
            with torch.no_grad():
                eval_fn = evaluate_bio_ae if args['bio_ae'] else evaluate
                eval_fn(val_set, model, step, use_wandb=args['use_wandb'])


if __name__ == '__main__':
    train()
