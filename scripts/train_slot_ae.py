import numpy as np
import torch
import argparse
import wandb
import einops
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from tqdm import tqdm
from disentangle.datasets.shapes3d import get_datasets
from disentangle.autoencoder.slot_ae import SlotAutoEncoder
from disentangle.utils.metrics import *
from pathlib import Path
from distutils.util import strtobool
from typing import Dict, Any
from common.utils import _setup_wandb, save_checkpoint, load_checkpoint


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_project', type=str, default='disentangle')
    parser.add_argument('--wandb_group', type=str, default='disentangle')
    parser.add_argument('--wandb_run_name', type=str, default='slot_ae_shapes3D')
    parser.add_argument('--wandb_tag', type=str, default='SlotAE')

    return vars(parser.parse_args())


def tensor_linspace(start: torch.Tensor, end: torch.Tensor, n_steps: int):
    steps = torch.arange(n_steps, dtype=torch.float32, device=start.device)
    step_size = (end - start) / (n_steps - 1)
    return start + steps[:, None].repeat(1, start.size(-1)) * step_size


def evaluate(val_set, model, step: int, use_wandb: bool = False):
    loss, bce_loss = [], []
    final_aux = {}
    for batch in iter(val_set):
        batch['x'] = torch.from_numpy(batch['x']).to('cuda')
        _, aux = model.batched_loss(batch)
        aux['x_true'] = batch['x']
        aux['z_true'] = batch['z']
        aux['z_hat'] = aux['outs']['z_hat']
        aux['x_hat_logits'] = aux['outs']['x_hat_logits']

        loss.append(aux['metrics']['loss'])
        bce_loss.append(aux['metrics']['bce_loss'])

        final_aux = aux

    if use_wandb:
        wandb.log({
            'eval/total_loss': sum(loss) / len(loss),
            'eval/bce_loss': sum(bce_loss) / len(bce_loss),
        }, step=step)
    log_reconstruction_metrics(final_aux, step, use_wandb=use_wandb)

    # if use_wandb:
    #     num_samples = 16
    #     num_perturbations = 16
    #     # generate perturbations to the latents and see what happens
    #     latent_mins = aux['outs']['z_hat'].min(dim=0).values
    #     latent_maxs = aux['outs']['z_hat'].max(dim=0).values
    #     for i_latent in range(model.num_latents):
    #         latent_perturbed = torch.tile(aux['outs']['z_hat'][:num_samples],
    #                                       (num_perturbations, 1, 1, 1))  # (num_perturbations, num_samples, num_latents, latent_dim)
    #         latent_perturbed[:, :, i_latent, :] = tensor_linspace(latent_mins[i_latent], latent_maxs[i_latent],
    #                                                           num_perturbations)
    #
    #         x = []
    #         with torch.no_grad():
    #             for i_perturbation in range(num_perturbations):
    #                 x.append(
    #                     F.sigmoid(
    #                         model.decoder(latent_perturbed[i_perturbation])
    #                     )
    #                 )
    #             x = torch.stack(x)
    #             image = einops.rearrange(x, 'v n c h w -> (n h) (v w) c')
    #         wandb.log({f'latent {i_latent} perturbations': wandb.Image(image.detach().cpu().numpy()), 'step': step})


def train():
    args = parse_command_line()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup wandb
    if args['use_wandb']:
        _setup_wandb(args, spec={})

    # load the shapes3D dataset
    dataset_cfg = {'seed': 0,
                   'possible_dirs': ['/home/sumeet/latent_quantization/data'],
                   'batch_size': 16,
                   'num_val_data': 10_000}
    dataset_metadata, train_set, val_set = get_datasets(dataset_cfg)

    # construct the model
    model = SlotAutoEncoder(obs_shape=(3, 64, 64), num_latents=10, slot_dim=64)
    model = model.to(device)
    ae_params = (list(model.encoder.parameters()) + list(model.decoder.parameters()))
    ae_optim = AdamW(ae_params, lr=3e-4, weight_decay=0.1)
    # optimizer for latent space
    latent_optim = Adam(model.latent.parameters(), lr=3e-4)

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
            save_checkpoint(str(cp_path), model, ae_optim, latent_optim)

        batch['x'] = torch.from_numpy(batch['x']).to(device)
        loss, aux = model.batched_loss(batch)
        ae_optim.zero_grad()
        latent_optim.zero_grad()
        loss.backward()
        ae_optim.step()
        latent_optim.step()

        if step == 0 or (step + 1) % eval_n_steps == 0 or (
                (step + 1 < eval_n_steps) and (step + 1) % (eval_n_steps // 10) == 0):
            with torch.no_grad():
                evaluate(val_set, model, step, use_wandb=args['use_wandb'])


if __name__ == '__main__':
    train()
