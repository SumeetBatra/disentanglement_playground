import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import einops
import wandb

from collections import defaultdict
from disentangle.utils.metrics import log_reconstruction_metrics
from disentangle import model_factory
from common.utils import _setup_wandb, save_checkpoint
from common.utils import get_shapes3d_dataset
from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['beta_vae', 'bio_ae', 'qlae', 'slot_ae', 'beta_tcvae'])
    parser.add_argument("--train_n_steps", type=int, default=2e5)
    parser.add_argument("--eval_n_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_n_steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    # wandb params
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_project', type=str, default='disentangle')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default='qlae_shapes3D')
    parser.add_argument('--wandb_tag', type=str, default=None)

    return vars(parser.parse_args())


def evaluate(val_set, model, exp_dir: Path, step: int, use_wandb: bool = False):
    losses, final_aux = defaultdict(list), {}
    for batch in iter(val_set):
        batch['x'] = torch.from_numpy(batch['x']).to(device)
        _, aux = model.batched_loss(batch)
        aux['x_true'] = batch['x']
        aux['x_hat_logits'] = aux['outs']['x_hat_logits']

        for k, v in aux['metrics'].items():
            losses[f'eval/{k}'].append(v)

        final_aux = aux

    log_reconstruction_metrics(final_aux, step, use_wandb=use_wandb)

    if use_wandb:
        for k, v in losses.items():
            losses[k] = sum(v) / len(v)
        wandb.log(losses, step=step)

    # traverse individual latent dimensions and visualize the results
    num_samples = num_perturbations = 16
    for latent_i in range(model.num_latents):
        print(f'Interpolating {num_perturbations} samples for latent {latent_i}')
        latent_mins = aux['outs']['z_hat'].min(dim=0).values
        latent_maxs = aux['outs']['z_hat'].max(dim=0).values
        latent_perturbed = torch.tile(aux['outs']['z_hat'][:num_samples], (num_perturbations, 1, 1))
        latent_perturbed[:, :, latent_i] = torch.linspace(latent_mins[latent_i], latent_maxs[latent_i],
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
        if use_wandb:
            wandb.log({f'latent {latent_i} perturbations': wandb.Image(image.detach().cpu().numpy()), 'step': step})
        else:
            # save locally
            image_dir = exp_dir / f'images/step_{step}/'
            image_dir.mkdir(parents=True, exist_ok=True)
            img_path = image_dir / f'{latent_i}.png'
            image = TF.to_pil_image(image.permute(2, 0, 1).detach().cpu())
            image.save(str(img_path))


def train():
    args = parse_command_line()

    # setup wandb
    if args['use_wandb']:
        _setup_wandb(args)

    # shapes3D dataset
    dataset_cfg = {'seed': args['seed'],
                   'possible_dirs': ['/home/sumeet/latent_quantization/data'],
                   'batch_size': args['batch_size'],
                   'num_val_data': 10_000}
    train_set, val_set = get_shapes3d_dataset(dataset_cfg)

    # construct the model
    model_key = args['model']
    model, model_optim, latent_optim = model_factory(model_key)
    model = model.to(device)
    exp_dir = Path(f'./results/{model_key}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    num_steps = args['train_n_steps']
    eval_n_steps = args['eval_n_steps']
    for step, batch in enumerate(tqdm(train_set, total=num_steps)):
        if step > num_steps:
            break

        if (step + 1) % args['checkpoint_n_steps'] == 0:
            cp_dir = exp_dir / f'checkpoints/'
            cp_dir.mkdir(parents=True, exist_ok=True)
            cp_path = cp_dir / f'checkpoint_{step + 1}.pt'
            save_checkpoint(str(cp_path), model, model_optim, latent_optim)

        batch['x'] = torch.from_numpy(batch['x']).to(device)
        loss, aux = model.batched_loss(batch)

        model_optim.zero_grad()
        if latent_optim:
            latent_optim.zero_grad()
        loss.backward()
        model_optim.step()
        if latent_optim:
            latent_optim.step()

        if step == 0 or (step + 1) % eval_n_steps == 0 or \
            ((step + 1 < eval_n_steps) and (step + 1) % (eval_n_steps // 10) == 0):
            with torch.no_grad():
                evaluate(val_set, model, exp_dir, step, use_wandb=args['use_wandb'])


if __name__ == '__main__':
    train()


