import argparse

from tqdm import tqdm
from disentangle.utils.metrics import *
from disentangle import model_factory
from pathlib import Path
from common.utils import _setup_wandb, save_checkpoint, get_shapes3d_dataset
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_n_steps', type=int, default=2e5)
    parser.add_argument('--checkpoint_n_steps', type=int, default=5_000)
    parser.add_argument('--eval_n_steps', type=int, default=5_000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    # wandb params
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_project', type=str, default='disentangle')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default='slot_ae_shapes3D')
    parser.add_argument('--wandb_tag', type=str, default='SlotAE')

    return vars(parser.parse_args())


def evaluate(val_set, model, step: int, use_wandb: bool = False):
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

        # visualize the different slots for a batch of samples
        num_samples, num_slots = 16, model.num_slots

        # get the slots, reconstructions, and masks
        outs = aux['outs']
        slots, x_hat_logits, masks = outs['slots'][:num_samples], outs['x_hat_logits'][:num_samples], outs['masks'][:num_samples]
        imgs = [F.sigmoid(x_hat_logits[:num_samples])]
        for slot_i in range(num_slots):
            mask_i = masks[:, slot_i]
            img = F.sigmoid(x_hat_logits * mask_i + (1.0 - mask_i))
            imgs.append(img)
        imgs = torch.stack(imgs)
        image = einops.rearrange(imgs, 's b c h w -> (b h) (s w) c')
        wandb.log({f'Slot Reconstructions': wandb.Image(image.detach().cpu().numpy()), 'step': step})


def train(args):
    # setup wandb
    if args['use_wandb']:
        _setup_wandb(args)

    # load the shapes3D dataset
    dataset_cfg = {'seed': args['seed'],
                   'batch_size': args['batch_size'],
                   'num_val_data': 2500}
    train_set, val_set = get_shapes3d_dataset(dataset_cfg)

    # construct the model
    model, model_optim, latent_optim = model_factory('slot_ae')
    model = model.to(device)

    # exp dir
    exp_dir = Path('./results/disentangle_shape3D/checkpoints')
    exp_dir.mkdir(exist_ok=True, parents=True)

    num_steps = args['train_n_steps']
    checkpoint_n_steps = args['checkpoint_n_steps']
    eval_n_steps = args['eval_n_steps']
    for step, batch in enumerate(tqdm(train_set, total=num_steps)):
        if step > num_steps:
            break

        if (step + 1) % checkpoint_n_steps == 0:
            cp_path = exp_dir / f'checkpoint_{step}.pt'
            save_checkpoint(str(cp_path), model, model_optim, latent_optim)

        batch['x'] = torch.from_numpy(batch['x']).to(device)
        loss, aux = model.batched_loss(batch)
        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        if step == 0 or (step + 1) % eval_n_steps == 0 or (
                (step + 1 < eval_n_steps) and (step + 1) % (eval_n_steps // 10) == 0):
            with torch.no_grad():
                evaluate(val_set, model, step, use_wandb=args['use_wandb'])


if __name__ == '__main__':
    args = parse_command_line()
    train(args)
