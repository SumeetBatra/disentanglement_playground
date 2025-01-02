import wandb
import torch

from typing import Dict, Any


def _setup_wandb(args: Dict[str, Any], spec: Dict[str, Any]) -> None:
    """Sets up wandb experiment tracking if enabled"""
    run_name = args['wandb_run_name']
    wandb.init(
        project=args['wandb_project'],
        entity=args['wandb_entity'],
        group=args['wandb_group'],
        name=run_name,
        tags=[args['wandb_tag']],
        config=spec
    )


def save_checkpoint(fp: str, model, ae_optim, latent_optim):
    torch.save({
        'model': model.state_dict(),
        'ae_optim': ae_optim.state_dict(),
        'latent_optim': latent_optim.state_dict()
    }, fp)


def load_checkpoint(fp: str, model, ae_optim, latent_optim):
    checkpoint = torch.load(fp)
    model.load_state_dict(checkpoint['model'])
    ae_optim.load_state_dict(checkpoint['ae_optim'])
    latent_optim.load_state_dict(checkpoint['latent_optim'])