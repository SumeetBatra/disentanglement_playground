import wandb
import torch

from typing import Dict, Any
from common.shapes3d import get_datasets


def _setup_wandb(args: Dict[str, Any]) -> None:
    """Sets up wandb experiment tracking if enabled"""
    run_name = args['wandb_run_name']
    wandb.init(
        project=args['wandb_project'],
        entity=args['wandb_entity'],
        group=args['wandb_group'],
        name=run_name,
    )


def save_checkpoint(fp: str, model, optim, latent_optim=None):
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'latent_optim': latent_optim.state_dict() if latent_optim is not None else None,
    }, fp)


def load_checkpoint(fp: str, model, optim, latent_optim=None):
    checkpoint = torch.load(fp)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    if latent_optim is not None:
        latent_optim.load_state_dict(checkpoint['latent_optim'])
    return model, optim, latent_optim


def get_shapes3d_dataset(dataset_cfg: Dict[str, Any]):
    dataset_metadata, train_set, val_set = get_datasets(dataset_cfg)
    return train_set, val_set