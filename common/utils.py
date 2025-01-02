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


def save_checkpoint(fp: str, model, optim):
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
    }, fp)


def load_checkpoint(fp: str, model, optim):
    checkpoint = torch.load(fp)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])


def get_shapes3d_dataset():
    dataset_cfg = {'seed': 0,
                   'possible_dirs': ['/Users/sumeetbatra/disentanglement_playground/data'],
                   'batch_size': 256,
                   'num_val_data': 10_000}
    dataset_metadata, train_set, val_set = get_datasets(dataset_cfg)
    return train_set, val_set