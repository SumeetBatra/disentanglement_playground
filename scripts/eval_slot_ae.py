import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from disentangle.autoencoder.slot_ae import SlotAutoEncoder
from disentangle.datasets.shapes3d import get_datasets


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)

    return vars(parser.parse_args())


def enjoy(args):
    # load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cp_path = args['checkpoint_path']
    num_slots = 10
    model = SlotAutoEncoder(obs_shape=(3, 64, 64), num_latents=num_slots, slot_dim=64).to(device)
    model.load_state_dict(torch.load(cp_path)['model'])

    # load the dataset
    dataset_cfg = {'seed': 0,
                   'possible_dirs': ['/home/sumeet/latent_quantization/data'],
                   'batch_size': 16,
                   'num_val_data': 10_000}
    dataset_metadata, train_set, val_set = get_datasets(dataset_cfg)

    batch = next(iter(val_set))
    batch['x'] = torch.from_numpy(batch['x']).to(device).float()
    outs = model(batch['x'])
    recon, masks = outs['x_hat_logits'], outs['masks']
    sample_idx = torch.randint(batch['x'].shape[0], (1,)).item()

    fig, axs = plt.subplots(2, (num_slots + 2) // 2, figsize=(15, 15))
    axs = axs.flatten()
    axs[0].imshow(batch['x'][sample_idx].permute(1, 2, 0).detach().cpu().numpy())
    recon_img = F.sigmoid(recon[sample_idx])
    axs[1].imshow(recon_img.permute(1, 2, 0).detach().cpu().numpy())
    axs[0].set_title('Original')
    axs[1].set_title('Recon')

    for slot_i in range(num_slots):
        mask = masks[sample_idx, slot_i]
        img = F.sigmoid(recon[sample_idx] * mask + (1.0 - mask))
        axs[slot_i + 2].imshow(img.permute(1, 2, 0).detach().cpu().numpy())

    plt.show()


if __name__ == '__main__':
    args = parse_command_line()
    enjoy(args)