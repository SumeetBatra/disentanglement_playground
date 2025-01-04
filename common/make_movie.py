import torch
import torch.nn.functional as F

from torchvision.transforms import ToPILImage
from disentangle import model_factory
from common.utils import get_shapes3d_dataset
from pathlib import Path


def make_gif(model, batch, save_path: str):
    '''
    String a sequence of latent traversals together into a gif
    '''

    num_perturbations, sample_idx = 20, torch.randint(0, len(batch), (1, ))
    outs = model(batch)
    latent_inds = [2, 5, 3, 8]
    x = []

    latent_mins = outs['z_hat'].min(dim=0).values
    latent_maxs = outs['z_hat'].max(dim=0).values
    prev_latents = []
    for latent_i in latent_inds:
        latent_perturbed = torch.tile(outs['z_hat'][sample_idx], (num_perturbations, 1, 1))
        latent_perturbed[:, :, latent_i] = torch.linspace(latent_mins[latent_i], latent_maxs[latent_i],
                                                          num_perturbations)[:, None]
        for prev_latent_i in prev_latents:
            latent_perturbed[:, :, prev_latent_i] = torch.Tensor([latent_maxs[prev_latent_i]] * num_perturbations).cuda()[:, None]
        prev_latents.append(latent_i)

        with torch.no_grad():
            for i_perturbation in range(num_perturbations):
                x.append(
                    F.sigmoid(
                        model.decoder(latent_perturbed[i_perturbation])['x_hat_logits']
                    )
                )
    x = torch.stack(x)
    frames = []
    to_pil = ToPILImage()
    for img in x:
        frames.append(to_pil(img.squeeze()))

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=1,
        loop=0,
    )


if __name__ == "__main__":
    model, *_ = model_factory('qlae')
    model = model.cuda()
    cp_path = 'results/qlae/checkpoint_200000.pt'
    model.load_state_dict(torch.load(cp_path)['model'])
    model.eval()
    save_dir = Path('videos')
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / 'qlae_shapes3D_animated.gif'
    train_set, _ = get_shapes3d_dataset()

    batch = next(iter(train_set))
    batch = torch.from_numpy(batch['x']).float().cuda()
    make_gif(model, batch, str(save_path))

