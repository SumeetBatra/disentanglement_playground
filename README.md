# Disentanglement Playground

<div align="center">
  <img src=./media/qlae_shapes3D_animated.gif alt="Shapes3D Animated" height="100" width="100" />
</div>

A simple playground for various unsupervised latent disentanglement methods.
This repo is aimed to be research friendly and easy to understand rather than optimized for performance.


Currently, the following models are implemented: 
- Quantized Latent Autoencoder (QLAE) ([Disentanglement via Latent Quantization](https://arxiv.org/abs/2305.18378))
- Slot Attention ([Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055))
- Beta-TCVAE ([Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942))
- Beta-VAE ([Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://arxiv.org/pdf/1804.03599))
- BioAE ([Disentanglement with Biological Constraints: A Theory of Functional Cell Types](https://arxiv.org/abs/2210.01768))

## Installation

Create a conda env (or whatever virtual env you prefer):

`conda create --name disentangle python=3.10`

Install the requirements:

`pip install -r requirements.txt`

Follow the instructions [here](https://github.com/google-deepmind/3d-shapes) to download the Shapes3D dataset. 
Put it in a folder named "datasets" in the root of this repo i.e.`disentanglement_playground/datasets/`. 
I've assumed that the repo is installed in your $HOME directory. If that's not the case, you'll have to point it to 
wherever you have downloaded it (change ```possible_dirs=[...]``` in `common/shapes3d.py`)


## Running Experiments 
At the bare minimum, you just need to specify which model you want to run! 
For example, to run QLAE: 

`python -m scripts.train --model=qlae --use_wandb`

With Weights and Biases enabled, intermediate result visualizations will be automatically logged with your experiment.
If you don't want to use Weights and Biases, the images will be saved locally to `results/{MODEL}/images`.
Training takes about 1.5 hours on an RTX 3090. 



## Acknowledgements 
Much of the infrastructure code and the QLAE model itself is adapted from 
https://github.com/kylehkhsu/latent_quantization
and rewritten in PyTorch. 
Models that were adapted from other repos will have links to the original implementations in their 
respective files.

