# TrVAE: Transformer Based VAE for Molecule Generation

This repo contains the PyTorch implementation of VAE with Transformer Architecture for molecular design. The power of Transformer combined with different KL Annealling schedule helps to explore the chemical space. The code is organized by folders that correspond to the following sections: 
- **data**: contain dataset from ChemBL (1.4M)
- **genmol**: contain generated molecules from trained model via generate.py
- **genmol_train**: generated molecules generated after each training epoch
- **model**: architecture of models
- **tensorboard**: results data for analysis


## Training
The default configuration of the model is: 
**d_model**: 512 | **d_latent**: 256 | **d_ff**: 1024 | **num_head**: 8 | **num_layer**: 8 | **dropout**: 0.5 | **lr**: 0.0003 | **epochs**: 32 | **batch_size**: 128 | **max_len**: 30 | **kl_type**: monotonic | **kl_start**: 0 | **kl_w_start**: 0 | **kl_w_end**: 0.0003 | **kl_ratio**: 0.9 | **name_checkpoint**: model | **epoch_checkpoint**: -1 

- The dimension of model (d_model) will be use throughout the Transformer Layer and  will be increased in the inner Feed Forward Layer (d_ff). Encoder will finally compress the the input to latent space (d_latent). 

You can retrain the model with the default configuration with a command

```bash
python train.py

```