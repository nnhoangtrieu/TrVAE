# TrVAE: Transformer Based VAE for Molecule Generation

This repo contains the PyTorch implementation of VAE with Transformer Architecture for molecular design. The power of Transformer combined with different KL Annealling schedule helps to explore the chemical space. The code is organized by folders that correspond to the following sections: 
- **data**: contain dataset from ChemBL (1.4M)
- **genmol**: contain generated molecules from trained model via generate.py
- **genmol_train**: generated molecules generated after each training epoch
- **model**: architecture of models
- **tensorboard**: results data for analysis


## Training
The default configuration of the model is:

**d_model**: 512 | **d_latent**: 256 | **d_ff**: 1024 | **num_head**: 8 | **num_layer**: 8 | **dropout**: 0.5 | **lr**: 0.0003 | **epochs**: 32 | **batch_size**: 128 | **max_len**: 30 | **kl_type**: monotonic | **kl_start**: 0 | **kl_w_start**: 0 | **kl_w_end**: 0.0003 | **kl_cycle**: 4 | **kl_ratio**: 0.9 | **name_checkpoint**: model | **epoch_checkpoint**: -1 

- The dimension of model (d_model) will be use throughout the Transformer Layer and  will be increased in the inner Feed Forward Layer (d_ff). Encoder will finally compress the the input to latent space (d_latent). 

You can retrain the model with the default configuration with a command

```bash
python train.py
```

The model is trained with SMILES that has length < 30 (max_len = 30) (~100k data). You can also retrain with default configuration but different maximum length by
```bash
python train.py --max_len 40 
```

### KL Annealling 
The biggest problem that every VAE architecture faces is KL Vanishing or Posterior Collapse. TrVAE implements 2 types of β scheduling:
![beta schedule](https://github.com/nnhoangtrieu/TrVAE/assets/107376082/34f68cc1-2cea-424a-97a2-4229119ef436)

- monotonic: β is increased monotonically throughout each epoch 
    - **kl_start**: epoch to start increasing β 
    - **kl_w_start**: starting value for β at the begin of training 
    - **kl_w_end**: highest value β can reach at the end of training

```bash
python train.py --kl_type "monotonic" --kl_start 5 --kl_w_start 0 --kl_w_end 0.003
```

- cyclic: β is increased and dropped each cycle 
    - **kl_w_start**: starting value for β at begin of each cycle
    - **kl_w_end**: highest value β can reach at the end of each cycle 
    - **kl_cycle**: the number of cycle during training 
    - **kl_ratio**: (should be between 0 and 1) how fast β is increased. Smaller β will increase the number of epoch β stays at the maximum (kl_w_end)

```bash
python train.py --kl_type "cyclic" --kl_w_start 0 --kl_w_end 0.0003 --kl_cycle 8 --kl_ratio 0.9
```

## Generate Molecules
You can look at folder **gen_train** to look at the recorded validity, uniqueness, and novelty the generated molecules, but most importantly is to look at the generate molecules themselves to really judge the performance of the model at each epoch. You can then specify the checkpoint of the model at which epoch to generate molecules. The generated molecules will be save in folder **genmol**

```bash
python generate.py --name_checkpoint "your_name_checkpoint" --epoch_checkpoint 5
```






