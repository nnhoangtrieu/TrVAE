# TrVAE: Transformer Based VAE for Molecule Generation

This repo contains the PyTorch implementation of VAE with Transformer Architecture for molecular design. The power of Transformer combined with different KL Annealling schedule helps to explore the chemical space. The code is organized by folders that correspond to the following sections: 
- data: contain dataset from ChemBL (1.4M)
- genmol: contain generated molecules from trained model via generate.py
- genmol_train: generated molecules generated after each training epoch
- model: architecture of models
- tensorboard: results data for analysis
