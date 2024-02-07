import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import numpy as np
import rdkit 
from rdkit.Chem import MolFromSmiles as get_mol
import utils
from utils import *
import model.base 
from model.base import Transformer


rdkit.rdBase.DisableLog('rdApp.*') # Disable rdkit warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 
def get_valid(smi) : 
    return smi if get_mol(smi) else None 
def get_novel(smi) : 
    return smi if smi not in smi_list else None 


train_set, val_set = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_set, batch_size=arg.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=arg.batch_size, shuffle=True)


model = Transformer(d_model=arg.d_model,
                    d_latent=arg.d_latent,
                    d_ff=arg.d_ff,
                    num_head=arg.num_head,
                    num_layer=arg.num_layer,
                    dropout=arg.dropout,
                    vocab=vocab).to(device)
optim = torch.optim.Adam(model.parameters(), lr = arg.lr, weight_decay=1e-6)

if arg.kl_type == 'cyclic' : 
    annealer = cyclic_annealer(arg.kl_w_start, arg.kl_w_end, arg.kl_cycle, arg.kl_ratio)
if arg.kl_type == 'monotonic' : 
    annealer = monotonic_annealer(arg.kl_start, arg.kl_w_start, arg.kl_w_end)

for epoch in range(arg.epochs) : 
    train_loss, val_loss, recon_loss, kl_loss = 0, 0, 0, 0
    beta = annealer[epoch]

    model.train() 

    for src in train_loader : 
        src = src.to(device) 
        src_mask = (src != vocab['<PAD>']).unsqueeze(-2)
        tgt = src.clone()
        tgt_mask = get_mask(tgt[:, :-1], vocab) 

        pred, mu, sigma = model(src, tgt[:, :-1], src_mask, tgt_mask)

        loss, recon, kl = loss_fn(pred, tgt[:, 1:], mu, sigma, beta)
        train_loss += loss.detach().item() 
        recon_loss += recon.detach().item()
        kl_loss += kl.detach().item()

        loss.backward(), optim.step(), optim.zero_grad(), clip_grad_norm_(model.parameters(), 5)

    
