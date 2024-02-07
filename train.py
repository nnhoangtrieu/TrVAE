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
    annealer = cyclic_annealer(arg.kl_w_start, arg.kl_w_end, arg.epochs, arg.kl_cycle, arg.kl_ratio)
if arg.kl_type == 'monotonic' : 
    annealer = monotonic_annealer(arg.kl_start, arg.kl_w_start, arg.kl_w_end)
writer = SummaryWriter()

print('#########################################################################')
print('############################## TRAINING #################################')
print('#########################################################################')

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

    


    # Generating Molecules
    model.eval()
    gen_mol = torch.empty(0).to(device)
    with torch.no_grad() : 
        for _ in range(1) : 
            z = torch.randn(500, max_len, arg.d_latent).to(device)
            z_mask = torch.ones(500, 1, max_len).to(device)
            tgt = torch.zeros(500, 1, dtype=torch.long).to(device)

            for _ in range(max_len - 1) : 
                pred = model.inference(z, tgt, z_mask, get_mask(tgt, vocab).to(device))
                _, idx = torch.topk(pred, 1, dim=-1)
                idx = idx[:, -1, :]
                tgt = torch.cat([tgt, idx], dim=1)

            gen_mol = torch.cat([gen_mol, tgt], dim=0)
        gen_mol = gen_mol.tolist() 
        gen_mol = parallel_f(read_gen_smi, gen_mol)
        valid_mol = parallel_f(get_valid, gen_mol)
        valid_mol = [m for m in valid_mol if m != None]
        unique_mol = set(valid_mol)

        uniqueness = (len(unique_mol) / len(valid_mol)) * 100 if valid_mol else 0
        novel_mol = [m for m in parallel_f(get_novel, unique_mol) if m is not None]
        novelty = (len(novel_mol) / len(unique_mol)) * 100 if unique_mol else 0
        validity = (len(valid_mol) / 30000) * 100    


        with open(f'data/{arg}.txt', 'a') as file : 
            if epoch == 0 : 
                file.write('Model Parameters:\n')
                for name, value in arg.__dict__.items() : 
                    file.write(f'{name} : {value}\n')
            file.write(f"Epoch: {epoch + 1} --- Train Loss: {train_loss / len(train_loader):3f}\n")
            file.write(f'Validity: {validity:.2f}% --- Uniqueness: {uniqueness:.2f}% --- Novelty: {novelty:.2f}%')

            for i, m in enumerate(set(novel_mol)) : 
                file.write(f'{i+1}. {m}\n')



    writer.add_scalar('Train Loss', train_loss / len(train_loader), epoch)
    writer.add_scalar('Recon Loss', recon_loss / len(train_loader), epoch)
    writer.add_scalar('KL Loss', kl_loss / len(train_loader), epoch)
    writer.add_scalar('Validity', validity, epoch)
    writer.add_scalar('Uniqueness', uniqueness, epoch)
    writer.add_scalar('Beta', beta, epoch)
    
    writer.add_scalar('Metrics/Validity', validity, epoch)
    writer.add_scalar('Metrics/Uniqueness', uniqueness, epoch)
    print(f'Epoch: {epoch + 1}:')
    print(f'\tTrain Loss: {train_loss / len(train_loader):.3f} --- Reconstruction Loss: {recon_loss / len(train_loader):.3f} --- KL Loss: {kl_loss / len(train_loader):.3f} --- Beta: {beta:5f}')
    print(f'\tValidity: {validity:.2f}% --- Uniqueness: {uniqueness:.2f}% --- Novelty: {novelty:.2f}%\n')