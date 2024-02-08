import model.base 
from model.base import Transformer
import argparse
import torch
import datetime
import utils 
from utils import *

cur_time = datetime.datetime.now()
def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = Transformer(d_model=arg.d_model,
                    d_latent=arg.d_latent,
                    d_ff=arg.d_ff,
                    num_head=arg.num_head,
                    num_layer=arg.num_layer,
                    dropout=arg.dropout,
                    vocab=vocab).to(device)
model.load_state_dict(torch.load(f'checkpoint/{arg.checkpoint_name}/e{arg.epoch_checkpoint}_{arg.checkpoint_name}.pth'))
model.eval()


gen_mol = torch.empty(0).to(device)
with torch.no_grad() : 
    for _ in range(60) : 
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
    # valid_mol = parallel_f(get_valid, gen_mol)
    # valid_mol = [m for m in valid_mol if m != None]
    # unique_mol = set(valid_mol)

    # uniqueness = (len(unique_mol) / len(valid_mol)) * 100 if valid_mol else 0
    # novel_mol = [m for m in parallel_f(get_novel, unique_mol) if m is not None]
    # novelty = (len(novel_mol) / len(unique_mol)) * 100 if unique_mol else 0
    # validity = (len(valid_mol) / 30000) * 100    


    with open(f'genmol/{cur_time}.txt', 'a') as file : 
        file.write('Model Parameters:\n')
        for name, value in arg.__dict__.items() : 
            file.write(f'{name} : {value}\n')
        for i, m in enumerate(gen_mol) : 
            file.write(f'{i+1}. {m}\n')