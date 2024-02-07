import random 
import os 
import numpy as np 
import torch 
import re 
import torch
import torch.nn.functional as F
import multiprocessing
import pickle
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('--d_model', type=int, default=512, help='model dimension')
parser.add_argument('--d_latent', type=int, default=256, help='latent dimension')
parser.add_argument('--d_ff', type=int, default=1024, help='feed forward dimension')
parser.add_argument('--num_head', type=int, default=8, help='number of attention heads')
parser.add_argument('--num_layer', type=int, default=8, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--epochs', type=int, default=32, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_len', type=int, default=30, help='longest length of input for dataset')
parser.add_argument('--kl_type', type=str, default='cyclic', help='KL type: monotonic/cyclic')
parser.add_argument('--kl_start', type=int, default=0, help='for monotonic, at which epoch to start increasing beta')
parser.add_argument('--kl_w_start', type=float, default=0, help='longest length of input for dataset')
parser.add_argument('--kl_w_end', type=float, default=0.0003, help='longest length of input for dataset')
parser.add_argument('--kl_cycle', type=int, default=4, help='longest length of input for dataset')
parser.add_argument('--kl_ratio', type=float, default=0.9, help='longest length of input for dataset')

arg = parser.parse_args()

print('\nModel Parameters:')
for name, value in arg.__dict__.items():
    print(f'\t{name}: {value}')

def seed_torch(seed=910):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def tokenizer(smile):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens


def get_vocab(smi_list) :
    dic = {'<START>': 0, '<END>': 1, '<PAD>': 2}
    for smi in smi_list :
        for char in smi :
            if char not in dic :
                dic[char] = len(dic) 
    return dic 


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def get_mask( target, smi_dic) :
        mask = (target != smi_dic['<PAD>']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)


def pad(smi, max_len) :
    return smi + [2] * (max_len - len(smi))


def encode(smi, vocab) :
    return [0] + [vocab[char] for char in smi] + [1]


def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)

def monotonic_annealer(n_epoch, kl_start, kl_w_start, kl_w_end):
    i_start = kl_start
    w_start = kl_w_start
    w_max = kl_w_end

    inc = (w_max - w_start) / (n_epoch - i_start)

    annealing_weights = []
    for i in range(n_epoch):
        k = (i - i_start) if i >= i_start else 0
        annealing_weights.append(w_start + k * inc)

    return annealing_weights

def cyclic_annealer(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch) * stop
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['<PAD>'])
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()).mean() / arg.batch_size
    return  reconstruction_loss + kl_loss * beta, reconstruction_loss, kl_loss




class MyDataset(torch.utils.data.Dataset) :
    def __init__(self, token_list) :
        self.token_list = token_list

    def __len__(self) :
        return len(self.token_list)

    def __getitem__(self, idx) :   
        return torch.tensor(self.token_list[idx], dtype=torch.long)
    






with open('./data/chembl24_canon_train.pickle','rb') as file :
    smi_list = pickle.load(file) 
    smi_list = [smi for smi in smi_list if len(smi) <= arg.max_len]
    token_list = [tokenizer(s) for s in smi_list]
    vocab = get_vocab(token_list)
    inv_vocab = {v: k for k, v in vocab.items()}
    token_list = [encode(t, vocab) for t in token_list]
    max_len = len(max(token_list, key=len))
    token_list = [pad(t, max_len) for t in token_list]

    print(f'\nNumber of data: {len(smi_list)}\n')

dataset = MyDataset(token_list)