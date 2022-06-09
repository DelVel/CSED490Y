from __future__ import division
import argparse
import math
import os
import random
import time
from collections import OrderedDict
import dill as pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset
from tqdm import tqdm
import sys
sys.path.append("/content/drive/My Drive/attention-is-all-you-need-pytorch")
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


import time
#import pickle
import gzip
from random import randint
from scipy import misc
from scipy import special
import matplotlib.pyplot as plt
# Loss Surface Visualization
from mpl_toolkits.mplot3d import Axes3D
import collections
#%matplotlib inline 
import gc
import pandas as pd

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    """ Apply label smoothing if needed """

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def eval_epoch(model, validation_data, device, opt):
    """ Epoch operation in evaluation phase """

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq = patch_src(batch.src).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


'''
CUDA_VISIBLE_DEVICES=8,9 python train.py -data_pkl m30k_deen_shr.pkl -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -x 1 -y 1 -b 192 -warmup 4000 -epoch 100 -lr_mul 0.5

'''
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default="m30k_deen_shr.pkl")  # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)  # bpe encoded data
    parser.add_argument('-val_path', default=None)  # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', default=True)
    parser.add_argument('-proj_share_weight', default=True)
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default='output')
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'last'], default='last')

    parser.add_argument('-no_cuda', default=False)
    parser.add_argument('-label_smoothing', default=True)

    parser.add_argument('-use_ckpt', default=None)
    parser.add_argument('-scale', type=float, default=1.0)
    parser.add_argument('-grid_num', type=int, default=10)
    parser.add_argument('-csv', default = 'csv_test.csv')
    #parser.add_argument('-range', type=float, default=1.0)
    parser.add_argument('-x', type=float, default=None)
    parser.add_argument('-y', type=float, default=None)

    opt = parser.parse_args()
    if opt.use_ckpt is not None:
        ckpt = torch.load(opt.use_ckpt)
        opt = ckpt['settings']
        global epoch, model_dict
        epoch = ckpt['epoch']
        model_dict = ckpt['model']

    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not get_output_dir(opt):
        print('No experiment result will be saved.')
        raise ValueError()

    if not os.path.exists(get_output_dir(opt)):
        os.makedirs(get_output_dir(opt))

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'
              'Using smaller batch w/o longer warmup may cause '
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    # ========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise ValueError()

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

    ms = np.linspace(-1.0, 1.0, opt.grid_num)
    bs = np.linspace(-1.0, 1.0, opt.grid_num)

    
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    
    fvec, svec = get_two_rand_vector(transformer)

    M, B = np.meshgrid(ms, bs)
    loss_list =[]

    for alpha, beta in zip(np.ravel(M), np.ravel(B)) :

        new_transformer = change_param2(transformer, alpha, beta, fvec, svec, scale=opt.scale)
        val_loss, val_acc = eval_epoch(new_transformer, validation_data,  device, opt)

        del new_transformer
        gc.collect()
        torch.cuda.empty_cache()
        print("alpha : {0:0.2f}, beta : {1:0.2f}, loss : {2:0.3f}, acc : {3:0.3f}".format(alpha, beta, val_loss,val_acc))
        loss_list.append(val_loss)
    print(loss_list)
    zs = np.array(loss_list)
    save_csv( opt.csv, M, B, loss_list )

    Z = zs.reshape(M.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(M, B, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.show()

def save_csv( save_dir, M, B, loss_list ):
    data = {
        'id':[],
        'x':[],
        'y':[],
        'loss':[]        
        }
    id_ = 0
    for alpha, beta in zip(np.ravel(M), np.ravel(B)) :
        data['id'].append(0)
        data['x'].append(alpha)
        data['y'].append(beta)
        data['loss'].append(loss_list[id_])
        id_ = id_+1
    df = pd.DataFrame(data)
    df.to_csv(save_dir, index=False)
    print(" {0} has been saved !".format(save_dir) )     
        

def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    if not opt.embs_share_weight:
        raise ValueError()

    data = pickle.load(open(opt.data_pkl, 'rb'))
    max_len = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= max_len and len(vars(x)['trg']) <= max_len

    train_ = TranslationDataset(
        fields=fields,
        path=opt.train_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = max_len + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train_, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    # ========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}

    train_ = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train_, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator



def change_param(model, alpha, beta, scale=1):
    from torch.nn.utils import prune
    pass1_w = torch.load('1passnoprune/model.chkpt')['model']
    model.load_state_dict(pass1_w)
    
    with torch.no_grad():
        state_dict = model.state_dict()
        #print(state_dict)
        for a in state_dict:
            shp = state_dict[a].shape
            #print(shp)
            first_random_vector = alpha * torch.rand(shp).cuda() * scale
            second_random_vector = beta * torch.rand(shp).cuda() * scale
            state_dict[a] = state_dict[a] + first_random_vector + second_random_vector
            #print("state {0} has been updated with scale {1}".format(a, scale))
        model.load_state_dict(state_dict)
    return model

def change_param2(model, alpha, beta, fvec, svec, scale=1):
    pass1_w = torch.load('1passnoprune/model.chkpt')['model']
    model.load_state_dict(pass1_w)
    
    with torch.no_grad():
        state_dict = model.state_dict()
        for a in state_dict:
            shp = state_dict[a].shape
            #print(shp)
            first_random_vector = alpha * fvec[a].cuda() * scale
            second_random_vector = beta * svec[a].cuda() * scale
            state_dict[a] = state_dict[a] + first_random_vector + second_random_vector
            #print("state {0} has been updated with scale {1}".format(a, scale))
        model.load_state_dict(state_dict)
    return model

def get_two_rand_vector(model):
    from torch.nn.utils import prune
    pass1_w = torch.load('1passnoprune/model.chkpt')['model']
    model.load_state_dict(pass1_w)

    first_vec = {}
    second_vec = {}
    with torch.no_grad():
        state_dict = model.state_dict()
        for a in state_dict:
            shp = state_dict[a].shape
            first_vec[a] = torch.rand(shp).cuda()
            second_vec[a] = torch.rand(shp).cuda()
    return first_vec, second_vec
    

def get_output_dir(opt):
    return f'{opt.x},{opt.y}'

if __name__ == '__main__':
    epoch = None
    model_dict = None

    # get_dir_vector()
    main()

