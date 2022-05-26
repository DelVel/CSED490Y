"""
This script handles the training process.
"""

import argparse
import math
import os
import random
import time
from collections import OrderedDict

import dill as pickle
import numpy as np
# noinspection PyPackageRequirements
import torch
# noinspection PyPep8Naming,PyPackageRequirements
import torch.nn.functional as F
# noinspection PyPackageRequirements
import torch.optim as optim
from torchtext.legacy.data import Dataset, BucketIterator
from torchtext.legacy.datasets import TranslationDataset
from tqdm import tqdm

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

__author__ = "Yu-Hsiang Huang"


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


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    """ Epoch operation in training phase"""

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # prepare data
        src_seq = patch_src(batch.src).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


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


def train(model, training_data, validation_data, optimizer, device, opt):
    """ Start training """

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        # noinspection PyPackageRequirements
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(get_output_dir(opt), 'tensorboard'))
    else:
        tb_writer = None

    log_train_file = os.path.join(get_output_dir(opt), 'train.log')
    log_valid_file = os.path.join(get_output_dir(opt), 'valid.log')

    print(f'[Info] Training performance will be written to file: {log_train_file} and {log_valid_file}')

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr_):
        print(f'  - {f"({header})":12} ppl: {ppl: 8.5f}, accuracy: {100 * accu:3.3f} %, lr: {lr_:8.5f}, '
              f'elapse: {(time.time() - start_time) / 60:3.3f} min')

    valid_losses = []

    os.makedirs('initial', exist_ok=True)
    init_path = 'initial/model.pt'
    if not os.path.exists(init_path):
        print(f'[Info] Saving initial model to {init_path}')
        torch.save(model.state_dict(), init_path)
    elif opt.x is None or opt.y is None:
        print(f'[Info] Loading initial model at {init_path}')
        model.load_state_dict(torch.load(init_path))

    global epoch, model_dict
    start = epoch if epoch is not None else 0
    if model_dict is not None:
        model.load_state_dict(model_dict)
    for epoch_i in range(start, opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        # noinspection PyProtectedMember
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = f'model_accu_{100 * valid_accu:3.3f}.chkpt'
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(get_output_dir(opt), model_name))
                print('    - [Info] The checkpoint file has been updated.')
        elif opt.save_mode == 'last':
            model_name = 'model.chkpt'
            torch.save(checkpoint, os.path.join(get_output_dir(opt), model_name))
            print('    - [Info] The checkpoint file has been recorded.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write(f'{epoch_i},{train_loss: 8.5f},{train_ppl: 8.5f},{100 * train_accu:3.3f}\n')
            log_vf.write(f'{epoch_i},{valid_loss: 8.5f},{valid_ppl: 8.5f},{100 * valid_accu:3.3f}\n')

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu * 100, 'val': valid_accu * 100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)


def main():
    """
    Usage: python train.py -data_pkl m30k_deen_shr.pkl -embs_share_weight -proj_share_weight -label_smoothing
    -output_dir output -b 256 -warmup 128000
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)  # all-in-1 data pickle or bpe field

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
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'last'], default='last')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-use_ckpt', default=None)
    parser.add_argument('-x', type=int, default=None)
    parser.add_argument('-y', type=int, default=None)

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

    transformer = prune(transformer, opt.x, opt.y)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


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


def get_dir_vector():
    initia_w = torch.load('initial/model.pt')
    pass1_w = torch.load('1passnoprune/model.chkpt')['model']
    direction = OrderedDict()
    for k in initia_w:
        diff = initia_w[k] - pass1_w[k]
        diff_sum = (diff ** 2).sum()
        norm_diff = diff / (diff_sum + 1e-6)
        direction[k] = norm_diff
    print(direction)
    torch.save(direction, 'vectors/opt_dir.pt')

    for k in initia_w:
        diff = torch.randn_like(initia_w[k])
        diff_sum = (diff ** 2).sum()
        norm_diff = diff / (diff_sum + 1e-6)
        direction[k] = norm_diff
    print(direction)
    torch.save(direction, 'vectors/rand_dir.pt')
    exit(0)


def prune(model, x, y):
    from torch.nn.utils import prune
    import re
    pass1_w = torch.load('1passnoprune/model.chkpt')['model']
    model.load_state_dict(pass1_w)
    all_params = []
    for a in model.named_parameters():
        if 'weight' not in a[0]:
            continue
        name = re.sub(r'\.(\d+)', r'[\1]', a[0])
        name = name.rsplit('.weight', 1)[0]
        all_params.append((eval(f'model.{name}'), 'weight'))
    prune.global_unstructured(all_params, pruning_method=prune.L1Unstructured, amount=0.1)
    with torch.no_grad():
        state_dict = model.state_dict()
        opt_vec = torch.load('vectors/opt_dir.pt')
        rand_vec = torch.load('vectors/rand_dir.pt')
        for a in state_dict:
            if 'weight_orig' not in a:
                continue
            weight_str = a.replace('weight_orig', 'weight')
            state_dict[a] = x * opt_vec[weight_str] + y * rand_vec[weight_str]
        model.load_state_dict(state_dict)
    return model


def get_output_dir(opt):
    return f'{opt.x},{opt.y}'


if __name__ == '__main__':
    epoch = None
    model_dict = None

    # get_dir_vector()
    main()
