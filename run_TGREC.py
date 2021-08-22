import math
import logging
import time
import random
import sys
import argparse
import os
import json

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
#import numba

from model import TGRec
from data import Data
from evaluation import *
from utils import EarlyStopMonitor
from tqdm import tqdm
import multiprocessing
import metrics


### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGSRec experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--reg', type=float, default=0.1, help='regularization')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=20, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=20, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty', 'disentangle'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--samplerate', type=float, default=1.0, help='samplerate for each user')
parser.add_argument('--popnegsample', action='store_true', help='use popularity based negative sampling')
parser.add_argument('--timepopnegsample', action='store_true', help='use timely popularity based negative sampling')
parser.add_argument('--negsampleeval', type=int, default=-1, help='number of negative sampling evaluation, -1 for all')
parser.add_argument('--disencomponents', type=int, help='number of various time encoding')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATASET = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim


MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


RANK_RESULTS_DIR = f"./rank_results/{args.data}"
if not os.path.isdir(RANK_RESULTS_DIR):
    os.mkdir(RANK_RESULTS_DIR)
RANK_RESULTS_FILE = RANK_RESULTS_DIR + f"/{args.bs}_{args.n_degree}_{args.n_epoch}_{args.n_head}_{args.drop_out}_{args.time}_{args.n_layer}_{NUM_NEIGHBORS}_{args.node_dim}_{args.time_dim}_{args.lr}"


SAVE_MODEL_DIR = f"./saved_models/{args.data}"
if not os.path.isdir(SAVE_MODEL_DIR):
    os.mkdir(SAVE_MODEL_DIR)
SAVE_MODEL_PATH = SAVE_MODEL_DIR + f"/checkpoint.{args.bs}_{args.n_degree}_{args.n_epoch}_{args.n_head}_{args.drop_out}_{args.time}_{args.n_layer}_{NUM_NEIGHBORS}_{args.node_dim}_{args.time_dim}_{args.lr}.pth.tar"

def bpr_loss(pos_score, neg_score):
    loss = -((pos_score - neg_score).sigmoid().log().mean())
    return loss


data = Data(DATASET, args)


random.seed(2020)
torch.manual_seed(2020)
np.random.seed(2020)


### Model initialize
device = torch.device('cuda:{}'.format(GPU))
n_nodes = data.max_idx
print(n_nodes, "n nodes")
n_edges = data.num_total_edges
print(n_edges, "n edges")
tgan = TGRec(data.train_ngh_finder, n_nodes+1, args,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
#criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(data.train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

early_stopper = EarlyStopMonitor()
start_time = time.time()
for epoch in range(NUM_EPOCH):
    # Training 
    # training use only training graph
    tgan.ngh_finder = data.train_ngh_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))
    for k in range(num_batch):

        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut, dst_l_cut = data.train_src_l[s_idx:e_idx], data.train_dst_l[s_idx:e_idx]
        ts_l_cut = data.train_ts_l[s_idx:e_idx]
        label_l_cut = data.train_label_l[s_idx:e_idx]
        size = len(src_l_cut)
        #_, dst_l_fake = train_rand_sampler.sample(size)
        if args.popnegsample:
            dst_l_fake = data.train_rand_sampler.popularity_based_sample_neg(src_l_cut)
        elif args.timepopnegsample:
            dst_l_fake = data.train_rand_sampler.timelypopularity_based_sample_neg(src_l_cut, ts_l_cut)
        else:
            dst_l_fake = data.train_rand_sampler.sample_neg(src_l_cut)
        
        #with torch.no_grad():
        #    pos_label = torch.ones(size, dtype=torch.float, device=device)
        #    neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        tgan = tgan.train()
        pos_score, neg_score = tgan.contrast_nosigmoid(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        #pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
    
        #loss = criterion(pos_prob, pos_label)
        #loss += criterion(neg_prob, neg_label)
        loss = bpr_loss(pos_score, neg_score)
        l2_reg = 0
        for name, p in tgan.named_parameters():
            if "node_hist_embed" in name:
                l2_reg += p.norm(2)
        loss += args.reg*l2_reg
        
        #pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()
        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            tgan = tgan.eval()
            #pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_score = np.concatenate([(pos_score).cpu().detach().numpy(), (neg_score).cpu().detach().numpy()])
            #pred_label = pred_score > 0.5
            scaler = MinMaxScaler()
            preds = np.transpose(scaler.fit_transform(np.transpose([pred_score])))[0]
            pred_label = preds > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))
    print('epoch: {}:'.format(epoch))
    print('Epoch mean loss: {}'.format(np.mean(m_loss)))
    print('train acc: {}'.format(np.mean(acc)))
    print('train auc: {}'.format(np.mean(auc)))
    print('train f1: {}'.format(np.mean(f1)))
    print('train ap: {}'.format(np.mean(ap)))
    if np.mean(acc) == 0.5 and np.mean(auc) == 0.5 and np.mean(f1) == 0:
        break
    
    if ((epoch+1) % 20 == 0 and (epoch+1) >= 200) or (epoch+1) == args.n_epoch:
        torch.save(
                {
                    'model_state_dict': tgan.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.mean(m_loss),
                    }, SAVE_MODEL_PATH
        )
        print("model saved")
        tgan.ngh_finder = data.full_ngh_finder
        #val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l, val_dst_l, val_ts_l, val_label_l)
        valid_result, valid_pred_output = eval_users(tgan, data.val_src_l, data.val_dst_l, data.val_ts_l, data.train_src_l, data.train_dst_l, args)
        print('validation: ', valid_result)
        
        #test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l)
        test_result, test_pred_output = eval_users(tgan, data.test_src_l, data.test_dst_l, data.test_ts_l, data.train_src_l, data.train_dst_l, args)
        print('test: ', test_result)
        #test_pred_output_file = RANK_RESULTS_FILE + "_test.json"
        #valid_pred_output_file = RANK_RESULTS_FILE + '_valid.json'
        #if (epoch+1) == args.n_epoch:
        #    with open(valid_pred_output_file, 'w') as f:
        #        for eachoutput in valid_pred_output:
        #            f.write(json.dumps(eachoutput) + "\n")
        #    with open(test_pred_output_file, 'w') as f:
        #        for eachoutput in test_pred_output:
        #            f.write(json.dumps(eachoutput) + "\n")
        print("time freq: ", tgan.time_encoder.basis_freq.data)
        print("time phase: ", tgan.time_encoder.phase.data)

endtime = time.time()
print("need time: ", (endtime - start_time))
