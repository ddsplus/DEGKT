#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGEKT 模型训练脚本
支持多个数据集：assist2009, assist2012, assist2017, statics2011, xes3g5m
含10轮早停逻辑和每个数据集单独文件夹保存最佳模型
"""
import sys
import os

# Bootstrap import paths - handle running from any directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KT_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(KT_ROOT)
for p in (REPO_ROOT, KT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Now safe to import KnowledgeTracing modules
from KnowledgeTracing.DirectedGCN.load_data import get_adj
from KnowledgeTracing.hgnn_models import hypergraph_utils as hgut
from KnowledgeTracing.model.Model import DKT
from KnowledgeTracing.data.dataloader import getLoader
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.evaluation import eval
from torch import optim as optima
import torch
import logging
from datetime import datetime
import numpy as np
import warnings
import random
import pandas as pd

warnings.filterwarnings('ignore')

'''check cuda'''
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
if use_gpu:
    torch.cuda.set_device(0)
print('GPU state: ', use_gpu)
print('Dataset: ' + C.DATASET + ', Ques number: ' + str(C.NUM_OF_QUESTIONS) + '\n')

''' save log '''
logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)
date = datetime.now()
handler = logging.FileHandler(
    f'log/{date.year}_{date.month}_{date.day}_result.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('This is a new training log')
logger.info('\nDataset: ' + str(C.DATASET) + ', Ques number: ' + str(C.NUM_OF_QUESTIONS) + ', Batch_size: ' + str(
    C.BATCH_SIZE))

'''set random seed'''

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(216)

def validate_dataset_dimensions():
    report = C.dataset_dimension_report(C.DATASET)

    print('Dataset dimension report:')
    print(
        '  pid_q_range: min={0}, max={1}, unique={2}'.format(
            report['pid_q_min'], report['pid_q_max'], report['pid_q_unique']
        )
    )
    print(
        '  configured_q: {0}'.format(C.NUM_OF_QUESTIONS)
    )
    print(
        '  h_shape: rows={0}, cols={1}, inferred_q={2}'.format(
            report['h_rows'], report['h_cols'], report['h_q_count']
        )
    )
    logger.info(
        'Dataset dimension report - pid_q_min=%s pid_q_max=%s pid_q_unique=%s configured_q=%s h_rows=%s h_cols=%s h_q_count=%s',
        report['pid_q_min'],
        report['pid_q_max'],
        report['pid_q_unique'],
        C.NUM_OF_QUESTIONS,
        report['h_rows'],
        report['h_cols'],
        report['h_q_count'],
    )

    if report['h_error'] is not None:
        raise ValueError(report['h_error'])

    pid_q_max = report['pid_q_max']
    if pid_q_max is not None and int(pid_q_max) > int(C.NUM_OF_QUESTIONS):
        raise ValueError(
            f'PID max question id exceeds configured question count for dataset={C.DATASET}: '
            f'pid_q_max={pid_q_max}, configured_q={C.NUM_OF_QUESTIONS}. '
            f'Please remap pid question ids or regenerate Dataset/H/{C.H}.csv from the same mapping.'
        )

    h_q_count = report['h_q_count']
    if h_q_count is not None and int(h_q_count) != int(C.NUM_OF_QUESTIONS):
        raise ValueError(
            f'H-derived question count mismatches configured question count for dataset={C.DATASET}: '
            f'h_q_count={h_q_count}, configured_q={C.NUM_OF_QUESTIONS}. '
            f'Please regenerate H or correct NUM_OF_QUESTIONS.'
        )


validate_dataset_dimensions()
trainLoaders, testLoaders = getLoader(C.DATASET)
loss_func = eval.lossFunc(C.HIDDEN, C.MAX_STEP, device)

def KTtrain():
    # Create dataset-specific model directory
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model', C.DATASET)
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f'Model directory: {model_dir}')
    
    adj = hgut.generate_G_from_H(pd.read_csv(r'../Dataset/H/' + C.H + '.csv', header=None))
    G = adj.cuda()
    expected_g_dim = int(2 * C.NUM_OF_QUESTIONS)
    if int(G.shape[0]) != expected_g_dim:
        raise ValueError(
            f'Graph/input dimension mismatch for dataset={C.DATASET}: '
            f'G.shape[0]={int(G.shape[0])}, but 2*NUM_OF_QUESTIONS={expected_g_dim}. '
            f'Please regenerate H or correct NUM_OF_QUESTIONS.'
        )
    adj_out, adj_in = get_adj()
    adj_in = adj_in.cuda()
    adj_out = adj_out.cuda()
    model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).cuda()
    optimizer = optima.Adam(model.parameters(), lr=C.LR)

    best_auc = 0.0
    best_epoch = 0
    best_acc = 0.0
    patience = 10  # Early stopping patience - 10 epochs without improvement
    patience_counter = 0  # Counter for epochs without improvement
    
    for epoch in range(C.EPOCH):
        print('epoch: ' + str(epoch + 1) + '            lr = ', optimizer.param_groups[0]["lr"])
        model, optimizer = eval.train_epoch(model, trainLoaders, optimizer,
                                            loss_func)
        logger.info(f'epoch {epoch + 1}')
        with torch.no_grad():
            auc, acc = eval.test_epoch(model, testLoaders, loss_func, device)
            
            if best_auc < auc:
                best_auc = auc
                best_acc = acc
                best_epoch = epoch + 1
                patience_counter = 0  # Reset patience counter
                
                # Save best model with descriptive filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_filename = f'best_model_{C.DATASET}_{best_epoch}_{timestamp}_auc{best_auc:.4f}_acc{best_acc:.4f}.pkl'
                model_path = os.path.join(model_dir, model_filename)
                torch.save(model, model_path)
                print(f'✓ Best model saved: {model_filename}')
                logger.info(f'Best model saved: {model_filename}')
            else:
                patience_counter += 1
                logger.info(f'No improvement. Patience: {patience_counter}/{patience}')

            print('Best auc at present: %f  acc:  %f  Best epoch: %d (patience: %d/%d)' % (best_auc, best_acc, best_epoch, patience_counter, patience))
            
            # Early stopping check
            if patience_counter >= patience:
                print(f'\n=== Early stopping triggered ===')
                print(f'No improvement for {patience} consecutive epochs')
                logger.info(f'Early stopping at epoch {epoch + 1} after {patience} epochs without improvement')
                break
    
    print(f'\n=== Training completed ===')
    print(f'Best AUC: {best_auc:.4f}, Best ACC: {best_acc:.4f} at epoch {best_epoch}')
    logger.info(f'Training completed. Best AUC: {best_auc:.4f}, Best ACC: {best_acc:.4f} at epoch {best_epoch}')

def KTtest():
    model = torch.load('../model/save2017model.pkl')
    print('loading the best model...')
    with torch.no_grad():
        eval.test_epoch(model, testLoaders, loss_func, device)




KTtrain()
# KTtest()
