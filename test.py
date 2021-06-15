import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

def feature_evaluation(cl_data_file, ablation, tc, dist, novel_idx, base_idx, num_classes, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()
    if ablation == 'sibling':
        cands = np.where((tc*(dist == 1))[:num_classes].sum(0) >= n_way)[0]
        parent = random.sample(list(cands), 1)
        cands = np.where(dist[:num_classes,parent] == 1)[0]
        select_class = base_idx(np.array(random.sample(list(cands), n_way)))
    elif ablation == 'cousin':
        select_class = np.zeros(n_way, dtype=np.int64)
        allowed = np.ones(num_classes, dtype=np.int64)
        for i in range(n_way):
            cands = np.where(allowed)[0]
            select_class[i] = random.sample(list(cands), 1)[0]
            print(select_class[i])
            allowed[dist[:num_classes,select_class[i]] < 3] = 0
        select_class = base_idx(select_class)
    else:
        select_class = np.array(random.sample(class_list, n_way))
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, select_class, len(class_list), is_feature = True)
    else:
        scores  = model.set_forward(z_all, select_class, len(class_list), is_feature = True)
    
    acc = []
    dists = []
    for each_score in scores:
        pred = each_score.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( n_way ), n_query )
        acc.append(np.mean(pred == y)*100 )
        dists.append(np.mean(dist[novel_idx(select_class[pred]), novel_idx(select_class[y])]))
    return acc, dists

if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []

    n_query = 15
    iter_num = 10000

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
    feature_dir = '%s/features/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.dataset == 'cross':
        params.dataset = 'CUB'

    tc = np.load('filelists/%s/tc.npy' %(params.dataset))
    distm = np.load('filelists/%s/dist.npy' %(params.dataset))
    if params.dataset in ['miniImagenet','cifar']:
        novel_idx = lambda x: x - 80
        base_idx = lambda x: x + 80
        num_classes = 20
    elif params.dataset == 'CUB':
        novel_idx = lambda x: (x - 3) // 4
        base_idx = lambda x: 4 * x + 3
        num_classes = 50
    model = BaselineFinetune( model_dict[params.model], tc, novel_idx, **few_shot_params )

    if torch.cuda.is_available():
        model = model.cuda()

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split

    novel_file = os.path.join( feature_dir, split_str +".hdf5") 
    cl_data_file = feat_loader.init_loader(novel_file)
        
    acc_all1, acc_all2 , acc_all3 = [],[],[]
    dist_all1, dist_all2 , dist_all3 = [],[],[]

    print(novel_file)
    print("evaluating over %d examples"%(n_query))

    for i in range(iter_num):
        acc, dist = feature_evaluation(cl_data_file, params.ablation, tc, distm, novel_idx, base_idx, num_classes, model, n_query = n_query , adaptation = params.adaptation, **few_shot_params)
            
        acc_all1.append(acc[0])
        acc_all2.append(acc[1])
        acc_all3.append(acc[2])

        dist_all1.append(dist[0])
        dist_all2.append(dist[1])
        dist_all3.append(dist[2])
        print("%d steps reached and the mean acc is %g , %g , %g"%(i, np.mean(np.array(acc_all1)),np.mean(np.array(acc_all2)),np.mean(np.array(acc_all3)) ))
        print("%d steps reached and the mean graph dist is %g , %g , %g"%(i, np.mean(np.array(dist_all1)),np.mean(np.array(dist_all2)),np.mean(np.array(dist_all3)) ))

    acc_mean1 = np.mean(acc_all1)
    acc_mean2 = np.mean(acc_all2)
    acc_mean3 = np.mean(acc_all3)
    acc_std1  = np.std(acc_all1)
    acc_std2  = np.std(acc_all2)
    acc_std3  = np.std(acc_all3)
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' %(iter_num, acc_mean1, 1.96* acc_std1/np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' %(iter_num, acc_mean2, 1.96* acc_std2/np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' %(iter_num, acc_mean3, 1.96* acc_std3/np.sqrt(iter_num)))
    
    dist_mean1 = np.mean(dist_all1)
    dist_mean2 = np.mean(dist_all2)
    dist_mean3 = np.mean(dist_all3)
    dist_std1  = np.std(dist_all1)
    dist_std2  = np.std(dist_all2)
    dist_std3  = np.std(dist_all3)
    print('%d Test Graph Dist at 100= %4.2f +- %4.2f' %(iter_num, dist_mean1, 1.96* dist_std1/np.sqrt(iter_num)))
    print('%d Test Graph Dist at 200= %4.2f +- %4.2f' %(iter_num, dist_mean2, 1.96* dist_std2/np.sqrt(iter_num)))
    print('%d Test Graph Dist at 300= %4.2f +- %4.2f' %(iter_num, dist_mean3, 1.96* dist_std3/np.sqrt(iter_num)))
