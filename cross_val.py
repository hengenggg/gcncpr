import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler
'''
10-fold验证，1/10为验证集，其余为训练集
'''
def prepare_val_data(graphs, args, val_idx, max_nodes=0,train_num=0):
    test_graphs=[]
    if not train_num == 0:
        test_graphs = graphs[train_num:train_num+164]
        graphs = graphs[:train_num]
        random.shuffle(test_graphs)
    random.shuffle(graphs)
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    # train_graphs_pos = list(filter(lambda x:x.graph['label']==0,train_graphs))
    # train_graphs_neg = list(filter(lambda x:x.graph['label']==1,train_graphs))
    # train_graphs_neg = random.sample(train_graphs_neg,len(train_graphs_pos)//10)
    # train_graphs = train_graphs_pos+train_graphs_neg
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)


    if test_graphs == None or len(test_graphs) == 0:
        test_dataset_loader = None
    else:
        dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
                                       features=args.feature_type)
        test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader,  test_dataset_loader,\
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim,dataset_sampler.res_dim

