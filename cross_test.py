import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler
'''
test_dataset 
'''
def prepare_test_data(graphs, args, max_nodes=0):
    graphs = graphs[500:]
    position_graphs = graphs[:82]
    negative_graphs = graphs[82:]
    random.shuffle(negative_graphs)
    negative_graphs = negative_graphs[:82]
    graphs = position_graphs+negative_graphs



    # minibatch

    dataset_sampler = GraphSampler(graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)


    return dataset_loader