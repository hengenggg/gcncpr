import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import pandas as pd
import util

def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)#data/ACP/ACP
    filename_graph_indic = prefix + '_graph_indicator.txt'#data/ACP/ACP_graph_indicator.txt
    # index of graphs that a given node belongs to
    graph_indic={}  #{1(node):1(graph),2:1,3:1 .....20:2....}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'#data/ACP/ACP_node_labels.txt
    node_labels=[]#[1,3,19,7...0,2]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1 #20
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'#data/ACP/ACP_node_attributes.txt
    node_attrs=[] #[[1,0,1,0,0..],...]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt' #data/ACP/ACP__graph_labels.txt
    graph_labels=[] #[1,1,1...2,...]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = [] #[1,2]
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            #if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    #graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}#{1(label):0(index),2:1}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels]) #[0,0,0,1.....]ndarray
    #if label_has_zero:
    #    graph_labels += 1

    filename_res = prefix + '_res.csv'
    res_data = []
    if(os.path.exists(filename_res)):
        df = pd.read_csv(filename_res,header=None)
        res_data = df.values

    filename_adj=prefix + '_A.txt'#data/ACP/ACP__A.txt

    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    #{1:[],2:[].....len(graph_labels):[]}
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}  #{1:[],2:[].....len(graph_labels):[]}
    num_edges = 0  #统计总共有多少条边
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" "))) #e0=2,e1=1
            adj_list[graph_indic[e0]].append((e0,e1))#graph_indic[e0]是图的index   adj_list:{1:[(2,1),(1,2)....],2:[..]}
            index_graph[graph_indic[e0]]+=[e0,e1] #[2,1,1,2,3,2....]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])] #[1,0,2....]

    graphs=[]
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])  #[(2,1),(1,2)....]
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1] # 0 or 1
        G.graph['res'] = res_data[i-1]
        for u in util.node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels #[0,0....0]
                node_label = node_labels[u-1] #0-19
                node_label_one_hot[node_label] = 1
                util.node_dict(G)[u]['label'] = node_label_one_hot #node's label
            if len(node_attrs) > 0:
                util.node_dict(G)[u]['feat'] = node_attrs[u-1] #node's attr
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]


        # relabeling
        mapping={}
        it=0
        for n in util.node_iter(G):#{1:0,2:1 ....}
            mapping[n]=it
            it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping)) #graph里的结点的index从0开始
    return graphs

