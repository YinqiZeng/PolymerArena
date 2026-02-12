import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse
import torch.nn.functional as F
import copy

import copy
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import (
    to_dense_adj,
    dense_to_sparse,
    degree
)


def draw_graph(adj, title=None, thres=0.5, figsize=(6.5, 6.5)):
    N = len(adj)
    row_g = np.floor(np.sqrt(N)).astype(np.int16)
    col_g = np.ceil(N / row_g).astype(np.int16)
    fig_g, axs_g = plt.subplots(nrows=row_g, ncols=col_g, figsize=figsize)

    if title is not None:
        fig_g.suptitle(f'{title}', fontsize=20)

    if row_g == 1:
        for i in range(col_g):
            axs_g[i].axis("off")
    else:
        for i in range(row_g):
            for j in range(col_g):
                axs_g[i][j].axis("off")

    axs_g = axs_g.flatten()

    for i in range(N):
        adj[i][np.where(adj[i] <= thres)] = 0
        G = nx.Graph()
        for j in range(len(adj[i])):
            for k in range(j + 1, len(adj[i])):
                G.add_edge(j, k, weight=min(1, adj[i][j, k] * 2))

        widths = nx.get_edge_attributes(G, 'weight')
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(
            G, pos, ax=axs_g[i],
            node_size=15, alpha=1.0
        )
        nx.draw_networkx_edges(
            G, pos, ax=axs_g[i],
            edgelist=widths.keys(),
            width=list(widths.values()),
            edge_color='black',
            alpha=1.0
        )
        axs_g[i].set_title(r'$\mathcal{G}_{%i}$' % (i + 1),
                           y=-0.15, fontsize=16)


def preprocess(data_in):
    """
    Regression-safe preprocessing.

    - DOES NOT convert y to one-hot
    - Preserves continuous targets
    - Still constructs node features if missing
    """
    dataset = copy.deepcopy(data_in)
    is_plain = dataset[0].x is None

    # -------- Node feature construction (unchanged logic) --------
    if is_plain:
        max_degree = 0
        degs = []
        for data in dataset:
            d = degree(data.edge_index[0], dtype=torch.long)
            degs.append(d)
            max_degree = max(max_degree, d.max().item())

        if max_degree < 2000:
            for data in dataset:
                d = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(d, num_classes=max_degree + 1).to(torch.float)
        else:
            all_deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = all_deg.mean().item(), all_deg.std().item()
            for data in dataset:
                d = degree(data.edge_index[0], dtype=torch.long)
                data.x = ((d - mean) / std).view(-1, 1)

    # -------- Edge weight + num_nodes --------
    for data in dataset:
        adj = to_dense_adj(data.edge_index)
        _, edge_weight = dense_to_sparse(adj)
        data.edge_weight = edge_weight
        data.num_nodes = len(adj.squeeze())
        if data.edge_attr is not None:
            data.edge_attr = None

    # -------- Regression output dimension --------
    y_dim = None
    if dataset[0].y is not None:
        y = dataset[0].y
        y_dim = y.numel() if y.dim() > 0 else 1

    return dataset, y_dim


def get_class_label(dataset):
    """
    Compatibility stub.

    GeoMix (original) assumes classification.
    For regression, we return a single dummy class
    to prevent crashes if accidentally called.
    """
    return [list(range(len(dataset)))]




'''
def draw_graph(adj, title = None, thres = 0.5, figsize = (6.5,6.5)):
    N = len(adj)
    row_g = np.floor(np.sqrt(N)).astype(np.int16)
    col_g = np.ceil(N/row_g).astype(np.int16)
    fig_g, axs_g = plt.subplots(nrows = row_g, ncols = col_g, figsize=figsize)
    if title is not None:
        fig_g.suptitle(f'{title}', fontsize = 20)
    if row_g == 1:
        for i in range(col_g):
            axs_g[i].axis("off")
    else:
        for i in range(row_g):
            for j in range(col_g):
                axs_g[i][j].axis("off")
    axs_g = axs_g.flatten()
    for i in range(N):
        adj[i][np.where(adj[i] <= thres)] = 0
        G = nx.Graph()
        for j in range(len(adj[i])):
            for k in range(j+1,len(adj[i])):
                G.add_edge(j, k, weight=min(1, adj[i][j,k] * 2))
        widths = nx.get_edge_attributes(G, 'weight')
        nodelist = G.nodes()
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G,pos,ax = axs_g[i],
                            nodelist=nodelist,
                            node_size=15,
                            alpha=1.0)
        nx.draw_networkx_edges(G,pos,ax = axs_g[i],
                            edgelist = widths.keys(),
                            width=list(widths.values()),
                            edge_color='black',
                            alpha=1.0)      
        axs_g[i].set_title(r'$\mathcal{G}_{%i}$'%(i+1),y=-0.15, fontsize = 16)


def preprocess(data_in):
    dataset = copy.deepcopy(data_in)
    is_plain = dataset[0].x is None
    
    ## convert graph label into one-hot
    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))
    num_classes = len(y_set)
    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0].view(-1, num_classes)
    if is_plain:    # use node degree as attributes
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max( max_degree, degs[-1].max().item() )

        if max_degree < 2000:
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ( (degs - mean) / std ).view( -1, 1 )
    
    for data in dataset:
        adj = to_dense_adj(data.edge_index)
        _, edge_weight = dense_to_sparse(adj)
        data.edge_weight = edge_weight
        data.num_nodes = len(adj.squeeze())
        if data.edge_attr is not None:
            data.edge_attr = None
                        
    return dataset, dataset[0].y.shape[1]


def get_class_label(dataset):
    class_ind = [[] for _ in range(len(dataset[0].y.squeeze()))]
    N = len(class_ind)
    for i in range(len(dataset)):
        class_ind[torch.argmax(dataset[i].y)].append(i)
    return class_ind

'''