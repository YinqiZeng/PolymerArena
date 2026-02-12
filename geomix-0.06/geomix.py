import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data

from geomix_utils import lgw, proj_graph
from utils import draw_graph
import matplotlib.pyplot as plt


def _iter_store_items(d: Data):
    if hasattr(d, "_store") and hasattr(d._store, "items"):
        return list(d._store.items())
    if hasattr(d, "__dict__") and "_store" in d.__dict__:
        return list(d.__dict__["_store"].items())
    return []


def _mix_or_copy_raw_fields(new_data: Data, d1: Data, d2: Data, lam: float):
    """
    Propagate all *_raw fields (y_reg_raw, y_arom_raw, etc.)
    """
    keys = set()
    for k, _ in _iter_store_items(d1):
        if isinstance(k, str) and k.endswith("_raw"):
            keys.add(k)
    for k, _ in _iter_store_items(d2):
        if isinstance(k, str) and k.endswith("_raw"):
            keys.add(k)

    for k in keys:
        v1 = getattr(d1, k, None)
        v2 = getattr(d2, k, None)

        if torch.is_tensor(v1) and torch.is_tensor(v2):
            setattr(new_data, k, (1 - lam) * v1.float() + lam * v2.float())
        elif v1 is not None:
            setattr(new_data, k, v1)
        elif v2 is not None:
            setattr(new_data, k, v2)


def _data_to_cpu(data: Data):
    """
    PyG DataLoader requires CPU-only tensors.
    """
    for k, v in data._store.items():
        if torch.is_tensor(v):
            data._store[k] = v.cpu()
    return data


def cand_ind(dataset, num_mixup):
    """
    Pair graphs with similar number of nodes (regression-safe).
    """
    num_graphs = len(dataset)
    assert num_graphs > 1

    node_counts = np.array([dataset[i].num_nodes for i in range(num_graphs)])
    index = []

    for _ in range(num_mixup):
        i = np.random.choice(num_graphs)
        diffs = np.abs(node_counts - node_counts[i]).astype(float)
        diffs[i] = np.inf
        j = np.argmin(diffs)
        index.append([i, j])

    return index


def geomix(dataset, args):
    """
    GeoMix for graph regression (edge_attr completely removed).
    """
    data_out = list(dataset)
    print("Mixup via Low-rank GW (edge_attr-free)")

    num_mixup = max(int(args.aug_ratio * len(dataset)), 1)
    index = cand_ind(dataset, num_mixup)
    mixup_size = []

    for g1, g2 in tqdm(index):
        # ---- adjacency (GPU OK internally) ----
        adj1 = to_dense_adj(
            dataset[g1].edge_index,
            max_num_nodes=dataset[g1].num_nodes
        ).squeeze().to(args.device)

        adj2 = to_dense_adj(
            dataset[g2].edge_index,
            max_num_nodes=dataset[g2].num_nodes
        ).squeeze().to(args.device)

        # ---- node features ----
        x1 = dataset[g1].x.squeeze().to(args.device).float()
        x2 = dataset[g2].x.squeeze().to(args.device).float()

        Q, R, g = lgw(adj1, adj2, x1, x2, args.num_nodes, alpha=args.alpha_fgw)

        coarsen_adj1, coarsen_adj2, coarsen_x1, coarsen_x2 = proj_graph(
            Q, R, g, adj1, adj2, x1, x2
        )

        mixup_size.append(coarsen_adj1.shape[0])

        y1, y2 = dataset[g1].y, dataset[g2].y

        # ---- lambda sampling ----
        if args.sample_dist == "uniform":
            lam_list = np.random.uniform(
                args.uniform_min, args.uniform_max, size=args.num_graphs
            )
        elif args.sample_dist == "beta":
            lam_list = np.random.beta(
                args.beta_alpha, args.beta_beta, size=args.num_graphs
            )
        else:
            raise ValueError("Invalid sampling distribution")

        for lam in lam_list:
            mixed_adj = (1 - lam) * coarsen_adj1 + lam * coarsen_adj2
            mixed_x = (1 - lam) * coarsen_x1 + lam * coarsen_x2
            mixed_adj.masked_fill_(mixed_adj.le(args.clip_eps), 0)

            edge_index, edge_weight = dense_to_sparse(mixed_adj)
            #edge_index, _ = dense_to_sparse(mixed_adj)


            mixed_y = None
            if y1 is not None and y2 is not None:
                mixed_y = (1 - lam) * y1.float() + lam * y2.float()

            new_data = Data(
                x=mixed_x,
                y=mixed_y,
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=mixup_size[-1],
            )

            # ---- propagate all *_raw labels ----
            _mix_or_copy_raw_fields(new_data, dataset[g1], dataset[g2], lam)

            if hasattr(dataset[g1], "y_mask_raw") and hasattr(dataset[g2], "y_mask_raw"):
                new_data.y_mask_raw = (
                    dataset[g1].y_mask_raw.float()
                    * dataset[g2].y_mask_raw.float()
                )

            data_out.append(_data_to_cpu(new_data))

    print(f"Average mixup graph size : {np.mean(mixup_size):.2f}")
    return data_out
