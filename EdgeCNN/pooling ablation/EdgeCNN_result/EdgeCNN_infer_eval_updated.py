#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auto-generated from Jupyter notebook:
#   EdgeCNN+infer+eval.ipynb
# Generated on: 2026-01-23T16:58:42.499311Z

# --- Code cell 1 ---
# NOTE: notebook magic removed: %%capture
# NOTE: notebook magic removed: !pip install rdkit torch_geometric sklearn.preprocessing


# --- Code cell 2 ---
import numpy as np
import pandas as pd
import copy
import math
import csv
import argparse
import pickle
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from rdkit import Chem
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeCNN, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import MLP, TopKPooling, SAGPooling
from torch_geometric.utils.smiles import from_smiles

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# --- Code cell 3 ---
# Parse command line arguments
parser = argparse.ArgumentParser(description='EdgeCNN training with different pooling functions')
parser.add_argument('--pooling', type=str, default='mean',
                    choices=['mean', 'add', 'max', 'topk', 'sag'],
                    help='Pooling function to use: mean, add, max, topk, or sag (default: mean)')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='Pooling ratio for topk and sag pooling (default: 0.5)')
parser.add_argument('--gpu', type=int, default=None,
                    help='GPU ID to use (e.g., 0, 1, 2, etc.). If not specified, will use CPU or first available GPU.')
parser.add_argument('--cache_dataset', action='store_true',
                    help='Cache the processed dataset to disk for faster loading next time')
parser.add_argument('--dataset_cache_path', type=str, default='dataset_cache.pkl',
                    help='Path to save/load the cached dataset (default: dataset_cache.pkl)')
parser.add_argument('--test_only', action='store_true',
                    help='Skip training and only run inference/evaluation using the saved best model')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to the saved model checkpoint (default: multitask_gcn_best_{pooling_type}.pth)')
args = parser.parse_args()

# Set pooling function based on argument
POOLING_TYPE = args.pooling
POOLING_RATIO = args.pooling_ratio
GPU_ID = args.gpu

# For simple global pooling functions
if POOLING_TYPE == 'mean':
    pooling_fn = global_mean_pool
    IS_LEARNABLE_POOLING = False
elif POOLING_TYPE == 'add':
    pooling_fn = global_add_pool
    IS_LEARNABLE_POOLING = False
elif POOLING_TYPE == 'max':
    pooling_fn = global_max_pool
    IS_LEARNABLE_POOLING = False
elif POOLING_TYPE in ['topk', 'sag']:
    pooling_fn = None  # Will be created inside the model
    IS_LEARNABLE_POOLING = True
else:
    raise ValueError(f"Unknown pooling type: {POOLING_TYPE}")

print(f"Using pooling function: {POOLING_TYPE}")
if IS_LEARNABLE_POOLING:
    print(f"Pooling ratio: {POOLING_RATIO}")

# Set up GPU device
if GPU_ID is not None:
    # User specified a GPU ID
    if torch.cuda.is_available():
        if GPU_ID < torch.cuda.device_count():
            device = torch.device(f'cuda:{GPU_ID}')
            print(f"Using specified GPU: {GPU_ID} ({torch.cuda.get_device_name(GPU_ID)})")
        else:
            print(f"Warning: GPU {GPU_ID} not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    else:
        print("Warning: CUDA not available. Falling back to CPU")
        device = torch.device('cpu')
else:
    # Auto-select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: 0 ({torch.cuda.get_device_name(0)})")
    else:
        print("Using CPU")

df = pd.read_csv("../final_result2.0_zero_filled.csv")


# --- Code cell 4 ---
torch.manual_seed(12345)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12345)


# --- Code cell 5 ---
# ----------------
# Task groups
# ----------------

TARGET_COLS_PHYS_CHEM = ["Solubility", "Molecular_Weight", "Log_P", "BoilingPoint"]     # 4
TARGET_COLS_OPT_QUANT = ["sigma_780nm", "max_sigma", "ISC(S1-T1)"]                      # 3
TARGET_COLS_STRUC_TOX = ["SAscore", "Tox_score"]                                        # 2
TARGET_COL_OTHER      = ["is_aromatic"]                                                 # 1

REG_COLS = TARGET_COLS_PHYS_CHEM + TARGET_COLS_OPT_QUANT + TARGET_COLS_STRUC_TOX        # 9 regression cols
BIN_COL  = TARGET_COL_OTHER[0]                                                          # 1 binary col


# --- Code cell 6 ---
# -----------------------------------------------------------------
# Preprocess SMILES strings to generate graphs using RDKit and PyG
# -----------------------------------------------------------------

def smiles_to_graph(smiles: str):
    try:
        # Returns Data(x, edge_index, edge_attr)
        # edge_attr is bond features (you can ignore for now unless you upgrade the model)
        data = from_smiles(smiles, with_hydrogen=True, kekulize=False)
        return data
    except Exception:
        print("wrong SMILES")
        return None


# --- Code cell 7 ---
# -----------------------
# PyG Dataset wrapper
# -----------------------

class GraphListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# --- Code cell 8 ---
# -------------------------
# create graphs dataset
# -------------------------

CACHE_DATASET = args.cache_dataset
DATASET_CACHE_PATH = args.dataset_cache_path

# Check if cached dataset exists
if CACHE_DATASET and os.path.exists(DATASET_CACHE_PATH):
    print(f"Loading cached dataset from {DATASET_CACHE_PATH}...")
    with open(DATASET_CACHE_PATH, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded cached dataset with {len(dataset)} graphs")
else:
    print("Processing dataset from CSV...")
    data_list = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row["SMILES"])
        if g is None:
            continue

        y_reg_raw  = row[REG_COLS].to_numpy(dtype=np.float32)              # (9,)
        y_arom_raw = np.array([row[BIN_COL]], dtype=np.float32)            # (1,) should be 0/1

        g.y_reg_raw  = torch.tensor(y_reg_raw, dtype=torch.float32)        # (9,)
        g.y_arom_raw = torch.tensor(y_arom_raw, dtype=torch.float32)       # (1,)

        data_list.append(g)

    dataset = GraphListDataset(data_list)
    print(f"Total graphs: {len(dataset)}")

    # Save dataset to cache if enabled
    if CACHE_DATASET:
        print(f"Saving dataset to cache: {DATASET_CACHE_PATH}...")
        with open(DATASET_CACHE_PATH, 'wb') as f:
            pickle.dump(dataset, f)
        print("Dataset cached successfully")


# --- Code cell 9 ---
# inspect one graph's tensor shapes
g0 = dataset[21651]
print("graph[0].x shape:", tuple(g0.x.shape))                      # (num_nodes, num_node_features)
print("graph[0].edge_index shape:", tuple(g0.edge_index.shape))    # (2, num_edges*2) if bidirectional
print("graph[0].edge_attr shape:", tuple(g0.edge_attr.shape))
print("graph[0].y_raw shape:", tuple(g0.y_reg_raw.shape))          # (len(TARGET_COLS),)
print("graph[0].y_raw shape:", tuple(g0.y_arom_raw.shape))


# --- Code cell 10 ---
# -----------------------------------------------------------
# Split dataset into training and test datasets (95/5 split)
# -----------------------------------------------------------

torch.manual_seed(12345)
dataset = dataset.shuffle()
train_cut = int(0.95 * len(dataset))
test_size = len(dataset) - train_cut
train_dataset = dataset[:train_cut]
test_dataset = dataset[train_cut:]


# --- Code cell 11 ---
# sanity check
print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of validation graphs: {len(test_dataset)}")


# --- Code cell 12 ---
# --------------------------------
# Masking, Normalizing & Batching
# --------------------------------

# Create per-graph ZERO mask on the RAW regression targets (REG_COLS)
# Attach as d.y_mask
for d in train_dataset:
    d.y_mask_raw = (d.y_reg_raw != 0).float()   # (9,) 1==valid, 0==zero
for d in test_dataset:
    d.y_mask_raw = (d.y_reg_raw != 0).float()   # (9,) 1==valid, 0==zero

# Fit scaler on TRAIN regression targets
scaler = StandardScaler()
train_reg = torch.stack([d.y_reg_raw for d in train_dataset], dim=0).cpu().numpy()  # (N_train, 9)
scaler.fit(train_reg)

# Attach normalized regression targets + raw binary targets + mask
for d in train_dataset:
    reg_norm = scaler.transform(d.y_reg_raw.view(1, -1).cpu().numpy()).astype(np.float32)[0]  # (9,)
    d.y_reg  = torch.tensor(reg_norm, dtype=torch.float32).view(1, -1)                        # (1, 9)
    d.y_mask = d.y_mask_raw.view(1, -1)                                                       # (1, 9)
    d.y_arom = d.y_arom_raw.view(1, -1).float()                                               # (1, 1) 0/1

for d in test_dataset:
    reg_norm = scaler.transform(d.y_reg_raw.view(1, -1).cpu().numpy()).astype(np.float32)[0]
    d.y_reg  = torch.tensor(reg_norm, dtype=torch.float32).view(1, -1)
    d.y_mask = d.y_mask_raw.view(1, -1)
    d.y_arom = d.y_arom_raw.view(1, -1).float()

# Build loaders AFTER targets are attached
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=2048, shuffle=False)


# --- Code cell 13 ---
# sanity check 1
print("train_reg shape:", train_reg.shape)              # should be (N_train, 9)
print("scaler.mean_.shape:", scaler.mean_.shape)        # (9,)
print("scaler.scale_.shape:", scaler.scale_.shape)      # (9,)
print("first 9 means:", scaler.mean_[:10])
print("first 9 stds:", scaler.scale_[:10])


# --- Code cell 14 ---
# sanity check 2 
train_y_norm = scaler.transform(train_reg)
print("col means (train, after):", train_y_norm.mean(axis=0))       # when transform again, mean close to 0, and stds close to 1
print("col stds  (train, after):", train_y_norm.std(axis=0))        # which is normal distribution


# --- Code cell 15 ---
# -------------
# Define Model
# -------------

class MultiTaskEdgeCNN(nn.Module):
    def __init__(self, hidden_channels=128, pooling_fn=global_mean_pool,
                 pooling_type='mean', pooling_ratio=0.5):
        super().__init__()
        torch.manual_seed(12345)

        self.pooling_fn = pooling_fn
        self.pooling_type = pooling_type
        self.pooling_ratio = pooling_ratio

        # GCN layers
        # GINConv layers, each with an MLP
        self.conv1 = EdgeCNN(9, hidden_channels, 1)
        self.conv2 = EdgeCNN(hidden_channels, hidden_channels, 1)
        self.conv3 = EdgeCNN(hidden_channels, hidden_channels, 1)

        # Hierarchical pooling layers (for topk and sag) - one after each conv layer
        if pooling_type == 'topk':
            self.pool1 = TopKPooling(hidden_channels, ratio=pooling_ratio)
            self.pool2 = TopKPooling(hidden_channels, ratio=pooling_ratio)
            self.pool3 = TopKPooling(hidden_channels, ratio=pooling_ratio)
        elif pooling_type == 'sag':
            self.pool1 = SAGPooling(hidden_channels, ratio=pooling_ratio)
            self.pool2 = SAGPooling(hidden_channels, ratio=pooling_ratio)
            self.pool3 = SAGPooling(hidden_channels, ratio=pooling_ratio)
        else:
            self.pool1 = None
            self.pool2 = None
            self.pool3 = None

        # Fully connected layers for different tasks (heads)

        # For phys_chem head: 128 -> 64 -> 4
        self.fc_phys_chem_1 = nn.Linear(128, 128)
        self.fc_phys_chem_2 = nn.Linear(128, 64)
        self.fc_phys_chem_3 = nn.Linear(64, 4)

        # For opt_quant head: 128 -> 64 -> 3
        self.fc_opt_quant_1 = nn.Linear(128, 128)
        self.fc_opt_quant_2 = nn.Linear(128, 64)
        self.fc_opt_quant_3 = nn.Linear(64, 3)

        # For struc_tox head: 128 -> 64 -> 2
        self.fc_struc_tox_1 = nn.Linear(128, 64)
        self.fc_struc_tox_2 = nn.Linear(64, 2)

        # For other head: 128 -> 64 -> 32 -> 1 (logits)
        self.fc_other_1 = nn.Linear(128, 64)
        self.fc_other_2 = nn.Linear(64, 32)
        self.fc_other_3 = nn.Linear(32, 1)  # logits for BCEWithLogitsLoss

    def forward(self, x, edge_index, batch):
        # Apply pooling based on type
        if self.pooling_type in ['topk', 'sag']:
            # Hierarchical pooling architecture: Conv -> Pool -> Conv -> Pool -> Conv -> Pool

            # Layer 1: Conv + Pool
            x = self.conv1(x, edge_index).relu()
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

            # Layer 2: Conv + Pool
            x = self.conv2(x, edge_index).relu()
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

            # Layer 3: Conv + Pool
            x = self.conv3(x, edge_index)
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

            # Final global pooling to get graph-level representation
            x = global_mean_pool(x, batch)
        else:
            # Simple global pooling: Conv -> Conv -> Conv -> Global Pool
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            x = self.conv3(x, edge_index)

            # Simple global pooling (mean, add, or max)
            x = self.pooling_fn(x, batch)

        # Apply dropout for regularization
        x = F.dropout(x, p=0.2, training=self.training)

        # Phys_chem # (B,4)
        phys = self.fc_phys_chem_1(x).relu()
        phys = self.fc_phys_chem_2(phys).relu()
        phys = self.fc_phys_chem_3(phys)

        # Opt_quant # (B,3)
        opt = self.fc_opt_quant_1(x).relu()
        opt = self.fc_opt_quant_2(opt).relu()
        opt = self.fc_opt_quant_3(opt)


        # Struc_tox # (B,2)
        tox = self.fc_struc_tox_1(x).relu()
        tox = self.fc_struc_tox_2(tox)

        # Other # (B,1) logits
        arom = self.fc_other_1(x).relu()
        arom = self.fc_other_2(arom).relu()
        arom = self.fc_other_3(arom)  # logits for BCEWithLogitsLoss

        return phys, opt, tox, arom


# --- Code cell 16 ---
# ---------------------------
# Training Tools & Parameter 
# ---------------------------

model = MultiTaskEdgeCNN(
    hidden_channels=128,
    pooling_fn=pooling_fn,
    pooling_type=POOLING_TYPE,
    pooling_ratio=POOLING_RATIO
).to(device)

# Use AdamW optimizer instead of Adam
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

# Learning rate scheduler: Reduce learning rate on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5, threshold=1e-4, threshold_mode='rel'
)

# ---------------------------------------------------------
# UPDATED masked MSE with new semantics:
# mask==1 means "use this entry"
# mask==0 means "ignore this entry"
# ---------------------------------------------------------

def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, k)
    valid_mask:  (B, k) with 1==valid/use, 0==ignore
    """
    valid = valid_mask.to(dtype=pred.dtype)
    se = (pred - target).pow(2) * valid
    denom = valid.sum().clamp_min(1.0)
    return se.sum() / denom

criterion_bce = nn.BCEWithLogitsLoss()  

print(model)


# --- Code cell 17 ---
# ------------------------
# Define Training process
# ------------------------

def train_epoch():
    model.train()
    total = 0.0
    n = 0

    for data in train_loader:
        data = data.to(device)
        data.x = data.x.float()
        phys_out, opt_out, tox_out, arom_logit = model(data.x, data.edge_index, data.batch)

        y_reg  = data.y_reg          # (B, 7)
        y_mask = data.y_mask         # (B, 7)
        y_arom = data.y_arom         # (B, 1)

        y_phys = y_reg[:, 0:4]
        y_opt  = y_reg[:, 4:7]
        y_tox  = y_reg[:, 7:9]

        m_phys = y_mask[:, 0:4]
        m_opt  = y_mask[:, 4:7]
        m_tox  = y_mask[:, 7:9]

        loss_phys = masked_mse(phys_out, y_phys, m_phys) 
        loss_opt  = masked_mse(opt_out,  y_opt,  m_opt)
        loss_tox  = masked_mse(tox_out,  y_tox,  m_tox)
        loss_arom = criterion_bce(arom_logit, y_arom)

        optimizer.zero_grad(set_to_none=True)
        loss_phys.backward(retain_graph=True)
        loss_opt.backward(retain_graph=True)
        loss_tox.backward(retain_graph=True)
        loss_arom.backward()
        
        optimizer.step()

        total += (loss_phys + loss_opt + loss_tox + loss_arom).item()
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def eval_loss(loader):
    model.eval()
    total = 0.0
    n = 0

    for data in loader:
        data = data.to(device)
        phys_out, opt_out, tox_out, arom_logit = model(data.x.float(), data.edge_index, data.batch)

        y_reg  = data.y_reg
        y_mask = data.y_mask
        y_arom = data.y_arom

        y_phys = y_reg[:, 0:4]
        y_opt  = y_reg[:, 4:7]
        y_tox  = y_reg[:, 7:9]

        m_phys = y_mask[:, 0:4]
        m_opt  = y_mask[:, 4:7]
        m_tox  = y_mask[:, 7:9]

        loss_phys = masked_mse(phys_out, y_phys, m_phys)
        loss_opt  = masked_mse(opt_out,  y_opt,  m_opt) 
        loss_tox  = masked_mse(tox_out,  y_tox,  m_tox) 
        loss_arom = criterion_bce(arom_logit, y_arom) 

        total += (loss_phys + loss_opt + loss_tox + loss_arom).item()
        n += 1

    return total / max(n, 1)


# --- Code cell 18 ---
# ----------------------------
# Train + log to CSV
# ----------------------------

if not args.test_only:
    # Training mode
    num_epochs = 500
    log_path = f"loss_log_{POOLING_TYPE}.csv"

    best_test = math.inf
    best_epoch = -1
    best_train_loss = None
    best_test_loss = None
    best_model_state = None
    best_optimizer_state = None
    best_scheduler_state = None

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss"])  # header

        for epoch in range(1, num_epochs + 1):

            train_loss = train_epoch()
            test_loss  = eval_loss(test_loader)
            scheduler.step(test_loss)

            writer.writerow([epoch, float(train_loss), float(test_loss)])
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

            # find best model
            if test_loss < best_test:
                best_test = float(test_loss)
                best_epoch = epoch

                best_train_loss = float(train_loss)
                best_test_loss  = float(test_loss)

                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_scheduler_state = copy.deepcopy(scheduler.state_dict())

    print(f"Saved: {log_path}")


    # --- Code cell 19 ---
    # ----------------------------
    # Save
    # ----------------------------

    # LAST model
    torch.save({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "reg_cols": REG_COLS,
        "bin_col": BIN_COL,
        "pooling_type": POOLING_TYPE,
        "pooling_ratio": POOLING_RATIO,
        },
        f"multitask_gcn_last_{POOLING_TYPE}.pth"
    )


    # BEST model
    torch.save({
        "epoch": best_epoch,
        "train_loss": best_train_loss,
        "test_loss": best_test_loss,

        "model_state_dict": best_model_state,
        "optimizer_state_dict": best_optimizer_state,
        "scheduler_state_dict": best_scheduler_state,

        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "reg_cols": REG_COLS,
        "bin_col": BIN_COL,
        "pooling_type": POOLING_TYPE,
        "pooling_ratio": POOLING_RATIO,
        },
        f"multitask_gcn_best_{POOLING_TYPE}.pth"
    )
else:
    # Test-only mode: Skip training
    print("Test-only mode: Skipping training...")


# --- Code cell 20 ---
# ----------------------------
# Load Model
# ----------------------------

# Determine the model path
if args.model_path is not None:
    model_path = args.model_path
else:
    model_path = f'multitask_gcn_best_{POOLING_TYPE}.pth'

print(f"Loading model from: {model_path}")

# Load the best model checkpoint
checkpoint = torch.load(model_path, weights_only=False)
scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)         # (9,)
scaler_scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)       # (9,)
saved_pooling_ratio = checkpoint.get("pooling_ratio", 0.5)  # Default to 0.5 if not saved

# Recreate the model architecture with the same pooling function
model = MultiTaskEdgeCNN(
    hidden_channels=128,
    pooling_fn=pooling_fn,
    pooling_type=POOLING_TYPE,
    pooling_ratio=saved_pooling_ratio
).to(device)

# Load the model state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()                                                                # Set model to evaluation mode

print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
print(f"Best train loss: {checkpoint['train_loss']:.6f}, Best test loss: {checkpoint['test_loss']:.6f}")


# --- Code cell 21 ---
test_df = pd.read_csv("../3000testbank.csv")
data_list = []

for _, row in test_df.iterrows():
    graph = smiles_to_graph(row["SMILES"])
    if graph is not None:
        data_list.append(graph)

test_dataset = GraphListDataset(data_list)


# --- Code cell 22 ---
# ----------------------------
# Make Prediction
# ----------------------------

# Function to compute RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Function to compute Pearson correlation
def pearson_r(x, y):
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0, 1]

# Initialize list for predictions
predictions = []

# Inverse transform function to denormalize the predictions
def reg_inverse_transform(z_norm_1x9: np.ndarray) -> np.ndarray:
    # z_norm = (x - mean)/scale  => x = z_norm*scale + mean
    return z_norm_1x9 * scaler_scale + scaler_mean

# Iterate through the test dataset and make predictions
for data in test_dataset:
    data = data.to(device)
    
    # Run the model to get predictions
    phys_out, opt_out, tox_out, arom_logit = model(data.x.float(), data.edge_index, data.batch)
    
    # Concatenate predictions from each task (normalized)
    pred_reg_norm = torch.cat([phys_out, opt_out, tox_out], dim=1).detach().cpu().numpy()  # (1, 9)
    pred_arom = arom_logit.detach().cpu().numpy().reshape(1, -1)
    
    # Apply inverse transformation to get predictions in raw scale
    pred_reg_raw = reg_inverse_transform(pred_reg_norm)  # (1, 9)
    
    # Store predictions
    predictions.append(np.concatenate([pred_reg_raw, pred_arom], axis=1))
    # predictions.append(pred_reg_raw)

# Convert predictions to a numpy array for easier processing later
predictions = np.array(predictions)

# Squeeze the second dimension (which has size 1)
predictions = np.squeeze(predictions, axis=1)  # Shape: (3000, 10)


# --- Code cell 23 ---
print(predictions.shape)


# --- Code cell 24 ---
# Load true values from CSV
true_values = test_df[["Solubility", "Molecular_Weight", "Log_P", "BoilingPoint", "sigma_780nm",
                       "max_sigma", "ISC(S1-T1)", "SAscore", "Tox_score", "is_aromatic"]].to_numpy()


# --- Code cell 25 ---
# ----------------------------
# Recording R2 & RMSE
# ----------------------------

# Initialize a list to store the results
results = []

# Compute RMSE and Pearson correlation for each property
for i, prop in enumerate(REG_COLS + [BIN_COL]):  # Includes both regression and arom_logit
    y_true = true_values[:, i]
    y_pred = predictions[:, i]

    # Filter out NaN values (only valid pairs will be considered)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"{prop}: no valid pairs (all NA)")
        continue

    # Compute RMSE
    property_rmse = rmse(y_true, y_pred)

    # Compute Pearson correlation
    property_pearson = pearson_r(y_true, y_pred)

    # Print the results for all properties, including "is_aromatic"
    print(f"{prop} RMSE: {property_rmse:.4f} , Pearson r: {property_pearson:.4f}")

    # Append the results for this property
    results.append([prop, property_pearson, property_rmse])

# Define the header for the CSV
header = ["Property", "Pearson r", "RMSE"]

# Write the results to a CSV file
output_csv = f"R2&RMSE_GCN_{POOLING_TYPE}.csv"
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header
    writer.writerows(results)  # Write the data rows

print(f"Results saved to {output_csv}")


# --- Code cell 26 ---
# ----------------------------
# Drawing R2
# ----------------------------

# Dictionary to store Pearson correlation for each property
r_by_prop = {}

# Iterate through each property
for i, prop in enumerate(REG_COLS + [BIN_COL]):  # Includes both regression and arom_logit
    y_true = true_values[:, i]
    y_pred = predictions[:, i]

    # Filter out NaN values (only valid pairs will be considered)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Skip properties with no valid data points
    if len(y_true) == 0:
        r_by_prop[prop] = np.nan
        print(f"{prop}: no valid pairs (all NA)")
        continue

    # Compute Pearson correlation
    r = pearson_r(y_true, y_pred)
    r_by_prop[prop] = r

    # Create the scatter plot for true vs predicted values
    plt.figure()
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.scatter(y_true, y_pred, s=5, color="purple")

    # Set the limits and padding for the plot
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0  # Handle constant values

    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], c="k")  # Line of perfect correlation
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)

    # Set the title and labels
    plt.title(f"{prop} (Pearson r = {r:.3f})")
    plt.xlabel(f"True {prop}")
    plt.ylabel(f"Pred {prop}")
    plt.gca().set_aspect("equal", adjustable="box")

    # Save the plot as a PDF
    plt.savefig(f"{prop}_pearsonr_GCN_{POOLING_TYPE}_plot.pdf", dpi=600, bbox_inches="tight")

    # Print the Pearson correlation for this property
    print(f"{prop}: Pearson r = {r:.3f}")

# Display all the plots
plt.show()
