import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt

from utils.helper import *
from utils.data_loader import * 
from utils.lightgcn import *
from utils.plot_curves import *
from utils.metrics import *
from utils.predictions import *


data_path  = '../Datasets/rating.txt'
trust_path = '../Datasets/trustnetwork.txt'

dl = DataLoader(
    data_path=data_path,
    trust_path=trust_path,
    data_name='Ciao',
    keep_unknown=False
)

edge_index, edge_weight = dl.get_social_graph()

print(f"Social graph has {edge_index.shape[1]} edges")

print("Computing Ciao statistics...")

_ = compute_ciao_statistics(data_path, trust_path=trust_path, verbose=True)

embed_dim = 32
num_layers = 5
normalize = True
lr = 1e-2
weight_decay = 1e-5
batch_size = 2048
seed = 42
rating_range = (1.0, 5.0) # scaled
loss_type = "bpr"
negatives = 5
early_stop_patience = 3

epochs = 20

model = LightGCN(
    dl,
    embed_dim=embed_dim,
    num_layers=num_layers,
    normalize=normalize,
    lr=lr,
    weight_decay=weight_decay,
    batch_size=batch_size,
    seed=seed,
    rating_range=rating_range,
    loss_type=loss_type,
    negatives=negatives,
    early_stop_patience=early_stop_patience
)

bpr_loss, val_rmse, val_mae = model.fit(epochs=epochs)

model_path = f"../Model/lightgcn_bpr_{num_layers}_layers_{negatives}_negatives.pth"

os.makedirs("../Model", exist_ok=True)

torch.save(model.state_dict(), model_path)
print(f"Model saved to : {model_path}")

save_path = f'../Plots/{epochs}_epochs_early_stop.png'

#plot_training_curves(train_loss=bpr_loss, rmse=val_rmse, mae=val_mae, save_path=save_path)

eval_pairs = [
    (dl.user_idx[u], dl.item_idx[i], r)
    for u, i, r in dl.get_val()
    if u in dl.user_idx and i in dl.item_idx
]

interacted = {}
for u, i, _ in dl.get_train() + dl.get_val():
    if u in dl.user_idx and i in dl.item_idx:
        interacted.setdefault(dl.user_idx[u], set()).add(dl.item_idx[i])

all_items = list(dl.item_idx.values())

hr, ndcg = topk_metrics(
    model, eval_pairs,
    all_items=all_items,
    interacted=interacted,
    K=10,
    n_neg=99,
    device=model.device
)

print(f"HR@10 = {hr:.4f} | NDCG@10 = {ndcg:.4f}")

acc  = rating_accuracy(model, dl, split="test")         
tol_acc    = tolerant_accuracy(model, dl, split="test", tol=1) 

print(f"Accuracy = {acc:.3f}")
print(f"±1 star accuracy = {tol_acc:.3f}")

negatives_list = [1, 5, 10, 15, 20, 25]

hr_list       = []
ndcg_list     = []
acc_list      = []
tol_acc_list  = []

for K in negatives_list:
    print(f"\n▶ Training with negatives = {K}")

    model = LightGCN(
        dl,
        embed_dim=embed_dim,
        num_layers=num_layers,
        normalize=normalize,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        rating_range=rating_range,
        loss_type="bpr",
        negatives=K,
        early_stop_patience=early_stop_patience
    )
    _ = model.fit(epochs=epochs)

    hr, ndcg = topk_metrics(
        model,
        eval_pairs,
        all_items=all_items,
        interacted=interacted,
        K=10,
        n_neg=99,
        device=model.device
    )
    print(f"HR@10 = {hr:.4f} | NDCG@10 = {ndcg:.4f}")

    acc     = rating_accuracy(model, dl, split="test")
    tol_acc = tolerant_accuracy(model, dl, split="test", tol=1)
    print(f"Accuracy = {acc:.3f} | ±1-star accuracy = {tol_acc:.3f}")

    hr_list.append(hr)
    ndcg_list.append(ndcg)
    acc_list.append(acc)
    tol_acc_list.append(tol_acc)
    
save_path_cv = f'../Plots/all_metrics_vs_negatives.png'

plot_all_metrics_vs_negatives(
    hr_list=hr_list,
    ndcg_list=ndcg_list,
    acc_list=acc_list,
    tol_acc_list=tol_acc_list,
    negatives=negatives_list,
    K=10,
    save_path=save_path_cv
)