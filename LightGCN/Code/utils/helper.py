from __future__ import annotations

import math
import random
import torch

from pathlib import Path
from typing import List, Tuple
from utils.data_loader import DataLoader 


__all__ = [
    "compute_ciao_statistics",
    "batch_iter",
    "_pairwise_sampler",
    "_build_user_item_graph"
]

def compute_ciao_statistics(
    data_path: str | Path,
    trust_path: str | Path,
    verbose: bool = True
) -> dict:

    dummy_loader = DataLoader(
        data_path=data_path,
        trust_path=trust_path,
        keep_unknown=True,
    )
    interactions = dummy_loader._data
    n_users = len(dummy_loader.user_idx)
    n_items = len(dummy_loader.item_idx)
    n_int = len(interactions)
    density = n_int / (n_users * n_items)
    sparsity = 1.0 - density

    stats = dict(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_int,
        density=density,
        sparsity=sparsity,
    )
    
    if verbose:
        print(f"STATISTICS FOR Ciao:")
        print(f"  Number of users:        {n_users:>10d}")
        print(f"  Number of items:        {n_items:>10d}")
        print(f"  Number of interactions: {n_int:>10d}")
        print(f"  Density:                {density:.6f}")
        print(f"  Sparsity:               {sparsity:.6f}")
        
    return stats


def batch_iter(
    data: List[Tuple[int, int, float]],
    batch_size: int = 64,
    shuffle: bool = True
):
    idxs = list(range(len(data)))
    
    if shuffle:
        import random
        random.shuffle(idxs)
        
    for i in range(0, len(data), batch_size):
        sub = [data[j] for j in idxs[i : i + batch_size]]
        
        if not sub:
            continue
        
        u, it, r = zip(*sub)
        
        yield (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(it, dtype=torch.long),
            torch.tensor(r, dtype=torch.float),
        )


def _build_user_item_graph(train_data, user_idx, item_idx):
    edges = [
        (user_idx[u], item_idx[i] + len(user_idx))
        for u, i, _ in train_data
        if u in user_idx and i in item_idx
    ]
    
    if not edges:
        raise RuntimeError("No user-item edges built - check train_data filters.")
    
    ei = torch.tensor(edges, dtype=torch.long).t()
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    
    return ei.contiguous()


def _pairwise_sampler(loader, batch_size, K=1, seed=42):
    rng = random.Random(seed)
    pos_dict: dict[str, list[str]] = {}
    
    for u, i, _ in loader.get_train():
        pos_dict.setdefault(u, []).append(i)

    all_items = list(loader.item_idx.keys())
    users = list(pos_dict.keys())

    while True:
        bu, bip, bin_ = [], [], []
        
        while len(bu) < batch_size:
            u = rng.choice(users)
            ip = rng.choice(pos_dict[u])
            
            for _ in range(K):
                ineg = ip
                
                while ineg in pos_dict[u]:
                    ineg = rng.choice(all_items)
                    
                bu.append(u)
                bip.append(ip)
                bin_.append(ineg)
                
        yield (
            torch.tensor(
                [loader.user_idx[x] for x in bu],
                dtype=torch.long
            ),
            torch.tensor(
                [loader.item_idx[x] for x in bip], 
                dtype=torch.long
            ),
            torch.tensor(
                [loader.item_idx[x] for x in bin_],
                dtype=torch.long
            ),
        )