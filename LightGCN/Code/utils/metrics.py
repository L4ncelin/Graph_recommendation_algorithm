from __future__ import annotations

import math
import random
import torch
from typing import List, Tuple
from tqdm import tqdm

__all__ = ["topk_metrics"]

@torch.no_grad()
def topk_metrics(
    model,
    eval_pairs: List[Tuple[int, int, float]],
    *,
    all_items: List[int],
    interacted: dict[int, set[int]],
    K: int = 10,
    n_neg: int = 99,
    device="cpu",
):
    rng = random.Random(42)
    hit = ndcg = 0.0

    model.eval()
    user_emb, item_emb = model.propagate()
    user_emb, item_emb = user_emb.to(device), item_emb.to(device)

    for u, i_pos, _ in tqdm(eval_pairs, desc="Evaluating users", unit="user"):
        cand = [i_pos]
        
        while len(cand) < n_neg + 1:
            j = rng.choice(all_items)
            
            if j not in interacted[u]:
                cand.append(j)

        cand_tensor = torch.tensor(cand, dtype=torch.long, device=device)
        scores = model._predict_pair(
            user_emb[u].unsqueeze(0),
            item_emb[cand_tensor]
        ).squeeze()

        rank = torch.argsort(scores, descending=True)
        pos_rank = (rank == 0).nonzero(as_tuple=True)[0].item()
        if pos_rank < K:
            hit += 1
            ndcg += 1.0 / math.log2(pos_rank + 2)

    n_users = len(eval_pairs)
    hr = hit / n_users
    ndcg_k = ndcg / n_users
    
    return hr, ndcg_k