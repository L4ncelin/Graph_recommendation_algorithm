from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from utils.helper import batch_iter
from tqdm import tqdm

__all__ = [
    "rating_accuracy",
    "tolerant_accuracy",
]


def _collect_explicit_pairs(dl, split: str) -> List[Tuple[int, int, float]]:
    if split == "val":
        raw = dl.get_val()
    elif split == "test":
        raw = dl.get_test()
    else:
        raise ValueError("split must be 'val' or 'test'")

    return [
        (dl.user_idx[u], dl.item_idx[i], r)
        for u, i, r in raw
        if u in dl.user_idx and i in dl.item_idx
    ]


@torch.no_grad()
def rating_accuracy(
    model,
    dl,
    *,
    split: str = "test",
    batch_size: int | None = None,
    show_progress: bool = False,
) -> float:
    pairs = _collect_explicit_pairs(dl, split)
    
    if not pairs:
        raise ValueError(f"No usable {split} pairs – did you filter them out?")

    bs = batch_size or model.batch_size
    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    user_emb, item_emb = model.propagate()
    user_emb, item_emb = user_emb.to(model.device), item_emb.to(model.device)

    iterator = batch_iter(pairs, bs, shuffle=False)
    if show_progress:
        iterator = tqdm(iterator, desc="Rating accuracy batches", unit="batch")

    for u_idx, i_idx, r in iterator:
        u_idx, i_idx = u_idx.to(model.device), i_idx.to(model.device)
        preds = model._predict_pair(user_emb[u_idx], item_emb[i_idx])
        y_true.extend(r.tolist())
        y_pred.extend(torch.round(preds).clamp(1, 5).cpu().tolist())

    return accuracy_score(y_true, y_pred)

def tolerant_accuracy(
    model,
    dl,
    *,
    split: str = "test",
    tol: int = 1,
    batch_size: int | None = None,
    show_progress: bool = False,
) -> float:
    pairs = _collect_explicit_pairs(dl, split)
    bs = batch_size or model.batch_size
    errs: list[float] = []

    model.eval()
    user_emb, item_emb = model.propagate()
    user_emb, item_emb = user_emb.to(model.device), item_emb.to(model.device)

    iterator = batch_iter(pairs, bs, shuffle=False)
    if show_progress:
        iterator = tqdm(iterator, desc="Tolerance accuracy batches", unit="batch")

    for u_idx, i_idx, r in iterator:
        u_idx, i_idx = u_idx.to(model.device), i_idx.to(model.device)
        preds = model._predict_pair(user_emb[u_idx], item_emb[i_idx])
        errs.extend((preds.cpu() - r).abs().tolist())

    errs = np.array(errs)
    return float(np.mean(errs <= tol))
