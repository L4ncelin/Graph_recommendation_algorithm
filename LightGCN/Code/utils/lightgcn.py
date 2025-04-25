from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.helper import batch_iter, _build_user_item_graph, _pairwise_sampler

__all__ = ["LightGCN"]

class LightGCN(nn.Module):

    def __init__(
        self,
        loader,
        *,
        embed_dim: int = 32,
        num_layers: int = 3,
        normalize: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 4096,
        seed: int = 42,
        rating_range: Tuple[float, float] = (1.0, 5.0),
        loss_type: str = "bpr",
        negatives: int = 5,
        early_stop_patience: int = 3,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed); random.seed(seed)

        self.loader = loader
        self.n_users = len(loader.user_idx)
        self.n_items = len(loader.item_idx)
        self.n_layers = num_layers
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = torch.device(device or "cpu")
        self.rating_min, self.rating_max = rating_range
        self.loss_type = loss_type.lower()
        self.negatives = negatives
        self.patience = early_stop_patience

        self.user_emb = nn.Embedding(self.n_users, embed_dim)
        self.item_emb = nn.Embedding(self.n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self._output_scale = nn.Parameter(torch.tensor(1.0))

        ui_ei = _build_user_item_graph(loader.get_train(), loader.user_idx, loader.item_idx)
        social_ei, _ = loader.get_social_graph()
        edge_index = torch.cat([ui_ei, social_ei], 1) if social_ei is not None and social_ei.numel() else ui_ei
        self.register_buffer("edge_index", edge_index)
        deg = torch.bincount(edge_index[0], minlength=self.n_users + self.n_items).float()
        deg_inv_sqrt = deg.pow_(-0.5); deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        self.register_buffer("deg_inv_sqrt", deg_inv_sqrt)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000) 
        self.to(self.device)

    def propagate(self):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], 0)
        outs = [ego]
        for _ in range(self.n_layers):
            row, col = self.edge_index
            msg = (self.deg_inv_sqrt[row] * self.deg_inv_sqrt[col]).unsqueeze(1) * ego[col]
            agg = torch.zeros_like(ego).index_add_(0, row, msg)
            ego = F.normalize(agg) if self.normalize else agg
            outs.append(ego)
        out = torch.mean(torch.stack(outs), 0)
        return out[: self.n_users], out[self.n_users :]

    def _predict_pair(self, u_emb, i_emb):
        inner = (u_emb * i_emb).sum(-1) * self._output_scale
        return self.rating_min + (self.rating_max - self.rating_min) * torch.sigmoid(inner)


    def fit(self, epochs: int):
        sampler   = _pairwise_sampler(self.loader, self.batch_size, self.negatives)
        val_pairs = [
            (self.loader.user_idx[u], self.loader.item_idx[i], r)
            for u, i, r in self.loader.get_val()
            if u in self.loader.user_idx and i in self.loader.item_idx
        ]
        self.scheduler.T_max = epochs

        loss_hist, rmse_hist, mae_hist = [], [], []
        best_rmse, wait = float("inf"), 0

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="ep"):
            self.train()
            user_all, item_all = self.propagate()
            u_idx, ip_idx, in_idx = (t.to(self.device) for t in next(sampler))
            y_pos = self._predict_pair(user_all[u_idx], item_all[ip_idx])
            y_neg = self._predict_pair(user_all[u_idx], item_all[in_idx])
            loss  = -torch.log(torch.sigmoid(y_pos - y_neg)).mean()
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); self.scheduler.step()
            loss_hist.append(loss.item())

            self.eval(); se = ae = 0.0
            with torch.no_grad():
                user_v, item_v = self.propagate()
                for u_idx, i_idx, r in tqdm(
                    batch_iter(val_pairs, self.batch_size, shuffle=False),
                    desc=" Val batches", 
                    leave=False,
                    unit="batch"
                ):
                    u_idx, i_idx, r = (t.to(self.device) for t in (u_idx, i_idx, r))
                    pred = self._predict_pair(user_v[u_idx], item_v[i_idx])
                    se += torch.sum((pred - r) ** 2).item(); ae += torch.sum(torch.abs(pred - r)).item()
                    
            rmse = math.sqrt(se / len(val_pairs)); mae = ae / len(val_pairs)
            rmse_hist.append(rmse); mae_hist.append(mae)
            tqdm.write(f"Epoch {epoch:3d} | BPR loss {loss:.4f} | val RMSE {rmse:.4f} | val MAE {mae:.4f}")

            if rmse < best_rmse - 2e-3:
                best_rmse, wait = rmse, 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stop (patience={self.patience}) at epoch {epoch}")
                    break

        return loss_hist, rmse_hist, mae_hist

    def evaluate_explicit(self, split="test", batch_size: int | None = None):
        if split not in {"val", "test"}:
            raise ValueError("split must be 'val' or 'test'")
        data = self.loader.get_val() if split == "val" else self.loader.get_test()
        pairs = [
            (self.loader.user_idx[u], self.loader.item_idx[i], r)
            for u, i, r in data
            if u in self.loader.user_idx and i in self.loader.item_idx
        ]
        bs = batch_size or self.batch_size; self.eval(); se = ae = 0.0
        with torch.no_grad():
            user_emb, item_emb = self.propagate()
            for u_idx, i_idx, r in batch_iter(pairs, bs, shuffle=False):
                u_idx, i_idx, r = (t.to(self.device) for t in (u_idx, i_idx, r))
                pred = self._predict_pair(user_emb[u_idx], item_emb[i_idx])
                se += torch.sum((pred - r) ** 2).item(); ae += torch.sum(torch.abs(pred - r)).item()
                
        return math.sqrt(se / len(pairs)), ae / len(pairs)
