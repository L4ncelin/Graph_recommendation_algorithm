from __future__ import annotations

import random
import re

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.model_selection import train_test_split

__all__ = ["DataLoader"]


class DataLoader:
    min_rating = 10.0
    max_rating = 50.0

    def __init__(
        self,
        data_path: str | Path,
        trust_path: str | Path,
        data_name: str = "Ciao",
        test_size: float = 0.2,
        seed: int = 42,
        keep_unknown: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.trust_path = Path(trust_path)
        self.data_name = data_name
        self.test_size = test_size
        self.seed = seed
        self.keep_unknown = keep_unknown

        self._data: List[Tuple[str, str, float]] = [] # (user, item, rating)
        self.train_data: List[Tuple[str, str, float]] = []
        self.val_data: List[Tuple[str, str, float]] = []
        self.test_data: List[Tuple[str, str, float]] = []
        
        self.user_idx: Dict[str, int] = {}
        self.item_idx: Dict[str, int] = {}
        
        self.social_edge_index: torch.Tensor | None = None
        self.social_edge_weight: torch.Tensor | None = None

        self._load_and_split()

    def get_train(self):
        return self.train_data

    def get_val(self):
        return self.val_data

    def get_test(self):
        return self.test_data

    def get_social_graph(self):
        return self.social_edge_index, self.social_edge_weight

    @staticmethod
    def _extract_triples(line: str) -> List[Tuple[str, str, float]]:
        line = line.rstrip()
        if not line:
            return []

        try:
            user_id, rest = line.split("::::", 1)
        except ValueError: 
            return []

        triples: List[Tuple[str, str, float]] = []

        parts = rest.split("::::")
        if len(parts) >= 3:
            try:
                rating_fast = float(parts[2])
                
            except ValueError:
                rating_fast = None
                
            if rating_fast is not None and DataLoader.min_rating <= rating_fast <= DataLoader.max_rating:
                scaled = 1.0 + 4.0 * (rating_fast - 10.0) / 40.0
                triples.append((user_id, parts[0], scaled))
                rest = "::::".join(parts[3:])
                
            else:
                rest = "::::".join(parts)

        for item_id, r_str in re.findall(r"(\d+):([1-5])", rest):
            triples.append((user_id, item_id, float(r_str)))

        return triples

    def _load_and_split(self) -> None:
        with self.data_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="Parsing ratings", unit=" lines"):
                self._data.extend(self._extract_triples(line))

        if not self._data:
            raise RuntimeError("Parsed zero rating triples – check file format.")

        train, rest = train_test_split(
            self._data,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
        )
        val, test = train_test_split(
            rest,
            test_size=0.5,
            random_state=self.seed,
            shuffle=True,
        )
        self.train_data, self.val_data, self.test_data = train, val, test

        users = sorted({u for u, _, _ in train})
        items = sorted({i for _, i, _ in train})
        
        self.user_idx = {u: idx for idx, u in enumerate(users)}
        self.item_idx = {i: idx for idx, i in enumerate(items)}

        if self.keep_unknown:
            offset_u = len(self.user_idx)
            offset_i = len(self.item_idx)
            
            new_users = sorted({u for u, _, _ in (val + test) if u not in self.user_idx})
            new_items = sorted({i for _, i, _ in (val + test) if i not in self.item_idx})
            
            self.user_idx.update({u: offset_u + k for k, u in enumerate(new_users)})
            self.item_idx.update({i: offset_i + k for k, i in enumerate(new_items)})

        self._build_social_graph()

    def _build_social_graph(self):
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []

        if not self.trust_path.exists():
            self.social_edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )
            self.social_edge_weight = torch.empty(
                (0,), 
                dtype=torch.float
            )
            
            return

        with self.trust_path.open(
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            for line in tqdm(f, desc="Parsing trust edges...", unit=" lines"):
                parts = [token for token in line.rstrip().split("::::") if token]
                
                if len(parts) < 2:
                    continue
                
                u, v = parts[0], parts[1]

                w = 1.0
                if len(parts) >= 3:
                    try:
                        w = float(parts[2])
                        
                    except ValueError:
                        pass 

                if u in self.user_idx and v in self.user_idx:
                    edges.append((self.user_idx[u], self.user_idx[v]))
                    weights.append(w)

        self.social_edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        self.social_edge_weight = (
            torch.tensor(weights, dtype=torch.float)
            if weights
            else torch.empty((0,), dtype=torch.float)
        )