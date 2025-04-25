import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r):

        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)
        # print(history_r)
        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]

            if self.uv == True:
                # user component
                history_tensor = torch.LongTensor(list(history)).to(self.v2e.weight.device)
                if history_tensor.min() < 0 or history_tensor.max() >= self.v2e.num_embeddings:
                    print("❌ Index out of bounds in history_tensor:", history_tensor.cpu().tolist())
                    print("Max index used:", history_tensor.max().item())
                    print("Embedding size:", self.v2e.num_embeddings)
                    exit(1)
                e_uv = self.v2e(history_tensor)
                # e_uv = self.v2e.weight[list(history)]
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]
            # print("tmp_label:", tmp_label)
            # print("max index:", max(tmp_label))
            # print("r2e size:", self.r2e.weight.shape[0])
            # e_r = self.r2e.weight[list(tmp_label)]
            e_r = self.r2e(torch.LongTensor(list(tmp_label)).to(self.r2e.weight.device))
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats