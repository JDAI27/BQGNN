from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINLayer(nn.Module):
    def __init__(self, hidden_dim: int, learn_eps: bool = True, dropout: float = 0.0, use_batchnorm: bool = True):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1)) if learn_eps else None
        self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_batchnorm else None
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def aggregate(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index: [2, E] with [src, dst]
        src, dst = edge_index
        out = torch.zeros_like(h)
        out.index_add_(0, dst, h.index_select(0, src))
        return out

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = self.aggregate(h, edge_index)
        if self.eps is not None:
            out = (1.0 + self.eps) * h + agg
        else:
            out = h + agg
        out = self.mlp(out)
        if self.bn is not None:
            out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class GIN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int,
                 dropout: float = 0.0, learn_eps: bool = True, readout: str = "sum"):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, learn_eps=learn_eps, dropout=dropout, use_batchnorm=True)
            for _ in range(num_layers)
        ])
        # Jumping knowledge: sum of layer-wise pooled predictions (as in Xu et al.)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers + 1)])
        assert readout in {"sum", "mean"}
        self.readout = readout

    @staticmethod
    def pool(h: torch.Tensor, readout: str = "sum") -> torch.Tensor:
        if readout == "sum":
            return h.sum(dim=0, keepdim=True)
        else:
            return h.mean(dim=0, keepdim=True)

    def forward_single(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        reps = [h]
        for layer in self.layers:
            h = layer(h, edge_index)
            reps.append(h)
        logits = 0.0
        for i, rep in enumerate(reps):
            pooled = self.pool(rep, self.readout)  # [1, hidden]
            logits = logits + self.heads[i](pooled)  # [1, C]
        return logits.squeeze(0)

    def forward_batch(self, batch_graphs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # batch_graphs: list of (x, edge_index)
        outs = []
        for x, ei in batch_graphs:
            outs.append(self.forward_single(x, ei))
        return torch.stack(outs, dim=0)

