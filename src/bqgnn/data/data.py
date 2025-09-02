from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class GraphData:
    x: torch.Tensor  # [N, F]
    edge_index: torch.Tensor  # [2, E] (dst aggregation: sum h[src] into dst)
    y: int

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)


def _ensure_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # add reverse edges and remove duplicates
    src, dst = edge_index
    rev = torch.stack([dst, src], dim=0)
    ei = torch.cat([edge_index, rev], dim=1)
    # unique pairs
    # Convert to linear indices: i * N + j
    idx = ei[0] * num_nodes + ei[1]
    uniq, inv = torch.unique(idx, sorted=False, return_inverse=True)
    # Map back to edges
    dst_unique = uniq % num_nodes
    src_unique = uniq // num_nodes
    return torch.stack([src_unique, dst_unique], dim=0)


def _features_or_default(x: torch.Tensor | None, edge_index: torch.Tensor, num_nodes: int,
                         mode: str = "degree") -> torch.Tensor:
    if x is not None and x.numel() > 0:
        return x.float()
    # default features
    mode = (mode or "degree").lower()
    if mode == "degree":
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
        return deg.view(-1, 1)
    # constant ones
    return torch.ones((num_nodes, 1), dtype=torch.float32)


def load_tu_dataset(name: str, root: str = "data/tu", default_feat: str = "degree") -> Tuple[List[GraphData], List[int]]:
    """Load TU datasets (MUTAG, NCI1, PROTEINS) into a simple GraphData list.

    Attempts torch_geometric first, then DGL as a fallback. Returns (graphs, labels).
    """
    os.makedirs(root, exist_ok=True)

    name = name.upper()
    name_map = {
        "PROTEINS": "PROTEINS",
        "PROTEINS_FULL": "PROTEINS",
        "MUTAG": "MUTAG",
        "NCI1": "NCI1",
    }
    std_name = name_map.get(name, name)

    # Try PyTorch Geometric
    graphs: List[GraphData] = []
    labels: List[int] = []
    try:
        from torch_geometric.datasets import TUDataset  # type: ignore
        pyg_ds = TUDataset(root=root, name=std_name)
        for data in pyg_ds:
            x = data.x if hasattr(data, "x") else None
            ei = data.edge_index.long()
            y_item = int(data.y.item()) if hasattr(data, "y") else 0
            n = int(data.num_nodes)
            ei = _ensure_undirected(ei, n)
            x = _features_or_default(x, ei, n, mode=default_feat)
            graphs.append(GraphData(x=x, edge_index=ei, y=y_item))
            labels.append(y_item)
        return graphs, labels
    except Exception:
        pass

    # Try DGL
    try:
        # Prefer modern TUDataset; fallback to LegacyTUDataset for older DGL
        try:
            from dgl.data import TUDataset  # type: ignore
            dgl_ds = TUDataset(name=std_name, raw_dir=root)
        except Exception:
            from dgl.data import LegacyTUDataset  # type: ignore
            dgl_ds = LegacyTUDataset(std_name, hidden_size=1, raw_dir=root)
        for g, y in zip(dgl_ds.graph_lists, dgl_ds.graph_labels):
            n = g.number_of_nodes()
            # Build edge_index (dst aggregation)
            src, dst = g.edges(order="eid")
            src = src.long()
            dst = dst.long()
            ei = torch.stack([src, dst], dim=0)
            ei = _ensure_undirected(ei, n)
            # Node features if present
            x = g.ndata.get("feat")
            x = _features_or_default(x, ei, n, mode=default_feat)
            y_item = int(y)
            graphs.append(GraphData(x=x, edge_index=ei, y=y_item))
            labels.append(y_item)
        return graphs, labels
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset {std_name}. Install torch_geometric or dgl. Original error: {e}"
        )
