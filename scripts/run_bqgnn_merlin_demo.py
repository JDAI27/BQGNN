#!/usr/bin/env python
from __future__ import annotations

import math
import os
import random
import tempfile
from typing import List, Tuple

import torch
import torch.nn as nn

# Direct some caches to a writable temp dir to quiet warnings
_tmp = os.path.join(tempfile.gettempdir(), "bqgnn_merlin_cache")
os.makedirs(_tmp, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _tmp)
os.environ.setdefault("XDG_CACHE_HOME", _tmp)

# Ensure local package is discoverable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bqgnn.models.bqgnn import BQGNN


def make_chain_graph(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.ones((n, 1), dtype=torch.float32)
    src = torch.arange(0, n - 1, dtype=torch.long)
    dst = torch.arange(1, n, dtype=torch.long)
    ei = torch.stack([src, dst], dim=0)
    return x, ei


def make_star_graph(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.ones((n, 1), dtype=torch.float32)
    center = torch.tensor([0], dtype=torch.long).repeat(n - 1)
    leaves = torch.arange(1, n, dtype=torch.long)
    ei = torch.stack([center, leaves], dim=0)
    return x, ei


def batch_iter(
    dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: List[int],
    batch_size: int,
    shuffle: bool = True,
):
    idx = list(range(len(dataset)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i : i + batch_size]
        yield [dataset[j] for j in sel], torch.tensor([labels[j] for j in sel], dtype=torch.long)


def main():
    # Reproducibility
    torch.manual_seed(12345)
    random.seed(12345)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny synthetic dataset: chains (class 0) vs stars (class 1)
    graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    labels: List[int] = []
    for _ in range(6):
        graphs.append(make_chain_graph(4))
        labels.append(0)
    for _ in range(6):
        graphs.append(make_star_graph(5))
        labels.append(1)

    # Model with Merlin enabled as a post-feature quantum layer
    model = BQGNN(
        num_layers=2,
        spectral_moments=3,
        head_hidden=16,
        num_classes=2,
        bosons=2,
        merlin_enable=True,
        merlin_n_modes=8,
        merlin_out_dim=None,  # defaults to classical feature dim
        merlin_shots=0,       # deterministic path
        merlin_no_bunching=False,
        merlin_circuit_type="SERIES",
        merlin_state_pattern="PERIODIC",
        merlin_dtype=torch.float32,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    # Quick training
    epochs = 10
    batch_size = 4
    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for batch_graphs, y in batch_iter(graphs, labels, batch_size, shuffle=True):
            x_ei = [(x.to(device), ei.to(device)) for (x, ei) in batch_graphs]
            logits = model.forward_batch(x_ei)
            loss = criterion(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.detach().item() * y.size(0)
            pred = logits.argmax(dim=1).detach().cpu()
            correct += int((pred == y).sum())
            total += y.size(0)
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        if ep % 2 == 0 or ep == 1:
            print(f"[ep {ep:02d}] loss={train_loss:.4f} acc={train_acc:.3f}")

    # Final sanity prediction
    model.eval()
    with torch.no_grad():
        logits = model.forward_batch([(g[0].to(device), g[1].to(device)) for g in graphs[:2]])
        probs = logits.softmax(dim=1)
    print("probs (first 2 graphs):", probs.cpu().numpy())


if __name__ == "__main__":
    main()
