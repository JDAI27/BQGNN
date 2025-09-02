from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from ..quantum import adjacency_from_edge_index, graph_features, build_hamiltonian, unitary


class BQGNN(nn.Module):
    """Bosonic Quantum GNN (scaffold)

    - Computes permutation-invariant CTQW features from the graph structure.
    - Optionally integrates with merlinquantum if installed (future hook).
    - A small MLP head maps features to class logits.
    """

    def __init__(
        self,
        num_layers: int = 3,
        times: Iterable[float] | None = None,
        spectral_moments: int = 4,
        head_hidden: int = 64,
        num_classes: int = 2,
        hamiltonian: str = "A",
        bosons: int = 1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        if times is None:
            self.times = [0.2 * (i + 1) for i in range(num_layers)]
        else:
            self.times = list(times)
        self.spectral_moments = spectral_moments
        self.hamiltonian = hamiltonian
        self.bosons = bosons

        feat_dim = spectral_moments + len(self.times) * 3  # mu_k + (Re trU, Im trU, mean_return)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_classes),
        )

        # MerlinQuantum hook
        self._mq = None
        try:
            import merlinquantum as mq  # type: ignore

            self._mq = mq
        except Exception:
            self._mq = None

    def _features(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        a = adjacency_from_edge_index(n, edge_index)

        if self._mq is not None and self.bosons > 1:
            # Placeholder hook: if merlinquantum provides a bosonic CTQW embedding API, call it here.
            # Example (pseudo): feats = self._mq.ctqw.bosonic_features(a.cpu().numpy(), self.times, self.bosons)
            # Convert to torch tensor afterward. For now, fall back to our features.
            pass

        feats = graph_features(a, times=self.times, spectral_moments=self.spectral_moments)
        return feats

    def forward_single(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self._features(x, edge_index)
        device = next(self.head.parameters()).device
        logits = self.head(feats.to(device))
        return logits

    def forward_batch(self, batch_graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        outs = []
        for x, ei in batch_graphs:
            outs.append(self.forward_single(x, ei))
        return torch.stack(outs, dim=0)
