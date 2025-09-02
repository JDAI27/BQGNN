from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple

import torch


def adjacency_from_edge_index(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    a = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    src, dst = edge_index
    a[dst, src] = 1.0
    a[src, dst] = 1.0
    return a


def build_hamiltonian(a: torch.Tensor, kind: str = "A", add_self_loops: bool = False) -> torch.Tensor:
    kind = (kind or "A").lower()
    n = a.size(0)
    if add_self_loops:
        a = a.clone()
        a.fill_diagonal_(1.0)
    if kind in {"a", "adj", "adjacency"}:
        return a
    # Laplacian
    deg = torch.diag(a.sum(dim=1))
    L = deg - a
    if kind in {"l", "laplacian"}:
        return L
    if kind in {"l_norm", "normalized_laplacian", "nl"}:
        d = a.sum(dim=1)
        dinv_sqrt = torch.where(d > 0, d.pow(-0.5), torch.zeros_like(d))
        Dm12 = torch.diag(dinv_sqrt)
        return torch.eye(n, device=a.device) - Dm12 @ a @ Dm12
    return a


def eigh_cached(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Cache eigendecomposition per-graph for performance
    # Attach cache to the function object
    if not hasattr(eigh_cached, "_cache"):
        eigh_cached._cache = {}
    key = hashlib.blake2b(H.detach().cpu().numpy().tobytes(), digest_size=16).hexdigest()
    cache = eigh_cached._cache
    if key in cache:
        return cache[key]
    # Symmetric by construction
    evals, evecs = torch.linalg.eigh(H)
    cache[key] = (evals.detach(), evecs.detach())
    return cache[key]


def unitary(H: torch.Tensor, t: float) -> torch.Tensor:
    evals, evecs = eigh_cached(H)
    evals_c = evals.to(torch.complex64)
    evecs_c = evecs.to(torch.complex64)
    phases = torch.exp(-1j * evals_c * t)
    U = evecs_c @ torch.diag(phases) @ evecs_c.conj().T
    return U


def graph_features(
    a: torch.Tensor,
    times: Iterable[float],
    spectral_moments: int = 4,
) -> torch.Tensor:
    """Compute permutation-invariant CTQW features.

    - Spectral moments mu_k = (1/N) sum lambda^k, k=1..K
    - For each t in times:
      - trU/N (real, imag)
      - mean return probability: mean_i |U_ii|^2
    Returns a 1D feature tensor.
    """
    H = build_hamiltonian(a, kind="A")
    evals, _ = eigh_cached(H)
    n = a.size(0)
    feats: List[torch.Tensor] = []

    # Spectral moments
    for k in range(1, spectral_moments + 1):
        mu_k = (evals**k).mean()
        feats.append(mu_k.to(torch.float32).view(1))

    for t in times:
        U = unitary(H, float(t))
        trU = torch.trace(U)
        feats.append(trU.real.to(torch.float32).view(1) / n)
        feats.append(trU.imag.to(torch.float32).view(1) / n)
        # Return prob
        diagU = torch.diagonal(U)
        mean_ret = (diagU.conj() * diagU).real.mean().to(torch.float32).view(1)
        feats.append(mean_ret)

    return torch.cat(feats, dim=0)
