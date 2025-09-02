from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

import torch
import torch.nn as nn

from ..quantum import adjacency_from_edge_index, graph_features, build_hamiltonian, unitary


class BQGNN(nn.Module):
    """Bosonic Quantum GNN with optional MerlinQuantum layer.

    - Default: uses deterministic CTQW-based graph features (classical path).
    - Optional: if MerlinQuantum is installed and enabled, applies a photonic
      ``QuantumLayer`` to the classical feature vector before the head.
    - A small MLP head maps features to class logits.
    """

    def __init__(
        self,
        num_layers: int = 3,
        times: Optional[Iterable[float]] = None,
        spectral_moments: int = 4,
        head_hidden: int = 64,
        num_classes: int = 2,
        hamiltonian: str = "A",
        bosons: int = 1,
        # MerlinQuantum options (see docs/merlin_api_summary.md)
        merlin_enable: bool = True,
        merlin_n_modes: int = 8,
        merlin_out_dim: Optional[int] = None,
        merlin_shots: int = 0,
        merlin_no_bunching: bool = False,
        merlin_circuit_type: str = "SERIES",
        merlin_state_pattern: str = "PERIODIC",
        merlin_dtype: torch.dtype = torch.float32,
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

        # Classical feature dimension: mu_k + (Re trU, Im trU, mean_return) per t
        self._feat_dim = spectral_moments + len(self.times) * 3

        # Require MerlinQuantum (package name is "merlin"; fallback "merlinquantum")
        self._mq = None
        try:
            import merlin as mq  # type: ignore
            self._mq = mq
        except Exception:
            try:
                import merlinquantum as mq  # type: ignore
                self._mq = mq
            except Exception as e:
                raise ImportError(
                    "MerlinQuantum is required for BQGNN. Install 'merlinquantum' (Python >=3.10)."
                ) from e

        # Always use Merlin
        self._use_merlin = True
        self._merlin_out_dim = int(merlin_out_dim) if merlin_out_dim is not None else self._feat_dim
        self._merlin_cfg = {
            "n_modes": int(merlin_n_modes),
            "n_photons": int(self.bosons),
            "shots": int(merlin_shots),
            "no_bunching": bool(merlin_no_bunching),
            "circuit_type": str(merlin_circuit_type).upper(),
            "state_pattern": str(merlin_state_pattern).upper(),
            "dtype": merlin_dtype,
        }

        self._build_merlin_layer()
        head_in = self._merlin_out_dim

        # Head uses the feature vector (classical or Merlin-transformed)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_classes),
        )

    def _build_merlin_layer(self) -> None:
        """Instantiate the Merlin QuantumLayer per docs/merlin_api_summary.md."""
        assert self._mq is not None
        mq = self._mq
        # Enums
        CircuitType = getattr(mq, "CircuitType")
        StatePattern = getattr(mq, "StatePattern")
        OutputMappingStrategy = getattr(mq, "OutputMappingStrategy", None)

        ct = getattr(CircuitType, self._merlin_cfg["circuit_type"], None)
        sp = getattr(StatePattern, self._merlin_cfg["state_pattern"], None)
        if ct is None or sp is None:
            raise ValueError(
                f"Invalid merlin circuit/state: {self._merlin_cfg['circuit_type']}/{self._merlin_cfg['state_pattern']}"
            )

        # Backend
        PhotonicBackend = getattr(mq, "PhotonicBackend")
        pb = PhotonicBackend(
            circuit_type=ct,
            n_modes=self._merlin_cfg["n_modes"],
            n_photons=self._merlin_cfg["n_photons"],
            state_pattern=sp,
        )

        # Ansatz
        AnsatzFactory = getattr(mq, "AnsatzFactory")
        if OutputMappingStrategy is not None:
            oms = getattr(OutputMappingStrategy, "LINEAR")
            ansatz = AnsatzFactory.create(
                PhotonicBackend=pb,
                input_size=self._feat_dim,
                output_size=self._merlin_out_dim,
                output_mapping_strategy=oms,
                dtype=self._merlin_cfg["dtype"],
            )
        else:
            ansatz = AnsatzFactory.create(
                PhotonicBackend=pb,
                input_size=self._feat_dim,
                output_size=self._merlin_out_dim,
                dtype=self._merlin_cfg["dtype"],
            )

        # Quantum layer (torch.nn.Module)
        QuantumLayer = getattr(mq, "QuantumLayer")
        self.merlin_layer = QuantumLayer(
            input_size=self._feat_dim,
            ansatz=ansatz,
            shots=self._merlin_cfg["shots"],
            no_bunching=self._merlin_cfg["no_bunching"],
            dtype=self._merlin_cfg["dtype"],
        )

    def _features(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        a = adjacency_from_edge_index(n, edge_index)
        # Base deterministic CTQW features
        feats = graph_features(a, times=self.times, spectral_moments=self.spectral_moments)
        # Apply Merlin transformation on feature vector
        feats_in = feats.view(1, -1)
        feats_out = self.merlin_layer(feats_in)
        return feats_out.view(-1)

    def forward_single(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Keep gradients if Merlin is enabled (layer parameters are trainable)
        feats = self._features(x, edge_index)
        device = next(self.head.parameters()).device
        logits = self.head(feats.to(device))
        return logits

    def forward_batch(self, batch_graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        outs = []
        for x, ei in batch_graphs:
            outs.append(self.forward_single(x, ei))
        return torch.stack(outs, dim=0)
