# MerlinQuantum API – Practical Summary for BQGNN

This note summarizes the MerlinQuantum Python API that is directly useful for our Bosonic Quantum GNN (BQGNN) work. It is tailored by the design goals in `Tex/bosonic_qgnn.tex`: multi-photon continuous-time quantum walk–style propagation, photonic simulation, and PyTorch integration.

References
- Homepage: https://merlinquantum.ai
- API index: https://merlinquantum.ai/reference/api/modules.html
- Package API: https://merlinquantum.ai/reference/api/merlin.html
- Core API: https://merlinquantum.ai/reference/api/merlin.core.html
- Quickstart examples: https://merlinquantum.ai/quickstart/first_quantum_layer.html

## Core Concepts

- Photonic circuits simulate multi-boson interference across `n_modes` with a chosen `circuit_type` and input photon pattern. This maps cleanly to our CTQW intuition from the Tex doc (modes ≈ nodes; multi-photon interference ≈ higher-order correlations).
- A high-level PyTorch layer (`QuantumLayer`) wraps the photonic computation and integrates with autograd; it can operate deterministically or via shot-based sampling.
- An `Ansatz` describes the logical circuit template; `AnsatzFactory` builds one from a `PhotonicBackend` configuration.
- Output mapping converts quantum probability distributions to classical vectors for downstream NN processing.

## Most Useful Symbols and Typical Usage

- `merlin.core.photonicbackend.PhotonicBackend`
  - Purpose: Configuration container choosing circuit architecture and injection pattern.
  - Key args: `circuit_type`, `n_modes`, `n_photons`, `state_pattern=StatePattern.PERIODIC`.
  - Notes: `n_modes` is the number of optical modes (set relative to graph size or to a fixed embedding size); `n_photons` controls interference order (model width in our Tex terminology).

- `merlin.Ansatz` and `merlin.AnsatzFactory.create(...)`
  - Purpose: Define a logical circuit template compatible with the backend and mapping.
  - Key args: `(PhotonicBackend, input_size, output_size=None, output_mapping_strategy=OutputMappingStrategy.LINEAR, device=None, dtype=None)`.
  - Notes: If `output_size` is omitted it can be inferred; choose mapping strategy based on how we pool quantum outputs (see Output Mapping below).

- `merlin.core.QuantumLayer` (inherits `torch.nn.Module`)
  - Purpose: Differentiable quantum layer to drop into PyTorch models.
  - Signature (abridged): `QuantumLayer(input_size, output_size=None, ansatz=None, circuit=None, input_state=None, n_photons=None, trainable_parameters=(), input_parameters=(), output_mapping_strategy=OutputMappingStrategy.LINEAR, device=None, dtype=None, shots=0, sampling_method='multinomial', no_bunching=True, index_photons=None)`.
  - Shapes: input `(B, input_size)` → output `(B, output_size)` after mapping.
  - Determinism vs sampling: `shots=0` computes exact probabilities (deterministic); `shots>0` samples outcomes (stochastic). Use `sampling_method='multinomial'` for classical sampling.
  - No-bunching: `no_bunching=True` forbids multiple photons in the same mode; set `False` to allow bunching (important for multi-boson interference features).
  - Device/dtype: pass explicit `device` and `dtype` to control compute placement and precision.

- Output Mapping (quantum → classical)
  - `merlin.OutputMapper` families, e.g. `merlin.LexGroupingMapper`, `merlin.ModGroupingMapper`.
  - `forward(probability_distribution) -> Tensor`: returns `(B, output_size)` or `(output_size,)`.
  - Strategy: choose `OutputMappingStrategy` to line up with our desired readout dimensionality. For BQGNN, prefer grouping strategies that preserve permutation-invariance over patterns (e.g., mod-grouping) when mapping large outcome spaces down to fixed `output_size`.

- Encoding and Generators
  - `merlin.FeatureEncoder.encode(circuit_type, n_modes, bandwidth_coeffs=None, total_shifters=None) -> torch.Tensor` yields encoded parameter tensors of shape `(B, num_parameters)`.
  - `merlin.core.generators.CircuitGenerator.generate_circuit(...)` and `StateGenerator.generate_state(...)` exist for lower-level control; most use-cases can stay at the `AnsatzFactory`/`QuantumLayer` level.

- Process and Conversion
  - `merlin.AutoDiffProcess.autodiff_backend(needs_gradient, apply_sampling, shots)` selects a differentiable execution mode consistent with sampling.
  - `merlin.CircuitConverter(circuit, input_specs=None, dtype=torch.complex64, device='cpu')` with `.set_dtype(...)`, `.to(dtype, device)`, `.to_tensor(...)` manages dtype/device of circuit parameters.

## Recommended Patterns for BQGNN

The following aligns Merlin’s abstractions with our Tex design (multi-photon CTQW semantics and permutation-invariant outputs).

1) Backend and Ansatz

```python
from merlin import PhotonicBackend, CircuitType, StatePattern, AnsatzFactory, OutputMappingStrategy

# Choose modes and photons
photonicbackend = PhotonicBackend(
    circuit_type=CircuitType.SERIES,        # or PARALLEL / PARALLEL_COLUMNS
    n_modes=n_nodes_or_embed_dim,          # tie to graph size or a fixed embedding
    n_photons=num_bosons,                  # model width via multi-boson interference
    state_pattern=StatePattern.PERIODIC    # PERIODIC or SEQUENTIAL depending on design
)

ansatz = AnsatzFactory.create(
    PhotonicBackend=photonicbackend,
    input_size=in_dim,                      # per-graph feature dim we encode
    output_size=out_dim,                    # desired classical output channels
    output_mapping_strategy=OutputMappingStrategy.LINEAR,  # or NONE + custom mapper
)
```

2) Quantum Layer in PyTorch

```python
import torch
from merlin import QuantumLayer

qlayer = QuantumLayer(
    input_size=in_dim,
    ansatz=ansatz,
    shots=0,                 # deterministic; set >0 for sampling (stochastic)
    no_bunching=False,       # allow multi-boson bunching effects if desired
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    dtype=torch.float32,
)

# B x in_dim → B x out_dim
y = qlayer(x)
```

3) Mapping Strategy and Invariance

- If using full outcome distributions, apply an explicit mapper such as `LexGroupingMapper` or `ModGroupingMapper` to reduce to `(B, out_dim)` while encouraging permutation-invariant summaries.
- Alternatively, stick with `OutputMappingStrategy.LINEAR` and aggregate with our classical heads (e.g., sum/mean-pooling in GNN heads) ensuring permutation-invariance across graph permutations.

4) Determinism and Gradients

- Exact mode: `shots=0` is deterministic. For CI and reproducibility, prefer this during unit tests.
- Sampling mode: set `shots>0` and control randomness via `torch.manual_seed(seed)`. Autograd support is routed through `AutoDiffProcess.autodiff_backend(...)`; prefer higher precision (`float64`) if doing gradient-sensitive tests, acknowledging increased compute.

5) Dtype and Device

- Use `.to(dtype, device)` on `CircuitConverter` if managing custom circuits; otherwise, pass `dtype`/`device` to `QuantumLayer` construction.
- Complex tensor dtypes are implied internally; user-facing `dtype` is typically float32/float64 for probabilities and mapped outputs.

## Shapes and Interfaces (at a glance)

- Inputs: `x: torch.Tensor` with shape `(B, input_size)` and dtype float32/float64.
- Outputs: `y: torch.Tensor` with shape `(B, output_size)` and same float dtype.
- Mapper: probability distribution tensors are reduced to `(B, output_size)` via grouping; without mappers, `OutputMappingStrategy` in the ansatz handles readout.

## Error Modes and Gotchas

- Dtype mismatches: `CircuitConverter.set_dtype(...)` restricts to pairs (float32/complex64 or float64/complex128).
- Device: invalid device strings or unsupported device types raise `TypeError` in `.to(...)`.
- Sampling: stochastic outputs require fixed seeds for reproducibility; gradients through sampling depend on the configured autodiff backend.
- Bunching: set `no_bunching=False` if the model relies on many-boson interference; enabling no-bunching may mute key effects.

## How This Fits Our Codebase

- `src/bqgnn/models/bqgnn.py` already contains a placeholder import of `merlinquantum` and a hook for a bosonic CTQW embedding.
- Recommended integration path:
  1) Add an optional `MerlinQuantumBlock(nn.Module)` that internally builds `PhotonicBackend → Ansatz → QuantumLayer` when `merlinquantum` is available.
  2) Expose flags in training configs to toggle `shots`, `no_bunching`, `n_modes`, `n_photons`, and `output_mapping_strategy`.
  3) In deterministic test runs, set `shots=0` and `torch.use_deterministic_algorithms(True)`; pin seeds for stochastic runs.

## Minimal Example (Factory route)

```python
from merlin import PhotonicBackend, CircuitType, StatePattern, AnsatzFactory, QuantumLayer, OutputMappingStrategy
import torch, torch.nn as nn

class MerlinQuantumBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_modes: int, n_photons: int,
                 shots: int = 0, no_bunching: bool = False, device=None, dtype=torch.float32):
        super().__init__()
        pb = PhotonicBackend(
            circuit_type=CircuitType.SERIES,
            n_modes=n_modes,
            n_photons=n_photons,
            state_pattern=StatePattern.PERIODIC,
        )
        ansatz = AnsatzFactory.create(
            PhotonicBackend=pb,
            input_size=in_dim,
            output_size=out_dim,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            device=device,
            dtype=dtype,
        )
        self.layer = QuantumLayer(
            input_size=in_dim,
            ansatz=ansatz,
            shots=shots,
            no_bunching=no_bunching,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
```

Swap `CircuitType`, `StatePattern`, and mapping strategies to explore design space consistent with the Tex document’s CTQW-based motivation.

## Links to Specific API Sections

- Photonic backend and state patterns: https://merlinquantum.ai/reference/api/merlin.core.photonicbackend.html
- Ansatz and factory: https://merlinquantum.ai/reference/api/merlin.core.ansatz.html
- Layer: https://merlinquantum.ai/reference/api/merlin.core.layer.html
- Process and converters: https://merlinquantum.ai/reference/api/merlin.core.process.html
- Generators: https://merlinquantum.ai/reference/api/merlin.core.generators.html
