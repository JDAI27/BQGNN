# BQGNN + Merlin: Design Options and Justifications

This document consolidates the choices implied by the Tex (bosonic QGNN/CTQW) and maps them to the MerlinQuantum API, with rationale, trade‑offs, and when to choose which. Use this as a menu; you can mix across axes.

## Goals and Constraints
- Goal: exploit multi‑boson interference to exceed 1‑WL message‑passing expressivity while remaining trainable in PyTorch.
- Invariance: predictions should be permutation‑invariant over graph isomorphisms.
- Determinism: enable exact, reproducible runs for CI and ablations (shots=0), and sampling for hardware realism.
- Practicality: keep parameter count, wall time, and memory within typical TU dataset scales.

## Axis A — Walk Hamiltonian H (graph encoding)
Options
- A1 Adjacency A: U(t)=exp(−i A t)
  - Pros: standard CTQW; aligns propagation with adjacency spectrum; simple; commonly used in Tex.
  - Cons: sensitive to degree scale; may over‑emphasize hubs.
- A2 Combinatorial Laplacian L=D−A
  - Pros: diffusion‑like spectrum; degree‑normalized stages; robust to degree variance.
  - Cons: dynamics differ by global phase only on regular graphs; subtle differences elsewhere.
- A3 Normalized Laplacian I − D^−1/2 A D^−1/2
  - Pros: scale‑free spectrum; better for heterogeneous graphs.
  - Cons: extra pre/post scaling; numerical stability considerations for isolated nodes.
Implementation
- Our code path: `build_hamiltonian(kind)` supports A | L | L_norm.
- Merlin mapping: H influences classical features; Merlin then transforms the feature vector.
When to choose
- A: small, relatively regular graphs; want “edge‑adjacency energy” features.
- L: emphasize cuts/partitions; reduce hub dominance.
- L_norm: degree‑heterogeneous datasets; scale invariance matters.

## Axis B — Number of Bosons (photons) M
Options
- B1 Single boson (M=1)
  - Pros: lowest complexity; dynamics akin to spectral filters.
  - Cons: expressivity close to classical spectral features; limited multi‑particle correlations.
- B2 Multi‑boson (M≥2)
  - Pros: multi‑mode interference and bunching enable higher‑order correlations; better chance to exceed 1‑WL.
  - Cons: state space grows combinatorially O(C(N+M−1,M)); potential to overfit; cost increases with M.
Implementation
- Merlin: `n_photons = M` in `PhotonicBackend`.
When to choose
- Start with M=2 for clear multi‑boson effects and acceptable cost; increase M only with performance evidence and available compute.

## Axis C — Bunching Constraint
Options
- C1 no_bunching=True (forbid multiple photons per mode)
  - Pros: constrains state space; stabilizes training; reduces variance.
  - Cons: removes hallmark multi‑boson bunching phenomena; may undercut expressivity.
- C2 no_bunching=False (allow bunching)
  - Pros: captures HOM‑like interference; leverages full bosonic statistics.
  - Cons: larger effective space; higher variance; may require more shots/assets to estimate reliably.
Implementation
- Merlin: `no_bunching` flag on `QuantumLayer` (we typically recommend False for CTQW expressivity studies).
When to choose
- Start with False (allow bunching) when targeting expressivity; consider True if instability appears.

## Axis D — Number of Modes N_modes (photonic lattice size)
Options
- D1 Tie to graph size (N_modes = |V|)
  - Pros: closest physical analogy (one mode per node); clean mapping to graph structure.
  - Cons: batch heterogeneity (variable N_modes) complicates weight sharing and caching.
- D2 Fixed embedding size (e.g., N_modes=8/16)
  - Pros: stable layer shape for batching and deployment; faster compile/convert paths.
  - Cons: requires a graph→mode embedding (e.g., spectral or node sampling); may lose fidelity on larger graphs.
Implementation
- Merlin: `n_modes` in `PhotonicBackend`.
When to choose
- TU datasets with small graphs → D1; mixed sizes or deployment targets requiring fixed kernels → D2.

## Axis E — Input State Pattern
Options
- E1 PERIODIC (e.g., 101010...) — evenly spaced injections
  - Pros: spreads photons; low mode‑bias; good default.
- E2 SEQUENTIAL (e.g., 111000...) — fill the first modes
  - Pros: stronger locality; may benefit graphs with localized patterns.
- E3 SPACED / DEFAULT — library presets
  - Pros: quick alternatives if PERIODIC/SEQUENTIAL underperform.
Implementation
- Merlin: `state_pattern` in `PhotonicBackend` (`StatePattern.PERIODIC|SEQUENTIAL|...`).
When to choose
- Use PERIODIC as default; probe SEQUENTIAL for datasets with hub/leaf polarity.

## Axis F — Walk Depth and Times {t_ℓ}
Options
- F1 Fixed times (e.g., t_ℓ = 0.2·(ℓ+1))
  - Pros: deterministic and simple; mirrors Tex heuristic; stable gradients.
- F2 Learnable times (t_ℓ as parameters)
  - Pros: adaptivity; potentially better task fit.
  - Cons: additional nonlinearity; may require constraints; parameter‑shift sensitivity.
Feature design (classical pre‑Merlin)
- Spectral moments μ_k = mean(λ^k), k=1..K
- For each t ∈ times: Re tr(U)/N, Im tr(U)/N, mean(|U_ii|^2)
When to choose
- Start with F1 (fixed); only promote to F2 after ablations show consistent lifts.

## Axis L — Number of Layers (L)
Meaning
- In our BQGNN, L controls the number of CTQW time slices used to build the classical feature vector before Merlin. With default features, input_size = spectral_moments + 3·L.

Options
- L1 Low depth (L = 1)
  - Pros: smallest feature dimension; fastest; least risk of overfitting.
  - Cons: captures a single time scale; can miss interference patterns that emerge at later times.
- L2 Moderate depth (L = 2–4)
  - Pros: covers multiple time scales; balances expressivity and cost; strong default per Tex intent.
  - Cons: increased input dim → slightly larger Merlin/MLP parameters.
- L3 High depth (L ≥ 5)
  - Pros: richer temporal basis; can capture long‑range spectral structure.
  - Cons: diminishing returns; higher compute and overfitting risk on small datasets.

Interactions and Rationale
- More L expands the classical feature basis the Merlin layer can mix; this often helps until redundancy saturates.
- L couples to times {t_ℓ}. With fixed spacing (e.g., 0.2·(ℓ+1)), increasing L extends the time horizon and frequency coverage.
- The Merlin ansatz input_size scales with L; if memory is tight, prefer increasing M or N_modes slightly before pushing L too high.

Practical Heuristics
- Start with L=2–3 on TU datasets; push to L=4–5 only if you observe steady gains.
- If you enable sampling (shots>0), prefer smaller L to reduce gradient variance.
- When node attributes dominate labels, consider smaller L and invest capacity in a learned pre‑projection (Axis L3 in “Classical Features”).

## Axis G — Output Mapping (quantum → classical)
Options
- G1 Linear mapping (OutputMappingStrategy.LINEAR)
  - Pros: simplest; good with rich classical head; fewer hyperparams.
  - Cons: may leave fine‑grained distribution structure untapped.
- G2 Grouping mappers (LexGroupingMapper, ModGroupingMapper)
  - Pros: compresses large outcome spaces to fixed size; can enforce invariances (e.g., modulo).
  - Cons: choice of grouping affects inductive bias; extra tuning.
Implementation
- Merlin: set `output_mapping_strategy` in `AnsatzFactory.create`, or attach mappers explicitly.
When to choose
- G1 with strong head; G2 if you need structured dimensionality control or invariances in readout.

## Axis H — Deterministic vs Sampling
Options
- H1 Deterministic (shots=0)
  - Pros: exact probabilities; fully reproducible; ideal for CI and ablations.
  - Cons: may not reflect hardware sampling noise.
- H2 Sampling (shots>0, multinomial)
  - Pros: hardware realism; regularization via noise.
  - Cons: stochastic gradients; careful seeding required; more calls for variance reduction.
Implementation
- Merlin: `shots` and `sampling_method='multinomial'` in `QuantumLayer`; set seeds and `torch.use_deterministic_algorithms(True)` for controlled testing.
When to choose
- Use H1 for experiments/design search; use H2 for robustness studies or when calibrating to hardware.

## Axis I — Precision and Device
Options
- I1 float32 (with complex64 under the hood)
  - Pros: fast, memory‑efficient; good default.
  - Cons: gradient checking and second‑order effects less precise.
- I2 float64 (with complex128 under the hood)
  - Pros: better numeric stability and grad checks.
  - Cons: slower; higher memory.
Implementation
- Merlin: `dtype` (float32/float64 equivalents) in `AnsatzFactory/QuantumLayer`.
When to choose
- float32 for training; float64 for sensitivity analyses or delicate gradient tests.

## Axis J — Hybrid Placement (where to inject quantum block)
Options
- J1 Post‑structural features (current default): CTQW features → Merlin layer → head
  - Pros: decouples graph topology extraction from photonic mixing; lower dimension input to Merlin; fast.
  - Cons: quantum block sees only summary statistics.
- J2 Node‑wise or patch‑wise quantum blocks
  - Pros: richer interactions; closer to graph‑as‑quantum computation.
  - Cons: heavier; needs per‑node encoding + pooling; significant complexity.
When to choose
- J1 for practicality and ablation clarity; consider J2 only for small graphs and targeted studies.

## Axis K — Training Strategy
Options
- K1 Autodiff end‑to‑end (Merlin AutoDiffProcess)
  - Pros: simple integration; leverages PyTorch autograd.
  - Cons: sensitive to sampling/noise settings.
- K2 Parameter‑shift explicit gradients
  - Pros: aligns with quantum gradient rules from Tex.
  - Cons: slower unless specialized; often unnecessary if Merlin’s autodiff suffices.
Implementation
- Merlin manages autodiff internally; ensure consistent seeds, determinism flags, and dtypes.

## Axis L — Classical Features (x fed into Merlin)
Options
- L1 CTQW structural features (current)
  - Pros: permutation‑invariant by construction; informative spectral/time statistics.
- L2 Degree/centrality histograms
  - Pros: cheap; helpful prior on structure.
- L3 Learned projection from node features
  - Pros: task‑specific; allows feature learning before quantum layer.
When to choose
- L1 as default; add L3 for feature‑rich datasets or when node attributes drive labels.

## Merlin Mapping Cheat‑Sheet
- `PhotonicBackend`: `circuit_type`, `n_modes`, `n_photons`, `state_pattern`.
- `AnsatzFactory.create`: `(PhotonicBackend, input_size, output_size, output_mapping_strategy, device, dtype)`.
- `QuantumLayer`: `(input_size, ansatz, shots, sampling_method, no_bunching, index_photons, device, dtype)`.
Notes
- `input_size` = spectral_moments + 3·L with the default CTQW feature set.
- Increasing L increases `input_size` linearly; adjust `merlin_out_dim` and head width if needed.

## Recommended Default (balanced)
- H: Adjacency A; M=2; no_bunching=False; N_modes=8; PERIODIC; fixed times t_ℓ=0.2·(ℓ+1); linear output mapping; shots=0; float32; J1 pipeline; L1 features.
- Rationale: captures multi‑boson effects with tractable cost, deterministic training, and minimal hyperparameters.

## What to tune first
1) N_modes (8→16) and M (2→3) under compute budget.
2) Output mapping (linear→grouping) to control head input size/invariance.
3) State pattern (PERIODIC↔SEQUENTIAL) for topology polarity.
4) shots (0→few K) to probe robustness under sampling.

## Risks and Mitigations
- Numeric/gradient instability: prefer float32 + deterministic algorithms; escalate to float64 for checks.
- Overfitting with high M or large N_modes: add regularization or reduce head size.
- Performance regressions with sampling: stick to shots=0 for baseline and CI.

---
If you pick a combination, I can pin the exact Merlin config and wire it into the model/trainer.
