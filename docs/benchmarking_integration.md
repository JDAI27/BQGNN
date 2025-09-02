# Benchmarking-GNNs Integration Summary

This repository now includes an adapter to run our `BQGNN` inside the classic Benchmarking-GNNs training harness (Dwivedi et al.).

## What we copied
- Structure: use their `main_*` scripts, JSON configs under `benchmarking-gnns/configs`, and model registry in `benchmarking-gnns/nets/TUs_graph_classification`.
- Interface: implemented a `BQGNNNet` wrapper with `forward(g, h, e)` and `loss(...)` compatible with their training functions in `benchmarking-gnns/train/train_TUs_graph_classification.py`.

## Files added/changed
- `benchmarking-gnns/nets/TUs_graph_classification/bqgnn_net.py`: wraps `src/bqgnn/models/BQGNN` for DGL batched graphs.
- `benchmarking-gnns/nets/TUs_graph_classification/load_net.py`: registers model name `"BQGNN"`.
- `benchmarking-gnns/configs/TUs_graph_classification_BQGNN_MUTAG_100k.json`: example config for MUTAG.

## How the adapter works
- Unbatches the input `DGLGraph` to per-graph objects and converts each to `(x, edge_index)` expected by our `BQGNN`.
  - `x`: placeholder ones (our CTQW features are structural).
  - `edge_index`: built from `g.edges(order='eid')` as a `[2, E]` long tensor.
- Calls `BQGNN.forward_batch(...)` to get `[B, n_classes]` logits.
- Uses standard cross-entropy for `loss(...)`.

## Config knobs (net_params)
- Common: `L` (num layers), `hidden_dim` (mapped to head hidden), `n_classes` (set by the harness).
- BQGNN extras: `bq_spectral_moments`, `bq_hamiltonian`, `bq_bosons`.
- Merlin (optional): `bq_merlin_enable`, `bq_merlin_n_modes`, `bq_merlin_out_dim`, `bq_merlin_shots`,
  `bq_merlin_no_bunching`, `bq_merlin_circuit_type`, `bq_merlin_state_pattern`, `bq_merlin_dtype`.

## Run example
Environment prerequisites: the original benchmark targets PyTorch 1.6 + DGL 0.6 (see `benchmarking-gnns/environment_cpu.yml`). If you use newer Torch, install a matching DGL build. For reproducible runs, prefer creating a dedicated env from their YAML.

- Command (once env + datasets are ready):
  - `python benchmarking-gnns/main_TUs_graph_classification.py --config benchmarking-gnns/configs/TUs_graph_classification_BQGNN_MUTAG_100k.json`

Notes
- Dataset download and splits are handled by `benchmarking-gnns/data/TUs.py` (uses DGL’s `LegacyTUDataset`), and writes indices under `benchmarking-gnns/data/TUs/`.
- If enabling Merlin, ensure `merlinquantum` is installed and your Python is ≥3.10.

## Caveats
- DGL builds are Torch-version specific. If you see missing `graphbolt`/C++ library errors, align your Torch and DGL versions, or use the benchmark’s conda env (PyTorch 1.6, DGL 0.6.1).
- The adapter loops over unbatched graphs; for large batches this is acceptable given TU sizes, but can be optimized later.

## Next steps (optional)
- Add more dataset configs (NCI1/PROTEINS/ENZYMES) under `benchmarking-gnns/configs/`.
- Expose Merlin settings via the JSON to test hybrid runs.
- Add a thin script to validate environment compatibility (Torch/DGL version probe) before launching runs.
