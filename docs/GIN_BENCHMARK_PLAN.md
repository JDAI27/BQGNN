# GIN Benchmark Plan (Classical Baseline)

## Overview
- Goal: Train and evaluate a GIN baseline (per benchmarking-gnns) on MUTAG, PROTEINS, NCI1 with 10-fold CV. Save logs, checkpoints, and result summaries.
- Repo used: `benchmarking-gnns` (DGL-based). We will extend TU dataset support and add configs for the three datasets.

## Environment Setup
- Conda (recommended):
  - `conda env create -f benchmarking-gnns/environment_cpu.yml -n gnn-bench`
  - `conda activate gnn-bench`
- Or minimal pip (CPU example):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install torch==1.12.1 dgl==0.9.1.post1 tensorboardX==2.6 numpy scikit-learn tqdm networkx`
- Verify: `python -c "import dgl, torch; print(dgl.__version__, torch.__version__)"`

## Dataset Support & Preparation
- Extend TU dataset list to include MUTAG and NCI1:
  - File: `benchmarking-gnns/data/data.py`
  - Change `TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full']` to include `'MUTAG', 'NCI1'`.
- Notes:
  - DGL names: `MUTAG`, `NCI1`, `PROTEINS_full` (used for PROTEINS in this plan).
  - Splits are auto-generated and cached to `benchmarking-gnns/data/TUs/*.index` on first run.

## Configs (copy then adjust)
- Base files (examples to copy):
  - `benchmarking-gnns/configs/TUs_graph_classification_GIN_PROTEINS_full_100k.json`
  - `benchmarking-gnns/configs/TUs_graph_classification_GIN_ENZYMES_100k.json`
- Create new configs:
  - `benchmarking-gnns/configs/TUs_graph_classification_GIN_MUTAG_100k.json`
  - `benchmarking-gnns/configs/TUs_graph_classification_GIN_NCI1_100k.json`
- Suggested initial values (tune later):
  - Common: `L=4`, `n_mlp_GIN=2`, `learn_eps_GIN=true`, `neighbor_aggr_GIN='sum'`, `readout='sum'`, `batch_norm=true`, `dropout=0.0`.
  - MUTAG: `hidden_dim=64`, `batch_size=32`, `init_lr=1e-3`.
  - PROTEINS_full: use provided config (hidden_dim=110, init_lr=1e-4).
  - NCI1: `hidden_dim=128`, `batch_size=64`, `init_lr=1e-3`.
  - Train params: `epochs=1000`, `lr_reduce_factor=0.5`, `lr_schedule_patience=25`, `min_lr=1e-6`, `weight_decay=0.0`, `max_time=12` (hours budget for 10 runs).

## Run Commands (from repo root)
- Activate env, then run from `benchmarking-gnns/` (relative paths expected):
  - `cd benchmarking-gnns`
  - MUTAG: `python main_TUs_graph_classification.py --config configs/TUs_graph_classification_GIN_MUTAG_100k.json`
  - PROTEINS_full: `python main_TUs_graph_classification.py --config configs/TUs_graph_classification_GIN_PROTEINS_full_100k.json`
  - NCI1: `python main_TUs_graph_classification.py --config configs/TUs_graph_classification_GIN_NCI1_100k.json`
- Override examples (optional): `--epochs 500 --batch_size 32 --init_lr 0.001`.

## Outputs & Verification
- Logs: `benchmarking-gnns/out/TUs_graph_classification/logs/GIN_<DATASET>_GPU*/RUN_*/`
- Checkpoints: `.../checkpoints/`
- Results summary: `.../results/result_GIN_<DATASET>_GPU*.txt` includes mean ± std test accuracy across 10 folds and timing.
- Quick sanity checks:
  - Training loss decreases; validation accuracy stabilizes.
  - Final print shows “FINAL RESULTS … TEST ACCURACY averaged: … with s.d. …”.

## Tuning & Reproducibility
- Tune `hidden_dim ∈ {64,110,128}`, `init_lr ∈ {1e-4,1e-3,3e-4}`, `dropout ∈ {0,0.5}`.
- Fix seeds via config (`params.seed=41`). Reruns reuse saved splits in `data/TUs/`.

## Next Steps
- Export consolidated CSV of results and create plots.
- Lock best hyperparameters per dataset before comparing against BQGNN.
