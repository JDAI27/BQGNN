from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from .data import load_tu_dataset, GraphData
from .models import GIN, BQGNN


@dataclass
class TrainConfig:
    dataset: str
    arch: str = "gin"  # gin | bqgnn
    seed: int = 41
    epochs: int = 300
    batch_size: int = 32
    init_lr: float = 1e-3
    weight_decay: float = 0.0
    lr_reduce_factor: float = 0.5
    lr_schedule_patience: int = 25
    min_lr: float = 1e-6
    model_hidden: int = 64
    model_layers: int = 4
    model_dropout: float = 0.0
    model_learn_eps: bool = True
    readout: str = "sum"
    # BQGNN specific
    bq_spectral_moments: int = 4
    bq_times: List[float] | None = None  # if None, auto linspace
    bq_head_hidden: int = 64
    bq_hamiltonian: str = "A"
    bq_bosons: int = 1
    default_feat: str = "degree"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _collate(graphs: List[GraphData]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(g.x, g.edge_index) for g in graphs]


def _batch_iter(graphs: List[GraphData], labels: List[int], batch_size: int, shuffle: bool = True):
    idx = np.arange(len(graphs))
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i : i + batch_size]
        yield [graphs[j] for j in sel], torch.tensor([labels[j] for j in sel], dtype=torch.long)


def train_one_fold(model: nn.Module, train_graphs: List[GraphData], train_labels: List[int],
                   val_graphs: List[GraphData], val_labels: List[int], cfg: TrainConfig) -> Tuple[float, float]:
    device = torch.device(cfg.device)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.init_lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.lr_reduce_factor, patience=cfg.lr_schedule_patience
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_graphs, y in _batch_iter(train_graphs, train_labels, cfg.batch_size, shuffle=True):
            batch = _collate(batch_graphs)
            x_ei = [(x.to(device), ei.to(device)) for (x, ei) in batch]
            logits = model.forward_batch(x_ei)
            loss = criterion(logits, y.to(device))

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach()) * y.size(0)
            preds = logits.argmax(dim=1).detach().cpu()
            correct += int((preds == y).sum())
            total += y.size(0)

        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            v_correct = 0
            v_total = 0
            v_loss_sum = 0.0
            for batch_graphs, y in _batch_iter(val_graphs, val_labels, cfg.batch_size, shuffle=False):
                batch = _collate(batch_graphs)
                x_ei = [(x.to(device), ei.to(device)) for (x, ei) in batch]
                logits = model.forward_batch(x_ei)
                loss = criterion(logits, y.to(device))
                v_loss_sum += float(loss) * y.size(0)
                preds = logits.argmax(dim=1).cpu()
                v_correct += int((preds == y).sum())
                v_total += y.size(0)
            val_loss = v_loss_sum / max(v_total, 1)
            val_acc = v_correct / max(v_total, 1)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc

    return best_train_acc, best_val_acc


def run_kfold_experiment(cfg: TrainConfig) -> Dict[str, float]:
    graphs, labels = load_tu_dataset(cfg.dataset, default_feat=cfg.default_feat)
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.seed)
    fold_train_acc: List[float] = []
    fold_val_acc: List[float] = []

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(y), y), start=1):
        tr_graphs = [graphs[i] for i in train_idx]
        tr_labels = [labels[i] for i in train_idx]
        va_graphs = [graphs[i] for i in val_idx]
        va_labels = [labels[i] for i in val_idx]

        in_dim = tr_graphs[0].x.size(1)
        n_classes = int(y.max() + 1)
        if cfg.arch.lower() == "gin":
            model = GIN(
                in_dim=in_dim,
                hidden_dim=cfg.model_hidden,
                num_layers=cfg.model_layers,
                num_classes=n_classes,
                dropout=cfg.model_dropout,
                learn_eps=cfg.model_learn_eps,
                readout=cfg.readout,
            )
        elif cfg.arch.lower() == "bqgnn":
            model = BQGNN(
                num_layers=cfg.model_layers,
                times=cfg.bq_times,
                spectral_moments=cfg.bq_spectral_moments,
                head_hidden=cfg.bq_head_hidden,
                num_classes=n_classes,
                hamiltonian=cfg.bq_hamiltonian,
                bosons=cfg.bq_bosons,
            )
        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

        tr_acc, va_acc = train_one_fold(model, tr_graphs, tr_labels, va_graphs, va_labels, cfg)
        fold_train_acc.append(tr_acc)
        fold_val_acc.append(va_acc)

    result = {
        "dataset": cfg.dataset,
        "mean_train_acc": float(np.mean(fold_train_acc)),
        "std_train_acc": float(np.std(fold_train_acc)),
        "mean_val_acc": float(np.mean(fold_val_acc)),
        "std_val_acc": float(np.std(fold_val_acc)),
    }
    return result


def run_and_save(cfg_dict: Dict, out_dir: str = "runs") -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = TrainConfig(
        dataset=cfg_dict.get("dataset", "MUTAG"),
        arch=str(cfg_dict.get("arch", "gin")),
        seed=int(cfg_dict.get("seed", 41)),
        epochs=int(cfg_dict.get("epochs", 300)),
        batch_size=int(cfg_dict.get("batch_size", 32)),
        init_lr=float(cfg_dict.get("init_lr", 1e-3)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.0)),
        lr_reduce_factor=float(cfg_dict.get("lr_reduce_factor", 0.5)),
        lr_schedule_patience=int(cfg_dict.get("lr_schedule_patience", 25)),
        min_lr=float(cfg_dict.get("min_lr", 1e-6)),
        model_hidden=int(cfg_dict.get("model", {}).get("hidden_dim", 64)),
        model_layers=int(cfg_dict.get("model", {}).get("num_layers", 4)),
        model_dropout=float(cfg_dict.get("model", {}).get("dropout", 0.0)),
        model_learn_eps=bool(cfg_dict.get("model", {}).get("learn_eps", True)),
        readout=str(cfg_dict.get("model", {}).get("readout", "sum")),
        default_feat=str(cfg_dict.get("features", "degree")),
        bq_spectral_moments=int(cfg_dict.get("model", {}).get("spectral_moments", 4)),
        bq_times=cfg_dict.get("model", {}).get("times", None),
        bq_head_hidden=int(cfg_dict.get("model", {}).get("head_hidden", 64)),
        bq_hamiltonian=str(cfg_dict.get("model", {}).get("hamiltonian", "A")),
        bq_bosons=int(cfg_dict.get("model", {}).get("bosons", 1)),
    )

    result = run_kfold_experiment(cfg)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gin_{cfg.dataset}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"config": cfg_dict, "result": result}, f, indent=2)
    return result
