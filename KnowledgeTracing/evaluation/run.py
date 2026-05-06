#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KT_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(KT_ROOT)
for path in (REPO_ROOT, KT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from KnowledgeTracing.Constant.Constants import SUPPORTED_DATASETS, build_config  # noqa: E402
from KnowledgeTracing.DirectedGCN.load_data import build_hypergraph_inputs, build_transition_adjacency  # noqa: E402
from KnowledgeTracing.data.dataloader import get_loaders  # noqa: E402
from KnowledgeTracing.evaluation.eval import evaluate, train_epoch  # noqa: E402
from KnowledgeTracing.model import DGEKT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train graph-enhanced knowledge tracing model.")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=5e-6)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("dgekt")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{timestamp}_train.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def save_checkpoint(model, config, metrics_dict, output_dir: str, epoch: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"best_model_{config.dataset}_epoch{epoch}_"
        f"auc{metrics_dict['auc']:.4f}_acc{metrics_dict['acc']:.4f}_{timestamp}.pt"
    )
    path = os.path.join(output_dir, filename)
    payload = {
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "metrics": metrics_dict,
        "epoch": epoch,
    }
    torch.save(payload, path)
    return path


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    set_seed(args.seed)

    config = build_config(
        args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        kd_weight=args.kd_weight,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    logger = create_logger(os.path.join(KT_ROOT, "log"))
    logger.info("Device: %s", device)
    logger.info("Config: %s", json.dumps(config.__dict__, ensure_ascii=False))

    train_loader, test_loader = get_loaders(config)
    logger.info("Train batches: %d | Test batches: %d", len(train_loader), len(test_loader))

    graph_inputs = build_transition_adjacency(config)
    graph_inputs.update(build_hypergraph_inputs(config))
    graph_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in graph_inputs.items()
    }

    model = DGEKT(config, graph_inputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model_dir = os.path.join(KT_ROOT, "model", config.dataset)
    best_auc = -1.0
    best_epoch = 0
    best_path = ""
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        test_metrics = evaluate(model, test_loader, device, config)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | test_loss=%.6f | test_auc=%.6f | test_acc=%.6f",
            epoch,
            config.epochs,
            train_loss,
            test_metrics["loss"],
            test_metrics["auc"],
            test_metrics["acc"],
        )
        print(
            f"Epoch {epoch}/{config.epochs} | train_loss={train_loss:.6f} | "
            f"test_auc={test_metrics['auc']:.6f} | test_acc={test_metrics['acc']:.6f}"
        )

        if test_metrics["auc"] > best_auc:
            best_auc = test_metrics["auc"]
            best_epoch = epoch
            stale_epochs = 0
            best_path = save_checkpoint(model, config, test_metrics, model_dir, epoch)
            logger.info("Best model updated: %s", best_path)
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            logger.info("Early stopping at epoch %d after %d stale epochs.", epoch, stale_epochs)
            break

    logger.info("Training complete | best_epoch=%d | best_auc=%.6f | checkpoint=%s", best_epoch, best_auc, best_path)
    print(f"Best epoch: {best_epoch} | best_auc={best_auc:.6f} | checkpoint={best_path}")


if __name__ == "__main__":
    main()
