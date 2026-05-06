# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from sklearn import metrics

from KnowledgeTracing.Constant.Constants import TrainConfig


def compute_loss(outputs: Dict[str, torch.Tensor], config: TrainConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = outputs["mask"]
    if mask.sum().item() == 0:
        raise ValueError("Encountered a batch with no valid prediction targets.")

    target = outputs["target"][mask]
    logit_h = outputs["logit_h"][mask]
    logit_g = outputs["logit_g"][mask]
    logit_e = outputs["logit_e"][mask]

    loss_h = F.binary_cross_entropy_with_logits(logit_h, target)
    loss_g = F.binary_cross_entropy_with_logits(logit_g, target)
    loss_e = F.binary_cross_entropy_with_logits(logit_e, target)

    prob_h = torch.sigmoid(logit_h)
    prob_g = torch.sigmoid(logit_g)
    prob_e = torch.sigmoid(logit_e)
    kd = config.kd_weight * (
        F.mse_loss(prob_h, prob_e.detach()) + F.mse_loss(prob_g, prob_e.detach())
    )
    total = loss_h + loss_g + loss_e + kd
    return total, {
        "loss_h": float(loss_h.detach().cpu()),
        "loss_g": float(loss_g.detach().cpu()),
        "loss_e": float(loss_e.detach().cpu()),
        "kd": float(kd.detach().cpu()),
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(model, loader, optimizer, device: torch.device, config: TrainConfig) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss, _ = compute_loss(outputs, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        num_batches += 1
    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, config: TrainConfig) -> Dict[str, float]:
    model.eval()
    all_targets = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss, _ = compute_loss(outputs, config)
        mask = outputs["mask"]
        probs = torch.sigmoid(outputs["logit_e"][mask])
        targets = outputs["target"][mask]
        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())
        total_loss += float(loss.detach().cpu())
        num_batches += 1

    if not all_targets:
        raise ValueError("No evaluation targets were collected.")

    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_probs).numpy()
    auc = metrics.roc_auc_score(y_true, y_prob)
    acc = metrics.accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    return {
        "loss": total_loss / max(1, num_batches),
        "auc": float(auc),
        "acc": float(acc),
    }
