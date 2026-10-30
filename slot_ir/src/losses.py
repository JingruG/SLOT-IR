from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_component_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Args:
    pred: [B, K_pred, L] predicted components
    """
    bsz, k_pred, length = pred.shape
    k_target = target.shape[1]
    losses = []
    for b in range(bsz):
        # Compute cost matrix (keep gradients for matching, but use detached for assignment)
        cost = torch.cdist(pred[b], target[b], p=2)  # [K_pred, K_target]
        # Hungarian algorithm needs numpy, so detach only for assignment
        cost_np = cost.detach().cpu().numpy()
        row, col = linear_sum_assignment(cost_np)  # Returns K_target assignments
        # Use assignment indices on original tensors (with gradients)
        matched_pred = pred[b, row]  # [K_target, L] - keeps gradients
        matched_target = target[b, col]  # [K_target, L]
        loss = F.mse_loss(matched_pred, matched_target)
        losses.append(loss)
    return torch.stack(losses).mean()


def match_slots_l2(pred: torch.Tensor, target: torch.Tensor):
    """For each sample, the true component matched to each slot."""
    import numpy as np
    cols = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.int64)
    for b in range(pred.shape[0]):
        cost = torch.cdist(pred[b].float(), target[b].float(), p=2).detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        cols[b, row] = col
    return cols


def spectral_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss for spectral shape matching."""
    # Normalize to unit vectors
    pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
    # Cosine similarity (1 - similarity) as loss
    cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()
    return 1.0 - cosine_sim


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    stage: int,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    if stage in (0, 1):
        # Translation MSE loss
        losses["translation"] = F.mse_loss(outputs["liq_pred"], batch["liq"])
        # Spectral similarity loss for better shape matching
        losses["similarity"] = spectral_similarity_loss(outputs["liq_pred"], batch["liq"])
        total = sum(weights.get(k, 1.0) * v for k, v in losses.items())
        losses["total"] = total
        return losses

    losses["component"] = hungarian_component_loss(outputs["components"], batch["components"])
    total = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    losses["total"] = total
    return losses
