"""BatchTopK Sparse Autoencoder for interpretability analysis."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopKSAE(nn.Module):
    """BatchTopK Sparse Autoencoder with adaptive sparsity."""

    def __init__(
        self,
        d_input: int = 512,
        d_latent: int = 2048,
        sparsity_percent: float = 0.10,  # 10% sparsity
        tied_weights: bool = False,
        normalize_decoder: bool = True,
        k_fixed: Optional[int] = None,
        topk_mode: str = "batch",
    ):
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.sparsity_percent = sparsity_percent
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder
        self.k_fixed = k_fixed
        if topk_mode not in ("batch", "sample"):
            raise ValueError(f"topk_mode must be 'batch' or 'sample', got '{topk_mode}'")
        self.topk_mode = topk_mode

        self.encoder = nn.Linear(d_input, d_latent, bias=True)

        if tied_weights:
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_latent, d_input, bias=True)

        self.register_buffer("inference_threshold", torch.tensor(0.0))
        self.register_buffer("step_count", torch.tensor(0))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize encoder and decoder weights."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.constant_(self.encoder.bias, 0.01)  # Small positive bias to avoid dead neurons

        if not self.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def _get_decoder_weight(self) -> torch.Tensor:
        """Get decoder weight matrix (tied or untied)."""
        if self.tied_weights:
            return self.encoder.weight.T
        else:
            return self.decoder.weight

    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        if not self.normalize_decoder:
            return

        with torch.no_grad():
            if self.tied_weights:
                # Normalize encoder rows (which become decoder columns when transposed)
                norms = self.encoder.weight.norm(dim=1, keepdim=True)
                self.encoder.weight.div_(norms + 1e-8)
            else:
                # Normalize decoder columns
                norms = self.decoder.weight.norm(dim=0, keepdim=True)
                self.decoder.weight.div_(norms + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent pre-activations."""
        return self.encoder(x)

    def batch_topk_sparsify(
        self, z_pre: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply BatchTopK sparsification."""
        B, D = z_pre.shape

        if training:
            # BatchTopK: select top k across entire batch
            z_flat = z_pre.reshape(-1)  # [B * d_latent]
            if self.k_fixed is not None:
                k_total = self.k_fixed * B
            else:
                k_total = int(self.sparsity_percent * z_flat.numel())
            k_total = max(1, min(k_total, z_flat.numel()))

            # Get top k values and their indices
            top_values, top_indices = torch.topk(z_flat, k=k_total, largest=True)

            # Create sparse tensor
            z_sparse = torch.zeros_like(z_flat)
            z_sparse[top_indices] = top_values
            z = z_sparse.reshape(B, D)

            # Track minimum positive activation for inference threshold
            min_positive = top_values.min().item() if len(top_values) > 0 else 0.0

            # Update inference threshold (exponential moving average)
            alpha = 0.01
            self.inference_threshold.mul_(1 - alpha).add_(min_positive * alpha)
            self.step_count.add_(1)

            # Compute statistics
            active_per_sample = (z > 0).sum(dim=1).float()
            stats = {
                "L0": active_per_sample.mean().item(),
                "L0_std": active_per_sample.std().item(),
                "min_activation": top_values.min().item(),
                "max_activation": top_values.max().item(),
                "threshold": min_positive,
            }
        else:
            # Inference: use threshold-based activation
            threshold = self.inference_threshold.item()
            z = F.relu(z_pre - threshold)

            active_per_sample = (z > 0).sum(dim=1).float()
            stats = {
                "L0": active_per_sample.mean().item(),
                "L0_std": active_per_sample.std().item(),
                "threshold": threshold,
            }

        return z, stats

    def topk_sparsify(
        self, z_pre: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply per-sample TopK sparsification."""
        B, D = z_pre.shape

        if training:
            if self.k_fixed is not None:
                k = self.k_fixed
            else:
                k = max(1, int(self.sparsity_percent * D))
            k = min(k, D)

            top_values, top_indices = torch.topk(z_pre, k=k, dim=1)  # [B, k]

            z = torch.zeros_like(z_pre)
            z.scatter_(1, top_indices, top_values)

            min_positive = top_values.min().item() if top_values.numel() > 0 else 0.0

            alpha = 0.01
            self.inference_threshold.mul_(1 - alpha).add_(min_positive * alpha)
            self.step_count.add_(1)

            active_per_sample = (z > 0).sum(dim=1).float()
            stats = {
                "L0": active_per_sample.mean().item(),
                "L0_std": active_per_sample.std().item(),
                "min_activation": top_values.min().item(),
                "max_activation": top_values.max().item(),
                "threshold": min_positive,
            }
        else:
            threshold = self.inference_threshold.item()
            z = F.relu(z_pre - threshold)

            active_per_sample = (z > 0).sum(dim=1).float()
            stats = {
                "L0": active_per_sample.mean().item(),
                "L0_std": active_per_sample.std().item(),
                "threshold": threshold,
            }

        return z, stats

    def sparsify(
        self, z_pre: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Dispatch to the appropriate sparsification method based on topk_mode."""
        if self.topk_mode == "sample":
            return self.topk_sparsify(z_pre, training=training)
        else:
            return self.batch_topk_sparsify(z_pre, training=training)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to reconstruction."""
        if self.tied_weights:
            return F.linear(z, self.encoder.weight.T, None)
        else:
            return self.decoder(z)

    def forward(
        self, x: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through SAE."""
        # Encode
        z_pre = self.encode(x)

        # Sparsify
        z, stats = self.sparsify(z_pre, training=self.training)

        # Decode
        x_recon = self.decode(z)

        if return_stats:
            stats["z"] = z.detach()
            return x_recon, stats
        else:
            return x_recon

    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparsified latent activations without reconstruction."""
        z_pre = self.encode(x)
        z, _ = self.sparsify(z_pre, training=self.training)
        return z

    def get_decoder_directions(self) -> torch.Tensor:
        """Get decoder direction matrix."""
        W = self._get_decoder_weight()  # [d_input, d_latent]
        return W.T  # [d_latent, d_input]

    def intervene(
        self,
        x: torch.Tensor,
        amplify_indices: list[int] | None = None,
        amplify_scale: float = 5.0,
        amplify_shift: float = 1.0,
        clamp_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Forward pass with SAE feature intervention."""
        # Encode and sparsify
        z_pre = self.encode(x)
        z, _ = self.sparsify(z_pre, training=False)

        # Compute reconstruction error BEFORE intervention
        x_recon = self.decode(z)
        x_err = x - x_recon  # Reconstruction error to preserve

        # Apply interventions on z
        z_modified = z.clone()

        if amplify_indices is not None and len(amplify_indices) > 0:
            idx = torch.tensor(amplify_indices, device=z.device)
            z_modified[:, idx] = amplify_scale * z_modified[:, idx] + amplify_shift

        if clamp_indices is not None and len(clamp_indices) > 0:
            idx = torch.tensor(clamp_indices, device=z.device)
            z_modified[:, idx] = 0.0

        # Decode modified latents and add back reconstruction error
        x_intervened = self.decode(z_modified) + x_err

        return x_intervened

    @torch.no_grad()
    def dead_feature_fraction(
        self, x: torch.Tensor, threshold: float = 1e-6, batch_size: int = 256
    ) -> float:
        """Compute fraction of dead features (never activated) on given data."""
        ever_active = torch.zeros(self.d_latent, dtype=torch.bool, device=x.device)
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            z = self.get_activations(batch)
            ever_active |= (z.abs() > threshold).any(dim=0)
        dead = (~ever_active).float().mean().item()
        return dead


class AuxiliaryLoss(nn.Module):
    """Auxiliary loss for reviving dead features."""

    def __init__(self, dead_threshold: int = 5):
        super().__init__()
        self.dead_threshold = dead_threshold

    def forward(
        self,
        residual: torch.Tensor,
        encoder: nn.Linear,
        feature_usage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute auxiliary loss for dead features."""
        if feature_usage is None:
            # No feature usage tracking, skip auxiliary loss
            return torch.tensor(0.0, device=residual.device)

        # Identify dead features (not used in last N batches)
        dead_mask = feature_usage < self.dead_threshold

        if not dead_mask.any():
            return torch.tensor(0.0, device=residual.device)

        # Project residual onto dead feature directions
        dead_encoder = encoder.weight[dead_mask]  # [n_dead, d_input]
        projections = torch.matmul(residual, dead_encoder.T)  # [B, n_dead]

        # Encourage dead features to activate on residual
        aux_loss = projections.pow(2).mean()

        return aux_loss

