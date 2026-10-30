from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTokenizer(nn.Module):
    """Multi-scale learnable window tokenizer. Each scale uses a convolutional
    projection followed by a learnable per-position bias (tanh-bounded)
    """

    def __init__(
        self,
        spectrum_dim: int,
        window_sizes: List[int],
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.window_sizes = window_sizes
        self.d_model = d_model
        self.filters = nn.ModuleDict()
        for size in window_sizes:
            conv = nn.Conv1d(1, d_model, kernel_size=size, padding="same")
            self.filters[str(size)] = conv
        self.boundary_offsets = nn.Parameter(torch.zeros(len(window_sizes), spectrum_dim))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
        spectrum: [B, L]
        """
        bsz, length = spectrum.shape
        spectrum = spectrum.unsqueeze(1)  # [B, 1, L]
        tokens: List[torch.Tensor] = []
        offsets: List[torch.Tensor] = []
        for scale_idx, size in enumerate(self.window_sizes):
            conv = self.filters[str(size)]
            filtered = conv(spectrum)
            filtered = filtered[..., :length]  # enforce original length
            filtered = filtered.transpose(1, 2)  # [B, L, d_model]
            offset = torch.tanh(self.boundary_offsets[scale_idx, : length])
            offsets.append(offset)
            offset = offset.unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
            shifted = filtered + offset
            pooled = F.avg_pool1d(
                shifted.transpose(1, 2),
                kernel_size=size,
                stride=size,
                ceil_mode=True,
            )
            pooled = pooled.transpose(1, 2)
            tokens.append(pooled)

        encoded = torch.cat(tokens, dim=1)
        encoded = self.layer_norm(self.dropout(encoded))
        return {
            "tokens": encoded,
            "offsets": torch.stack(offsets, dim=0),
        }


def merge_token_dicts(token_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    tokens = torch.cat([td["tokens"] for td in token_dicts], dim=1)
    offsets = torch.cat([td["offsets"] for td in token_dicts], dim=0)
    return {"tokens": tokens, "offsets": offsets}

