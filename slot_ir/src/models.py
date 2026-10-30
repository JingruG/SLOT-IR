from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .tokenization import AdaptiveTokenizer

from .sae import BatchTopKSAE


class SpectralEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_encoder_layers)

        self.positional = nn.Parameter(torch.randn(1, 2048, cfg.d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        pos = self.positional[:, :length, :]
        encoded = self.encoder(tokens + pos)
        return encoded


class ConditionalDecoderLayer(nn.Module):
    """Decoder layer with cross-attention to gas memory."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.cross_attn_gas = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, tgt: torch.Tensor, memory_gas: torch.Tensor,
        return_attentions: bool = False
    ):
        # Self-attention
        tgt = self.norm1(tgt + self.dropout(self.self_attn(tgt, tgt, tgt)[0]))

        # Cross-attention to gas memory
        if return_attentions:
            gas_attn_out, gas_attn_weights = self.cross_attn_gas(tgt, memory_gas, memory_gas, need_weights=True)
            tgt = self.norm2(tgt + self.dropout(gas_attn_out))
        else:
            gas_attn_out, _ = self.cross_attn_gas(tgt, memory_gas, memory_gas, need_weights=False)
            gas_attn_weights = None
            tgt = self.norm2(tgt + self.dropout(gas_attn_out))

        # FFN
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))

        if return_attentions:
            return tgt, {"gas": gas_attn_weights}
        return tgt


class ConditionalDecoder(nn.Module):
    """Decoder with cross-attention to gas memory for gas-to-liquid translation."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ConditionalDecoderLayer(cfg) for _ in range(cfg.n_cross_layers)])
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.spectrum_dim),
            nn.ReLU(),  # Ensure non-negative spectra
        )

    def forward(self, memory_gas: torch.Tensor, return_attentions: bool = False):
        # Use mean-pooled gas memory as initial target
        tgt = memory_gas.mean(dim=1, keepdim=True)  # [B, 1, d_model]

        all_attentions = []
        for layer in self.layers:
            if return_attentions:
                tgt, attn = layer(tgt, memory_gas, return_attentions=True)
                all_attentions.append(attn)
            else:
                tgt = layer(tgt, memory_gas, return_attentions=False)

        # Project to spectrum
        spectrum = self.output_proj(tgt.squeeze(1))  # [B, spectrum_dim]

        if return_attentions:
            return spectrum, all_attentions
        return spectrum


class FCRoutingHead(nn.Module):
    """FC projection routing head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.max_components = cfg.max_components
        self.d_model = cfg.d_model
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.max_components * cfg.d_model),
        )

    def forward(self, memory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pooled = memory.mean(dim=1)  # [B, d_model]
        flat = self.proj(pooled)     # [B, K * d_model]
        slots = flat.view(-1, self.max_components, self.d_model)
        return {"slots": slots, "attn": []}


class ComponentPredictionHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.spectrum_dim),
            nn.ReLU(),
        )
        self.contribution = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.spectrum_dim),
            nn.Sigmoid(),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        spectra = self.mlp(slots)
        contribution = self.contribution(slots)
        confidence = torch.sigmoid(self.confidence_head(slots))
        return {
            "components": spectra,
            "contribution": contribution,
            "confidence": confidence,
        }


class SlotUnmixer(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        use_sae_stage1: bool = False,
        use_sae_stage2: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AdaptiveTokenizer(cfg.spectrum_dim, cfg.token_window_sizes, cfg.d_model, dropout=cfg.dropout)
        self.encoder = SpectralEncoder(cfg)
        self.translation_decoder = ConditionalDecoder(cfg)
        self.routing = FCRoutingHead(cfg)
        self.prediction = ComponentPredictionHead(cfg)

        # Interpretability modules (optional)
        self.use_sae_stage1 = use_sae_stage1
        self.use_sae_stage2 = use_sae_stage2

        if self.use_sae_stage1:
            # Stage 1 SAE: 4x overcomplete (2048 for 512 input)
            self.sae_stage1 = BatchTopKSAE(
                d_input=cfg.d_model,
                d_latent=2048,
                sparsity_percent=0.10,
            )

        if self.use_sae_stage2:
            # Stage 2 SAE: 2x overcomplete (1024 for 512 input)
            self.sae_stage2 = BatchTopKSAE(
                d_input=cfg.d_model,
                d_latent=1024,
                sparsity_percent=0.10,
            )

    def forward(
        self,
        gas: Optional[torch.Tensor] = None,
        mixture: Optional[torch.Tensor] = None,
        fg: Optional[torch.Tensor] = None,
        stage: int = 1,
        return_sae_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if stage in (0, 1):
            if gas is None:
                raise ValueError("gas spectrum required for stage <= 1")
            tokens = self.tokenizer(gas)["tokens"]
            memory_gas = self.encoder(tokens)

            # Check if return_attentions is requested
            return_attentions = getattr(self, '_return_attentions', False)
            if return_attentions:
                liq_pred, attentions = self.translation_decoder(memory_gas, return_attentions=True)
                outputs["liq_pred"] = liq_pred
                outputs["attentions"] = attentions
            else:
                outputs["liq_pred"] = self.translation_decoder(memory_gas, return_attentions=False)
            return outputs

        if mixture is None:
            raise ValueError("mixture spectrum required for stage >= 2")
        tokens = self.tokenizer(mixture)["tokens"]
        memory = self.encoder(tokens)
        routing = self.routing(memory)
        slots = routing["slots"]  # [B, K, d_model]

        # Optional: SAE feature extraction for Stage 2
        sae_features_stage2 = None
        if self.use_sae_stage2:
            B, K, d = slots.shape
            slots_flat = slots.reshape(B * K, d)
            sae_features_flat = self.sae_stage2.get_activations(slots_flat)  # [B*K, 1024]
            sae_features_stage2 = sae_features_flat.reshape(B, K, -1)  # [B, K, 1024]
            outputs["sae_features_stage2"] = sae_features_stage2

        pred = self.prediction(slots)
        outputs.update(pred)
        outputs["attn"] = routing["attn"]
        outputs["token_offsets"] = self.tokenizer.boundary_offsets
        outputs["slots"] = slots  # Include slots in output for analysis
        return outputs

