"""DeepTrans-HSU: Deep Transformer for Hyperspectral Unmixing."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for spectral sequences."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Args:
        x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class SpectralEmbedding(nn.Module):
    """Embed 1D spectrum into higher-dimensional space."""

    def __init__(self, input_dim: int = 1250, d_model: int = 128, patch_size: int = 5):
        super().__init__()

        self.patch_size = patch_size
        self.input_dim = input_dim
        # Ensure input_dim is divisible by patch_size, pad if needed
        if input_dim % patch_size != 0:
            # Pad to next multiple of patch_size
            self.padded_dim = ((input_dim // patch_size) + 1) * patch_size
        else:
            self.padded_dim = input_dim
        self.n_patches = self.padded_dim // patch_size
        self.d_model = d_model

        # Linear projection for each patch
        self.projection = nn.Linear(patch_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_patches)

    def forward(self, x):
        """Args:
        x: [batch, input_dim] spectrum
        """
        batch_size = x.size(0)
        
        # Pad if necessary to make divisible by patch_size
        if x.size(-1) < self.padded_dim:
            pad_size = self.padded_dim - x.size(-1)
            x = nn.functional.pad(x, (0, pad_size), mode='constant', value=0.0)
        elif x.size(-1) > self.padded_dim:
            # Crop if somehow larger
            x = x[..., :self.padded_dim]

        # Reshape into patches [batch, n_patches, patch_size]
        x = x.view(batch_size, self.n_patches, self.patch_size)

        # Project to d_model
        x = self.projection(x)  # [batch, n_patches, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Args:
        x: [batch, seq_len, d_model]
        """
        # Self-attention block
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # Feedforward block
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x


class DeepTransHSU(nn.Module):
    """Deep Transformer for Hyperspectral Unmixing."""

    def __init__(self, n_components: int = 2, input_dim: int = 1250,
                 d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1, patch_size: int = 5):
        super().__init__()

        self.n_components = n_components
        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_size = patch_size

        # Spectral embedding
        self.embedding = SpectralEmbedding(input_dim, d_model, patch_size)

        # Transformer encoder
        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layers.append(
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            )
        self.encoder = nn.Sequential(*encoder_layers)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Abundance predictor
        self.abundance_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_components),
            nn.Softmax(dim=-1)  # Sum-to-one constraint
        )

        # Component decoder (from transformer features to component spectra)
        # Use n_patches from embedding (which handles padding)
        n_patches = self.embedding.n_patches
        self.component_decoder = nn.Sequential(
            nn.Linear(d_model * n_patches, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_components * input_dim),
        )

    def forward(self, mixture):
        """Args:
        mixture: [batch, input_dim] mixture spectra
        """
        batch_size = mixture.size(0)

        # Embed spectrum
        x = self.embedding(mixture)  # [batch, n_patches, d_model]

        # Transformer encoding
        encoded = self.encoder(x)  # [batch, n_patches, d_model]

        # Abundance prediction (use global pooled features)
        pooled = encoded.mean(dim=1)  # [batch, d_model] - global average pooling
        abundances = self.abundance_head(pooled)  # [batch, n_components]

        # Component generation (use full sequence features)
        flattened = encoded.reshape(batch_size, -1)  # [batch, n_patches * d_model]
        components_flat = self.component_decoder(flattened)  # [batch, n_comp * input_dim]

        # Reshape to component spectra
        components = components_flat.view(batch_size, self.n_components, self.input_dim)
        
        # Crop to original input_dim if padding was used
        if components.size(-1) > self.input_dim:
            components = components[..., :self.input_dim]

        # Apply ReLU to ensure non-negativity
        components = torch.relu(components)

        # Don't normalize components - this destroys intensity information
        # The model should learn the correct intensity scale

        # Reconstruct mixture
        reconstruction = torch.einsum('bn,bnd->bd', abundances, components)

        return abundances, components, reconstruction

    def unmix(self, mixture: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unmix a single spectrum."""
        self.eval()
        with torch.no_grad():
            mixture_tensor = torch.FloatTensor(mixture).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                mixture_tensor = mixture_tensor.cuda()

            abundances, components, reconstruction = self.forward(mixture_tensor)

            return (
                components.squeeze(0).cpu().numpy(),
                abundances.squeeze(0).cpu().numpy(),
                reconstruction.squeeze(0).cpu().numpy()
            )

    def unmix_batch(self, mixtures: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unmix a batch of spectra."""
        self.eval()
        with torch.no_grad():
            mixture_tensor = torch.FloatTensor(mixtures)
            if next(self.parameters()).is_cuda:
                mixture_tensor = mixture_tensor.cuda()

            abundances, components, reconstructions = self.forward(mixture_tensor)

            return (
                components.cpu().numpy(),
                abundances.cpu().numpy(),
                reconstructions.cpu().numpy()
            )
