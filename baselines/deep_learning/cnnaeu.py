"""CNNAEU: CNN-based Autoencoder for spectral Unmixing."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class CNNEncoder(nn.Module):
    """1D CNN encoder for spectral feature extraction."""

    def __init__(self, input_dim: int = 1250, hidden_dims: list = [128, 64, 32]):
        super().__init__()

        layers = []
        in_channels = 1

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
                nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),  # More stable than BatchNorm
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Calculate output size after pooling
        # Account for integer division truncation
        self.output_size = input_dim // (2 ** len(hidden_dims))
        self.output_channels = hidden_dims[-1]
        self.input_dim = input_dim  # Store for decoder

    def forward(self, x):
        """Args:
        x: [batch, seq_len] mixture spectra
        """
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        return self.encoder(x)


class CNNDecoder(nn.Module):
    """1D CNN decoder for component spectrum generation."""

    def __init__(self, n_components: int, input_channels: int, input_size: int,
                 output_dim: int = 1250, hidden_dims: list = [32, 64, 128]):
        super().__init__()

        self.n_components = n_components
        self.input_channels = input_channels
        self.input_size = input_size
        self.output_dim = output_dim

        layers = []
        in_channels = input_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.ConvTranspose1d(in_channels, hidden_dim, kernel_size=4,
                                  stride=2, padding=1),
                nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),  # More stable than BatchNorm
                nn.ReLU(),
            ])
            in_channels = hidden_dim

        # Final layer to output component spectra
        final_conv = nn.Conv1d(hidden_dims[-1], n_components, kernel_size=5, padding=2)
        # Initialize bias to small positive value to prevent all-zero outputs after ReLU
        # If decoder outputs negative values, ReLU clips them to zero, causing component collapse
        nn.init.constant_(final_conv.bias, 0.01)
        layers.append(final_conv)

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """Args:
        x: [batch, channels, seq_len] encoded features
        """
        x = self.decoder(x)  # [batch, n_components, ~output_dim]

        # Ensure non-negative spectra using Softplus instead of ReLU
        # Softplus has smooth gradients everywhere, preventing dying gradient problem
        # ReLU clips negative values to zero, killing gradients and causing component collapse
        x = nn.functional.softplus(x, beta=5)  # beta=5 makes it sharper, closer to ReLU but differentiable

        # Handle dimension mismatch: if output is smaller than target, pad; if larger, crop
        if x.size(-1) < self.output_dim:
            # Pad to match target dimension
            pad_size = self.output_dim - x.size(-1)
            x = nn.functional.pad(x, (0, pad_size), mode='replicate')
        elif x.size(-1) > self.output_dim:
            # Crop to match target dimension
            x = x[..., :self.output_dim]

        return x


class CNNAEU(nn.Module):
    """CNN-based Autoencoder for spectral Unmixing (CNNAEU)."""

    def __init__(self, n_components: int = 2, input_dim: int = 1250,
                 encoder_dims: list = [128, 64, 32],
                 decoder_dims: list = [32, 64, 128]):
        super().__init__()

        self.n_components = n_components
        self.input_dim = input_dim

        # Encoder
        self.encoder = CNNEncoder(input_dim, encoder_dims)

        # Abundance predictor (from encoded features)
        self.abundance_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder.output_channels * self.encoder.output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_components),
            nn.Softmax(dim=-1)  # Ensure sum-to-one constraint
        )

        # Decoder (generate component spectra)
        self.decoder = CNNDecoder(
            n_components=n_components,
            input_channels=self.encoder.output_channels,
            input_size=self.encoder.output_size,
            output_dim=input_dim,
            hidden_dims=decoder_dims
        )

    def forward(self, mixture):
        """Args:
        mixture: [batch, input_dim] mixture spectra
        """
        # Encode
        encoded = self.encoder(mixture)  # [batch, channels, reduced_seq]

        # Predict abundances
        abundances = self.abundance_predictor(encoded)  # [batch, n_components]

        # Generate component spectra
        components = self.decoder(encoded)  # [batch, n_components, input_dim]

        # Reconstruct mixture as weighted sum
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
