"""CyCU-Net: Cycle-Consistent Unmixing Network."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class CycleEncoder(nn.Module):
    """Encoder for cycle-consistent unmixing (shared between forward and reverse)."""

    def __init__(self, input_dim: int = 1250, hidden_dims: list = [256, 128, 64]):
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
        self.output_size = input_dim // (2 ** len(hidden_dims))
        self.output_channels = hidden_dims[-1]
        self.input_dim = input_dim

    def forward(self, x):
        """Args:
        x: [batch, seq_len] or [batch, n_components, seq_len] spectra
        """
        if x.dim() == 2:
            # Single spectrum: [batch, seq_len]
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
            return self.encoder(x)  # [batch, channels, reduced_seq]
        else:
            # Multiple spectra: [batch, n_components, seq_len]
            batch_size, n_components, seq_len = x.shape
            x = x.view(batch_size * n_components, seq_len)
            x = x.unsqueeze(1)  # [batch*n_comp, 1, seq_len]
            encoded = self.encoder(x)  # [batch*n_comp, channels, reduced_seq]
            # Reshape back
            _, channels, reduced_seq = encoded.shape
            encoded = encoded.view(batch_size, n_components, channels, reduced_seq)
            return encoded


class CycleDecoder(nn.Module):
    """Decoder for cycle-consistent unmixing (forward: mixture → components)."""

    def __init__(self, n_components: int, input_channels: int, input_size: int,
                 output_dim: int = 1250, hidden_dims: list = [64, 128, 256]):
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

        # Handle dimension mismatch
        if x.size(-1) < self.output_dim:
            pad_size = self.output_dim - x.size(-1)
            x = nn.functional.pad(x, (0, pad_size), mode='replicate')
        elif x.size(-1) > self.output_dim:
            x = x[..., :self.output_dim]

        return x


class ReverseDecoder(nn.Module):
    """Reverse decoder for cycle consistency (components → mixture)."""

    def __init__(self, input_channels: int, input_size: int,
                 output_dim: int = 1250, hidden_dims: list = [64, 128, 256]):
        super().__init__()

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

        # Final layer to output single mixture spectrum
        layers.append(
            nn.Conv1d(hidden_dims[-1], 1, kernel_size=5, padding=2)
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """Args:
        x: [batch, n_components, channels, seq_len] encoded component features
        """
        batch_size, n_components, channels, seq_len = x.shape

        # Combine component features (average pooling across components)
        x = x.mean(dim=1)  # [batch, channels, seq_len]

        x = self.decoder(x)  # [batch, 1, ~output_dim]
        x = x.squeeze(1)  # [batch, ~output_dim]

        # Ensure non-negative using Softplus instead of ReLU
        # Softplus has smooth gradients everywhere, preventing dying gradient problem
        x = nn.functional.softplus(x, beta=5)

        # Handle dimension mismatch
        if x.size(-1) < self.output_dim:
            pad_size = self.output_dim - x.size(-1)
            x = nn.functional.pad(x, (0, pad_size), mode='replicate')
        elif x.size(-1) > self.output_dim:
            x = x[..., :self.output_dim]

        return x


class CyCUNet(nn.Module):
    """Cycle-Consistent Unmixing Network (CyCU-Net)."""

    def __init__(self, n_components: int = 2, input_dim: int = 1250,
                 encoder_dims: list = [256, 128, 64],
                 decoder_dims: list = [64, 128, 256]):
        super().__init__()

        self.n_components = n_components
        self.input_dim = input_dim

        # Shared encoder (used for both forward and reverse)
        self.encoder = CycleEncoder(input_dim, encoder_dims)

        # Forward decoder (mixture → components)
        self.component_decoder = CycleDecoder(
            n_components=n_components,
            input_channels=self.encoder.output_channels,
            input_size=self.encoder.output_size,
            output_dim=input_dim,
            hidden_dims=decoder_dims
        )

        # Reverse encoder (for components → mixture)
        self.reverse_encoder = CycleEncoder(input_dim, encoder_dims)

        # Reverse decoder (components → mixture)
        self.mixture_decoder = ReverseDecoder(
            input_channels=self.reverse_encoder.output_channels,
            input_size=self.reverse_encoder.output_size,
            output_dim=input_dim,
            hidden_dims=decoder_dims
        )

        # Abundance predictor (from mixture encoding)
        self.abundance_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder.output_channels * self.encoder.output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_components),
            nn.Softmax(dim=-1)  # Sum-to-one constraint
        )

    def forward(self, mixture):
        """Forward pass with cycle consistency."""
        batch_size = mixture.size(0)

        # ===== Forward Path: Mixture → Components =====
        # Encode mixture
        mixture_encoded = self.encoder(mixture)  # [batch, channels, reduced_seq]

        # Predict abundances
        abundances = self.abundance_predictor(mixture_encoded)  # [batch, n_components]

        # Decode to components
        components = self.component_decoder(mixture_encoded)  # [batch, n_components, input_dim]

        # Forward reconstruction (weighted sum)
        reconstruction = torch.einsum('bn,bnd->bd', abundances, components)

        # ===== Reverse Path: Components → Mixture =====
        # Encode components
        components_encoded = self.reverse_encoder(components)  # [batch, n_components, channels, reduced_seq]

        # Decode to mixture
        cycle_mixture = self.mixture_decoder(components_encoded)  # [batch, input_dim]

        # ===== Cycle Path: Components → Mixture → Components =====
        # Encode cycle mixture
        cycle_mixture_encoded = self.encoder(cycle_mixture)  # [batch, channels, reduced_seq]

        # Decode to cycle components
        cycle_components = self.component_decoder(cycle_mixture_encoded)  # [batch, n_components, input_dim]

        return {
            "abundances": abundances,
            "components": components,
            "reconstruction": reconstruction,
            "cycle_mixture": cycle_mixture,
            "cycle_components": cycle_components,
        }

    def unmix(self, mixture: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unmix a single spectrum."""
        self.eval()
        with torch.no_grad():
            mixture_tensor = torch.FloatTensor(mixture).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                mixture_tensor = mixture_tensor.cuda()

            outputs = self.forward(mixture_tensor)

            return (
                outputs["components"].squeeze(0).cpu().numpy(),
                outputs["abundances"].squeeze(0).cpu().numpy(),
                outputs["reconstruction"].squeeze(0).cpu().numpy()
            )

    def unmix_batch(self, mixtures: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unmix a batch of spectra."""
        self.eval()
        with torch.no_grad():
            mixture_tensor = torch.FloatTensor(mixtures)
            if next(self.parameters()).is_cuda:
                mixture_tensor = mixture_tensor.cuda()

            outputs = self.forward(mixture_tensor)

            return (
                outputs["components"].cpu().numpy(),
                outputs["abundances"].cpu().numpy(),
                outputs["reconstruction"].cpu().numpy()
            )
