"""NNLS (Non-Negative Least Squares) Baseline for IR Spectral Unmixing."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls as scipy_nnls
from typing import Dict, List, Optional, Tuple


class NNLS:
    """Non-Negative Least Squares unmixing baseline."""

    def __init__(
        self,
        n_components: int = 2,
        maxiter: int | None = None,
    ):
        """Args:
        n_components: Number of components to extract (top-K coefficients).
        """
        self.n_components = n_components
        self.maxiter = maxiter
        self.endmember_library_: Optional[np.ndarray] = None  # [L, N_lib]

    def fit(
        self,
        mixtures: np.ndarray,
        reference_spectra: Optional[np.ndarray] = None,
    ) -> None:
        """Build the spectral library for NNLS."""
        if reference_spectra is not None and len(reference_spectra) > 0:
            # Library matrix A: [L, N_lib] (each column is one library spectrum)
            self.endmember_library_ = reference_spectra.T.astype(np.float64)
        else:
            # Fallback: use training mixtures as library (not recommended)
            self.endmember_library_ = mixtures.T.astype(np.float64)

        n_bands, n_lib = self.endmember_library_.shape
        print(f"  NNLS library: {n_lib} spectra × {n_bands} wavenumbers")

    def _solve_single(self, mixture: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve NNLS for a single mixture spectrum."""
        A = self.endmember_library_  # [L, N_lib]
        b = mixture.astype(np.float64)

        if self.maxiter is not None:
            coeffs, residual = scipy_nnls(A, b, maxiter=self.maxiter)
        else:
            coeffs, residual = scipy_nnls(A, b)

        return coeffs, residual

    def unmix(self, mixture: np.ndarray) -> Dict[str, np.ndarray]:
        """Unmix a single mixture spectrum."""
        if self.endmember_library_ is None:
            raise ValueError("Must call fit() before unmix().")

        coeffs, residual = self._solve_single(mixture)

        # Select top-K by coefficient magnitude
        top_k_idx = np.argsort(coeffs)[::-1][: self.n_components]
        top_coeffs = coeffs[top_k_idx]

        # Normalize abundances
        coeff_sum = top_coeffs.sum()
        abundances = top_coeffs / coeff_sum if coeff_sum > 1e-12 else top_coeffs

        # Component spectra = the selected library entries
        components = self.endmember_library_[:, top_k_idx].T  # [K, L]

        # Full reconstruction
        reconstruction = (self.endmember_library_ @ coeffs).astype(np.float32)

        return {
            "components": components.astype(np.float32),
            "abundances": abundances.astype(np.float32),
            "reconstruction": reconstruction,
            "residual": float(residual),
            "top_indices": top_k_idx,
            "all_coefficients": coeffs.astype(np.float32),
        }

    def predict_batch(self, mixtures: np.ndarray) -> Dict[str, np.ndarray]:
        """Unmix a batch of mixture spectra."""
        if self.endmember_library_ is None:
            raise ValueError("Must call fit() before predict_batch().")

        B, L = mixtures.shape
        K = self.n_components

        all_components = np.zeros((B, K, L), dtype=np.float32)
        all_abundances = np.zeros((B, K), dtype=np.float32)
        all_reconstructions = np.zeros((B, L), dtype=np.float32)
        all_residuals = np.zeros(B, dtype=np.float32)
        all_top_indices = np.zeros((B, K), dtype=np.int64)

        for i in range(B):
            result = self.unmix(mixtures[i])
            all_components[i] = result["components"]
            all_abundances[i] = result["abundances"]
            all_reconstructions[i] = result["reconstruction"]
            all_residuals[i] = result["residual"]
            all_top_indices[i] = result["top_indices"]

        return {
            "components": all_components,
            "abundances": all_abundances,
            "reconstruction": all_reconstructions,
            "residual": all_residuals,
            "top_indices": all_top_indices,
        }
