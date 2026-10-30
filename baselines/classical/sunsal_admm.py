"""SUnSAL (Sparse Unmixing by variable Splitting and Augmented Lagrangian), as published."""

from typing import Dict, Optional

import numpy as np
import torch


class SUnSALADMM:
    def __init__(self, n_components: int = 2, lambda_reg: float = 1e-3, mu0: float = 0.01,
                 max_iter: int = 5000, tol: float = 1e-6, device: Optional[str] = None):
        self.n_components = n_components
        self.lambda_reg = lambda_reg
        self.mu0 = mu0
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, mixtures: Optional[np.ndarray] = None, reference_spectra: np.ndarray = None) -> None:
        """Store the library. No training; `mixtures` is ignored."""
        self.library_ = reference_spectra.astype(np.float32)                   # [n, L]
        A = torch.from_numpy(self.library_).double().to(self.device).T          # [L, n]
        # A = U S W^T, so (A^T A + mu I)^-1 v = (v - W diag(s^2 / (s^2 + mu)) W^T v) / mu
        _, s, Wt = torch.linalg.svd(A, full_matrices=False)
        self._A = A
        self._W = Wt.T                                                          # [n, r]
        self._s2 = s ** 2                                                       # [r]

    def _apply_inverse(self, V: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """(A^T A + mu_j I)^-1 applied to each column v_j of V, with one mu per column."""
        WtV = self._W.T @ V                                                     # [r, N]
        D = self._s2[:, None] / (self._s2[:, None] + mu[None, :])               # [r, N]
        return (V - self._W @ (D * WtV)) / mu[None, :]

    def solve(self, mixtures: np.ndarray) -> np.ndarray:
        """Return non-negative sparse weights [N, n]."""
        Y = torch.from_numpy(mixtures).double().to(self.device).T               # [L, N]
        n, N = self._A.shape[1], Y.shape[1]
        lam = self.lambda_reg
        AtY = self._A.T @ Y                                                     # [n, N]
        mu = torch.full((N,), self.mu0, dtype=torch.float64, device=self.device)

        x = self._apply_inverse(AtY, mu)
        z = x.clone()
        d = torch.zeros_like(x)
        tol = np.sqrt(n) * self.tol
        active = torch.ones(N, dtype=torch.bool, device=self.device)
        z0 = z.clone()

        for i in range(1, self.max_iter + 1):
            if i % 10 == 1:
                z0 = z.clone()
            # z step: soft threshold, then project onto x >= 0
            v = x - d
            z_new = torch.clamp(torch.sign(v) * torch.clamp(v.abs() - lam / mu[None, :], min=0), min=0)
            # x step
            x_new = self._apply_inverse(AtY + mu[None, :] * (z_new + d), mu)
            d_new = d - (x_new - z_new)
            # only update spectra that have not converged
            m = active[None, :]
            z = torch.where(m, z_new, z)
            x = torch.where(m, x_new, x)
            d = torch.where(m, d_new, d)

            if i % 10 == 1:
                res_p = torch.linalg.norm(x - z, dim=0)
                res_d = mu * torch.linalg.norm(z - z0, dim=0)
                active = active & ((res_p > tol) | (res_d > tol))
                if not bool(active.any()):
                    break
                up = active & (res_p > 10 * res_d)
                down = active & (res_d > 10 * res_p)
                mu = torch.where(up, mu * 2, torch.where(down, mu / 2, mu))
                d = torch.where(up[None, :], d / 2, torch.where(down[None, :], d * 2, d))

        return z.T.cpu().numpy()

    def predict_batch(self, mixtures: np.ndarray) -> Dict[str, np.ndarray]:
        weights = self.solve(mixtures)
        top = np.argsort(-weights, axis=1)[:, :self.n_components]
        return {
            "components": self.library_[top],
            "abundances": np.take_along_axis(weights, top, axis=1),
            "library_indices": top,
        }
