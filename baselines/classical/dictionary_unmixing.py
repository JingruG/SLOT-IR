"""MCR-ALS and VCA+FCLS with one learned spectrum per molecule (no reference library)."""

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def nnls_many(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve min ||A x_j - b_j|| s.t. x_j >= 0 for every column b_j of B."""
    Q, R = np.linalg.qr(A, mode="reduced")
    QB = Q.T @ B
    X = np.zeros((A.shape[1], B.shape[1]))
    for j in range(B.shape[1]):
        X[:, j], _ = nnls(R, QB[:, j])
    return X


def fcls(endmembers: np.ndarray, Y: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """Fully constrained least squares (Heinz and Chang 2001)."""
    R = endmembers.shape[0]
    A = np.vstack([delta * endmembers.T, np.ones((1, R))])      # [L+1, R]
    B = np.vstack([delta * Y.T, np.ones((1, Y.shape[0]))])      # [L+1, N]
    return nnls_many(A, B).T


# ---------------------------------------------------------------------------
# VCA (Nascimento and Bioucas-Dias 2005), following the authors' algorithm
# ---------------------------------------------------------------------------

def _estimate_snr(Y: np.ndarray, r_m: np.ndarray, x: np.ndarray) -> float:
    L, N = Y.shape
    p = x.shape[0]
    P_y = np.sum(Y ** 2) / N
    P_x = np.sum(x ** 2) / N + r_m @ r_m
    return 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))


def vca(Y: np.ndarray, R: int, seed: int = 0):
    """Vertex component analysis."""
    rng = np.random.default_rng(seed)
    L, N = Y.shape

    y_m = Y.mean(axis=1)
    Y_o = Y - y_m[:, None]
    Ud = np.linalg.svd(Y_o @ Y_o.T / N)[0][:, :R]
    x_p = Ud.T @ Y_o
    snr = _estimate_snr(Y, y_m, x_p)
    snr_th = 15 + 10 * np.log10(R)

    if snr < snr_th:
        # Low SNR: project onto the (R-1)-dim affine subspace
        d = R - 1
        Ud = Ud[:, :d]
        Yp = Ud @ x_p[:d] + y_m[:, None]
        x = x_p[:d]
        c = np.max(np.linalg.norm(x, axis=0))
        y = np.vstack([x, c * np.ones((1, N))])
    else:
        # High SNR: projective projection onto the R-dim subspace
        d = R
        Ud = np.linalg.svd(Y @ Y.T / N)[0][:, :d]
        x_p = Ud.T @ Y
        Yp = Ud @ x_p
        x = Ud.T @ Y
        u = x.mean(axis=1)
        y = x / (u @ x)[None, :]

    indices = np.zeros(R, dtype=int)
    A = np.zeros((R, R))
    A[-1, 0] = 1
    for i in range(R):
        w = rng.standard_normal(R)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / np.linalg.norm(f)
        v = f @ y
        indices[i] = int(np.argmax(np.abs(v)))
        A[:, i] = y[:, indices[i]]

    return Yp[:, indices].T, indices, float(snr)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _top_k_components(spectra: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """For each mixture, return the k spectra with the largest weights. [N, k, L]"""
    top = np.argsort(-weights, axis=1)[:, :k]
    return spectra[top]


class VCAFCLSDictionary:
    """VCA extracts M endmembers from the training mixtures; FCLS gives weights."""

    def __init__(self, n_components: int, n_spectra: int, seed: int = 0):
        self.n_components = n_components
        self.n_spectra = n_spectra
        self.seed = seed

    def fit(self, mixtures: np.ndarray, save_path: Optional[Path] = None) -> None:
        if save_path is not None and Path(save_path).exists():
            self.spectra_ = np.load(save_path)["spectra"]
            print(f"  Loaded learned spectra from {save_path}")
            return
        t = time.time()
        self.spectra_, idx, snr = vca(mixtures.T.astype(np.float64), self.n_spectra, seed=self.seed)
        print(f"  VCA: {self.n_spectra} endmembers, SNR estimate {snr:.1f} dB, "
              f"{len(np.unique(idx))} distinct spectra selected ({time.time() - t:.1f}s)")
        if save_path is not None:
            np.savez(save_path, spectra=self.spectra_, indices=idx, snr=snr)

    def predict_batch(self, mixtures: np.ndarray) -> Dict[str, np.ndarray]:
        weights = fcls(self.spectra_, mixtures.astype(np.float64))
        return {
            "components": _top_k_components(self.spectra_, weights, self.n_components).astype(np.float32),
            "abundances": weights,
        }


class MCRALSDictionary:
    """MCR-ALS with M components: Y ~ C S, C >= 0, S >= 0 (Tauler 1995)."""

    def __init__(self, n_components: int, n_spectra: int, max_iter: int = 200,
                 tol: float = 1e-3, seed: int = 0):
        self.n_components = n_components
        self.n_spectra = n_spectra
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    @staticmethod
    def _normalize(S: np.ndarray) -> np.ndarray:
        return S / (S.max(axis=1, keepdims=True) + 1e-12)

    def fit(self, mixtures: np.ndarray, save_path: Optional[Path] = None) -> None:
        Y = mixtures.astype(np.float64)                       # [N, L]
        ckpt = Path(str(save_path) + ".ckpt.npz") if save_path is not None else None

        if save_path is not None and Path(save_path).exists():
            self.spectra_ = np.load(save_path)["spectra"]
            print(f"  Loaded learned spectra from {save_path}")
            return

        if ckpt is not None and ckpt.exists():
            z = np.load(ckpt)
            S, start, history = z["spectra"], int(z["iteration"]) + 1, list(z["lack_of_fit"])
            print(f"  Resuming MCR-ALS from iteration {start}")
        else:
            S0, _, snr = vca(Y.T, self.n_spectra, seed=self.seed)
            S = self._normalize(np.maximum(S0, 0))
            start, history = 0, []
            print(f"  MCR-ALS initialized with {self.n_spectra} VCA spectra (SNR estimate {snr:.1f} dB)")

        norm_Y = np.linalg.norm(Y)
        for it in range(start, self.max_iter):
            t = time.time()
            C = nnls_many(S.T, Y.T).T                         # [N, M]
            S = nnls_many(C, Y)                               # [M, L]
            C = C * (S.max(axis=1) + 1e-12)[None, :]
            S = self._normalize(S)
            lof = 100 * np.linalg.norm(Y - C @ S) / norm_Y   # lack of fit, %
            history.append(lof)
            change = abs(history[-2] - lof) / history[-2] if len(history) > 1 else np.inf
            print(f"  iter {it:3d}  lack of fit {lof:.4f}%  rel. change {change:.2e}  ({time.time() - t:.1f}s)",
                  flush=True)
            if ckpt is not None:
                np.savez(ckpt, spectra=S, iteration=it, lack_of_fit=np.array(history))
            if change < self.tol:
                break

        self.spectra_ = S
        self.lack_of_fit_ = history
        if save_path is not None:
            np.savez(save_path, spectra=S, lack_of_fit=np.array(history))

    def predict_batch(self, mixtures: np.ndarray) -> Dict[str, np.ndarray]:
        weights = nnls_many(self.spectra_.T, mixtures.astype(np.float64).T).T
        return {
            "components": _top_k_components(self.spectra_, weights, self.n_components).astype(np.float32),
            "abundances": weights,
        }
