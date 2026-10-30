"""Spectral augmentation for Stage 2 mixture training."""

from __future__ import annotations

import numpy as np


def shift_windows(
    spectrum,
    *,
    seed=None,
    percent=0.2,
    window_width=30,
    shift_range=5.0,
    blend=True,
):
    """Randomly shift local spectral windows to add positional noise."""
    if shift_range == 0 or percent <= 0 or window_width <= 1:
        return np.asarray(spectrum, dtype=float).copy()

    y = np.asarray(spectrum, dtype=float)
    out = y.copy()
    rng = np.random.default_rng(seed)

    half = window_width // 2
    valid_lo, valid_hi = half, max(half, len(y) - half - 1)
    n_points = min(int(len(y) * percent), len(y))
    centers = rng.choice(np.arange(valid_lo, valid_hi + 1), n_points, replace=False)

    for p in centers:
        start, end = p - half, p - half + window_width
        window = y[start:end]

        shift = rng.normal(0.0, shift_range)
        s_int = int(np.trunc(shift))
        s_frac = shift - s_int

        x = np.arange(window_width)
        shifted_window = np.interp(
            x - s_frac, x, window, left=window[0], right=window[-1]
        )

        insert_at = int(np.clip(start + s_int, 0, len(y) - window_width))
        if blend:
            t = np.arange(window_width)
            win = 0.5 * (1 - np.cos(2 * np.pi * (t / (window_width - 1))))
            existing = out[insert_at : insert_at + window_width]
            out[insert_at : insert_at + window_width] = (
                (1 - win) * existing + win * shifted_window
            )
        else:
            out[insert_at : insert_at + window_width] = shifted_window

    scale = y.sum() / out.sum() if out.sum() != 0 else 1
    out = out * scale
    return out


def augment_sample(
    components: np.ndarray,
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize mixture from components with optional variable concentration."""
    K, L = components.shape
    augmented = np.zeros_like(components)

    # Apply shift_windows to each component individually
    for k in range(K):
        spec = components[k].copy()
        seed_val = int(rng.integers(0, 2**31))
        spec = shift_windows(
            spec,
            seed=seed_val,
            percent=cfg.shift_percent,
            window_width=cfg.shift_window_width,
            shift_range=cfg.shift_range,
            blend=True,
        )
        augmented[k] = spec

    # Mixing
    if getattr(cfg, "variable_concentration", False):
        # Sample weights from Dirichlet, reject if any below concentration_min
        conc_min = getattr(cfg, "concentration_min", 0.1)
        for _ in range(100):
            weights = rng.dirichlet(np.ones(K))
            if np.all(weights >= conc_min):
                break
        mixture = np.einsum("k,kl->l", weights, augmented)
    else:
        # Equal-weight mixing (accounts for dilution in equal-volume mixtures)
        mixture = augmented.mean(axis=0)

    return mixture
