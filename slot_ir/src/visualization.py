from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

from .config import VisualizationConfig, DATA_ROOT


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _wavenumber_axis(length: int = 1250) -> np.ndarray:
    """Return wavenumber axis from high (4000) to low (400) cm⁻¹ to match spectroscopy convention."""
    return np.linspace(4000, 400, num=length)


def _reverse_data_for_plotting(data: np.ndarray) -> np.ndarray:
    """Reverse data array along wavenumber dimension for plotting."""
    if data.ndim == 1:
        return np.flip(data)
    elif data.ndim == 2:
        return np.flip(data, axis=1)
    else:
        # For higher dimensions, flip along last axis (wavenumber dimension)
        return np.flip(data, axis=-1)


def _load_smiles_list() -> Optional[List[str]]:
    """Load SMILES list from metadata.json if available."""
    try:
        metadata_path = DATA_ROOT / "processed" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get("smiles_list", None)
    except Exception:
        pass
    return None


def plot_translation_examples(
    gas: torch.Tensor,
    liq_pred: torch.Tensor,
    liq_true: torch.Tensor,
    cfg: VisualizationConfig,
    tag: str,
    max_samples: int = 4,
    sample_indices: Optional[List[int]] = None,
) -> Path:
    """Plot gas → liquid translation examples for Stage 1 interpretability."""
    save_dir = cfg.save_dir / "visualization" / "translation"
    _ensure_dir(save_dir)
    
    # Create individual samples subdirectory
    individual_dir = save_dir / "individual"
    _ensure_dir(individual_dir)
    
    # Load SMILES if available
    smiles_list = _load_smiles_list()
    
    n_samples = min(gas.shape[0], max_samples)
    wavenumbers = _wavenumber_axis(gas.shape[-1])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples), dpi=cfg.dpi)
    if n_samples == 1:
        axes = [axes]
    
    for idx in range(n_samples):
        ax = axes[idx]
        # Reverse data for plotting: index 0 = 400 -> becomes index 0 = 4000
        gas_data = _reverse_data_for_plotting(gas[idx].cpu().numpy())
        liq_true_data = _reverse_data_for_plotting(liq_true[idx].cpu().numpy())
        liq_pred_data = _reverse_data_for_plotting(liq_pred[idx].cpu().numpy())
        
        ax.plot(wavenumbers, gas_data, color=cfg.color_gas, label="Gas", alpha=0.8, linewidth=1.5)
        ax.plot(wavenumbers, liq_true_data, color=cfg.color_liquid_true, label="Liquid (True)", alpha=0.8, linewidth=1.5)
        ax.plot(wavenumbers, liq_pred_data, color=cfg.color_liquid_pred, label="Liquid (Pred)", alpha=0.8, linewidth=1.5, linestyle="--")
        
        # Compute metrics for this sample
        mse = torch.nn.functional.mse_loss(liq_pred[idx], liq_true[idx]).item()
        cos_sim = torch.nn.functional.cosine_similarity(liq_pred[idx:idx+1], liq_true[idx:idx+1]).item()
        
        # Get SMILES if available
        smiles_str = ""
        if smiles_list and sample_indices and idx < len(sample_indices):
            dataset_idx = sample_indices[idx]
            if dataset_idx < len(smiles_list):
                smiles_str = f" | SMILES: {smiles_list[dataset_idx]}"
        
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Sample {idx+1} | MSE: {mse:.6f}, Cosine: {cos_sim:.4f}{smiles_str}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(4000, 400)  # Show 4000 on left, 400 on right
        
        # Save individual plot
        fig_individual = plt.figure(figsize=(10, 4), dpi=cfg.dpi)
        ax_individual = fig_individual.add_subplot(111)
        ax_individual.plot(wavenumbers, gas_data, color=cfg.color_gas, label="Gas", alpha=0.8, linewidth=1.5)
        ax_individual.plot(wavenumbers, liq_true_data, color=cfg.color_liquid_true, label="Liquid (True)", alpha=0.8, linewidth=1.5)
        ax_individual.plot(wavenumbers, liq_pred_data, color=cfg.color_liquid_pred, label="Liquid (Pred)", alpha=0.8, linewidth=1.5, linestyle="--")
        ax_individual.set_xlabel("Wavenumber (cm⁻¹)")
        ax_individual.set_ylabel("Intensity (a.u.)")
        ax_individual.set_title(f"Sample {idx+1} | MSE: {mse:.6f}, Cosine: {cos_sim:.4f}{smiles_str}")
        ax_individual.legend(loc="upper right")
        ax_individual.grid(True, alpha=0.3)
        ax_individual.set_xlim(4000, 400)
        plt.tight_layout()
        individual_path = individual_dir / f"{tag}_sample{idx}.png"
        plt.savefig(individual_path)
        plt.close(fig_individual)
    
    plt.tight_layout()
    path = save_dir / f"{tag}_translation.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_mixture_breakdown(
    mixture: torch.Tensor,
    components: torch.Tensor,
    cfg: VisualizationConfig,
    tag: str,
    components_true: Optional[torch.Tensor] = None,
    save_dir: Optional[Path] = None,
) -> Path:
    """Plot mixture breakdown with predicted and optionally ground truth components."""
    if save_dir is None:
        save_dir = cfg.save_dir / "visualization" / "mixture_breakdown"
    else:
        save_dir = Path(save_dir)
    _ensure_dir(save_dir)
    wavenumbers = _wavenumber_axis(mixture.shape[-1])
    
    # Reverse data for plotting: index 0 = 400 -> becomes index 0 = 4000
    mixture_data = _reverse_data_for_plotting(mixture.cpu().numpy())
    
    # Determine number of subplots
    if components_true is not None:
        n_subplots = 3  # Combined, Pred only, GT only
    else:
        n_subplots = 2  # Combined (mixture+pred), Pred only
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 5 * n_subplots), dpi=cfg.dpi)
    if n_subplots == 1:
        axes = [axes]
    
    subplot_idx = 0
    
    # Define component colors (support up to 3+ components)
    component_colors = [
        cfg.color_component_a,
        cfg.color_component_b,
        getattr(cfg, 'color_component_c', '#6a994e'),  # green for 3rd component
    ]
    # Add more colors if needed (using matplotlib's default color cycle)
    try:
        # New matplotlib API (3.7+)
        color_cycle = plt.colormaps['tab10']
    except (AttributeError, KeyError):
        # Fallback for matplotlib without this API
        color_cycle = cm.get_cmap('tab10')
    for i in range(3, 10):
        # Normalize index to [0, 1] range for colormap
        normalized_idx = (i % 10) / 10.0
        rgba = color_cycle(normalized_idx)
        # Convert RGBA tuple to hex color
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        component_colors.append(hex_color)
    
    # Subplot 1: Mixture + Predicted (+ Ground Truth) Components
    ax = axes[subplot_idx]
    ax.plot(wavenumbers, mixture_data, color=cfg.color_mixture, label="Mixture (GT)", 
            linewidth=2.0, alpha=0.2)
    num_components = components.shape[0]
    for idx in range(num_components):
        color = component_colors[idx % len(component_colors)]
        comp_data = _reverse_data_for_plotting(components[idx].cpu().numpy())
        ax.plot(wavenumbers, comp_data, color=color, label=f"Component {idx+1} (Pred)", 
                linewidth=2.0, linestyle="-", alpha=0.6)
    if components_true is not None:
        num_components_true = components_true.shape[0]
        for idx in range(num_components_true):
            color = component_colors[idx % len(component_colors)]
            comp_true_data = _reverse_data_for_plotting(components_true[idx].cpu().numpy())
            ax.plot(
                wavenumbers,
                comp_true_data,
                color=color,
                label=f"Component {idx+1} (GT)",
                linewidth=2.0,
                linestyle="--",
                alpha=0.6,
            )
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity (a.u.)", fontsize=11)
    title = "Mixture + Predicted Components"
    if components_true is not None:
        title += " (with Ground Truth)"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4000, 400)
    subplot_idx += 1
    
    # Subplot 2: Predicted Components Only
    ax = axes[subplot_idx]
    num_components = components.shape[0]
    for idx in range(num_components):
        color = component_colors[idx % len(component_colors)]
        comp_data = _reverse_data_for_plotting(components[idx].cpu().numpy())
        ax.plot(wavenumbers, comp_data, color=color, label=f"Component {idx+1} (Pred)", 
                linewidth=2.0, linestyle="-", alpha=0.9)
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity (a.u.)", fontsize=11)
    ax.set_title("Predicted Components", fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4000, 400)
    subplot_idx += 1
    
    # Subplot 3: Ground Truth Components Only (if available)
    if components_true is not None:
        ax = axes[subplot_idx]
        num_components_true = components_true.shape[0]
        for idx in range(num_components_true):
            color = component_colors[idx % len(component_colors)]
            comp_true_data = _reverse_data_for_plotting(components_true[idx].cpu().numpy())
            ax.plot(wavenumbers, comp_true_data, color=color, label=f"Component {idx+1} (GT)", 
                    linewidth=2.0, linestyle="-", alpha=0.9)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Intensity (a.u.)", fontsize=11)
        ax.set_title("Ground Truth Components", fontsize=12, fontweight='bold')
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(4000, 400)
    
    plt.tight_layout()
    path = save_dir / f"{tag}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def plot_attention_maps(attn: Dict[str, torch.Tensor], cfg: VisualizationConfig, tag: str) -> Path:
    """Plot attention maps from routing transformer."""
    save_dir = cfg.save_dir / "visualization" / "attention"
    _ensure_dir(save_dir)
    
    if "cross" not in attn:
        print(f"[Warning] No cross-attention found in attn dict. Keys: {attn.keys()}")
        return save_dir / f"{tag}_cross.png"
    
    cross_attn = attn["cross"]
    # Handle MultiheadAttention output shape: [batch, num_heads, seq_len, seq_len] or [batch, seq_len, seq_len]
    if cross_attn.dim() == 4:
        # Average over heads: [batch, num_heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
        cross_attn = cross_attn.mean(dim=1)
    
    # Average over batch and convert to numpy
    heatmap = cross_attn.mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(12, 4), dpi=cfg.dpi)
    plt.imshow(heatmap, aspect="auto", cmap=cfg.attention_cmap)
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Memory Tokens")
    plt.ylabel("Component Slots")
    plt.title("Cross-Attention")
    plt.tight_layout()
    path = save_dir / f"{tag}_cross.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_loss_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    cfg: VisualizationConfig,
    stage: int,
) -> Path:
    """Plot training and validation loss curves."""
    save_dir = cfg.save_dir / "visualization" / "loss_curves"
    _ensure_dir(save_dir)
    
    # Plot total loss
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=cfg.dpi)
    axes = axes.flatten()
    
    # Total loss
    if "total" in train_losses and "total" in val_losses:
        ax = axes[0]
        ax.plot(train_losses["total"], label="Train", alpha=0.7)
        if len(val_losses["total"]) > 0:
            val_steps = np.linspace(0, len(train_losses["total"]) - 1, len(val_losses["total"]))
            ax.plot(val_steps, val_losses["total"], label="Val", marker="o", markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Individual losses
    plot_idx = 1
    for key in train_losses:
        if key == "total" or plot_idx >= 4:
            continue
        ax = axes[plot_idx]
        ax.plot(train_losses[key], label=f"Train {key}", alpha=0.7)
        if key in val_losses and len(val_losses[key]) > 0:
            val_steps = np.linspace(0, len(train_losses[key]) - 1, len(val_losses[key]))
            ax.plot(val_steps, val_losses[key], label=f"Val {key}", marker="o", markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{key.capitalize()} Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 4):
        axes[idx].axis("off")
    
    plt.tight_layout()
    path = save_dir / f"stage{stage}_losses.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_evaluation_metrics(
    metrics_history: Dict[int, Dict[str, float]],
    cfg: VisualizationConfig,
    stage: int,
) -> Path:
    """Plot evaluation metrics over time."""
    save_dir = cfg.save_dir / "visualization" / "metrics"
    _ensure_dir(save_dir)
    
    # Group metrics by name
    metric_names = set()
    for metrics in metrics_history.values():
        metric_names.update(metrics.keys())
    metric_names = sorted([m for m in metric_names if m not in ["total", "translation", "component", "mixture", "boundary", "residual", "confidence", "similarity"]])
    
    if not metric_names:
        return save_dir / "empty.png"
    
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=cfg.dpi)
    
    # Handle axes array properly (matplotlib returns different types)
    if n_metrics == 1:
        # Single subplot: axes is Axes object
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        # Multiple subplots: axes is numpy array
        axes = list(axes.flatten())
    else:
        # Shouldn't happen, but handle gracefully
        axes = [axes]
    
    steps = sorted(metrics_history.keys())
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics_history[step].get(metric_name, 0) for step in steps]
        ax.plot(steps, values, marker="o", markersize=4)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(metric_names), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    path = save_dir / f"stage{stage}_metrics.png"
    plt.savefig(path)
    plt.close()
    return path
