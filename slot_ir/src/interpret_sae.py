#!/usr/bin/env python3
"""Train a BatchTopK SAE on frozen SLOT-IR slot embeddings and test its features against functional groups.

Usage (from slot_ir/):
    python -m src.interpret_sae --config configs/stage2_binary.json \
        --checkpoint results/main/binary/seed_0/checkpoints/stage2_best.pt --output-dir results/sae
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses import match_slots_l2
from src.config import load_config, DATA_ROOT
from src.data import MixtureDataset, split_indices, _load_mixture_dataset, build_dataloader
from src.models import SlotUnmixer
from src.sae import BatchTopKSAE
from src.train_sae import train_sae


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_FG_NAMES = ['OH', 'NCO', 'C-O-C', 'COO', 'C-Cl', 'C=C', 'COOH', 'CONH', 'NH2', 'Ar-O-C']

def _load_ir_bands_config() -> Tuple[Dict, Dict]:
    """Load IR band definitions and FG-to-band mappings from configs/ir_bands.json."""
    config_path = Path(__file__).parent.parent / 'configs' / 'ir_bands.json'
    if not config_path.exists():
        print(f"  Warning: {config_path} not found, IR band overlays disabled")
        return {}, {}
    with open(config_path) as f:
        data = json.load(f)
    bands = {k: tuple(v) for k, v in data.get('known_ir_bands', {}).items()}
    fg_map = data.get('fg_to_bands', {})
    return bands, fg_map


KNOWN_IR_BANDS, FG_TO_BANDS = _load_ir_bands_config()


def load_fg_names(n_fgs: Optional[int] = None) -> List[str]:
    """Load functional group names from metadata.json."""
    metadata_path = DATA_ROOT / "processed" / "metadata.json"
    if not metadata_path.exists():
        names = DEFAULT_FG_NAMES
    else:
        with open(metadata_path) as f:
            metadata = json.load(f)
        # Use the full ordered list
        if 'functional_group_names' in metadata:
            names = metadata['functional_group_names']
        elif 'selected_functional_group_names' in metadata:
            names = metadata['selected_functional_group_names']
        else:
            names = DEFAULT_FG_NAMES

    if n_fgs is not None:
        return names[:n_fgs]
    return names


def get_wavenumbers(n_points: int = 1250) -> np.ndarray:
    """Get wavenumber array (400-4000 cm^-1)."""
    return np.linspace(400, 4000, n_points)


# ── SAE Training / Embedding Collection ─────────────────────────────────────

class SAETrainer:
    """Train SAE on slot embeddings using existing train_sae from src.train_sae."""

    def __init__(
        self,
        d_input: int = 512,
        d_latent: int = 1024,
        sparsity_percent: float = 0.05,
        k_fixed: Optional[int] = None,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.sae = BatchTopKSAE(
            d_input=d_input,
            d_latent=d_latent,
            sparsity_percent=sparsity_percent,
            k_fixed=k_fixed,
            tied_weights=False,
            normalize_decoder=True
        ).to(self.device)

    def collect_embeddings(
        self,
        model: SlotUnmixer,
        dataset: MixtureDataset,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Collect slot embeddings from all samples in dataset."""
        model.eval()
        all_embeddings = []
        all_metadata = []

        # Get functional groups from molecule database
        molecule_db = dataset.molecule_db
        fg_matrix = molecule_db.get('functional_groups', None)  # [N_molecules, n_fg]

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )

        sample_offset = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting embeddings"):
                mixture = batch['mixture'].to(self.device)
                B = mixture.shape[0]
                comp_indices = batch.get('indices', None)  # [B, K] or None

                tokens = model.tokenizer(mixture)['tokens']
                memory = model.encoder(tokens)
                routing = model.routing(memory)
                slots = routing['slots']  # [B, K, d_model]
                K = slots.shape[1]
                # Slots have no fixed order: give each slot the labels of the true
                # molecule matched to its decoded spectrum (L2 Hungarian, as in training).
                matched = match_slots_l2(model.prediction(slots)['components'],
                                         batch['components'].to(self.device))

                for b in range(B):
                    for slot_idx in range(K):
                        all_embeddings.append(slots[b, slot_idx].cpu())

                        component_fg = None
                        if fg_matrix is not None and comp_indices is not None:
                            cidx = comp_indices[b, matched[b, slot_idx]].item()
                            component_fg = fg_matrix[cidx].numpy()

                        all_metadata.append({
                            'sample_idx': sample_offset + b,
                            'slot_idx': slot_idx,
                            'component_fgs': component_fg,
                        })

                sample_offset += B

        embeddings = torch.stack(all_embeddings)  # [N, d_model]
        return embeddings, all_metadata

    def train(
        self,
        embeddings: torch.Tensor,
        n_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3
    ) -> Dict:
        """Train SAE on embeddings using src.train_sae."""
        n_total = len(embeddings)
        n_train = int(0.8 * n_total)

        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(0))
        train_data = embeddings[perm[:n_train]]
        val_data = embeddings[perm[n_train:]]

        self.sae = train_sae(
            sae=self.sae,
            train_data=train_data,
            val_data=val_data,
            device=self.device,
            epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
        )

        # Compute final stats
        self.sae.eval()
        with torch.no_grad():
            sample_batch = embeddings[:min(256, n_total)].to(self.device)
            recon, stats = self.sae(sample_batch, return_stats=True)
            final_loss = F.mse_loss(recon, sample_batch).item()
            final_L0 = stats['L0']

        return {
            'loss': [final_loss],
            'L0': [final_L0],
        }


# ── Feature-FG Correlation Analysis ─────────────────────────────────────────

class FeatureFGCorrelationAnalyzer:
    """Analyze correlation between SAE features and functional groups."""

    def __init__(self, sae: BatchTopKSAE, device: str = 'cuda'):
        self.sae = sae
        self.device = device
        self.sae.eval()

    def analyze(
        self,
        embeddings: torch.Tensor,
        metadata: List[Dict],
        fg_names: List[str],
    ) -> Dict:
        """Compute feature-FG correlations for all provided FGs."""
        n_fgs = len(fg_names)

        with torch.no_grad():
            activations = self.sae.get_activations(embeddings.to(self.device))
            activations = activations.cpu().numpy()

        # Build FG matrix (use all available columns, clamp to n_fgs)
        fg_matrix = []
        valid_indices = []
        for i, meta in enumerate(metadata):
            if meta['component_fgs'] is not None:
                row = meta['component_fgs']
                if len(row) >= n_fgs:
                    fg_matrix.append(row[:n_fgs])
                else:
                    # Pad with zeros if data has fewer columns than requested
                    padded = np.zeros(n_fgs)
                    padded[:len(row)] = row
                    fg_matrix.append(padded)
                valid_indices.append(i)

        if len(valid_indices) == 0:
            return {'error': 'No FG labels available'}

        fg_matrix = np.array(fg_matrix)
        activations_valid = activations[valid_indices]

        # Compute correlations
        n_features = activations_valid.shape[1]
        correlations = np.zeros((n_features, n_fgs))
        p_values = np.zeros((n_features, n_fgs))

        for f in range(n_features):
            if activations_valid[:, f].std() > 1e-8:
                for g in range(n_fgs):
                    if fg_matrix[:, g].std() > 1e-8:
                        corr, pval = spearmanr(activations_valid[:, f], fg_matrix[:, g])
                        correlations[f, g] = corr if not np.isnan(corr) else 0
                        p_values[f, g] = pval if not np.isnan(pval) else 1

        # Find top features per FG
        top_features_per_fg = {}
        for g, fg_name in enumerate(fg_names):
            sorted_idx = np.argsort(np.abs(correlations[:, g]))[::-1]
            top_features_per_fg[fg_name] = [
                {
                    'feature_idx': int(idx),
                    'correlation': float(correlations[idx, g]),
                    'p_value': float(p_values[idx, g])
                }
                for idx in sorted_idx[:10]
            ]

        # Find monosemantic features (high correlation with ONE FG, low with others)
        monosemantic_features = []
        for f in range(n_features):
            abs_corrs = np.abs(correlations[f])
            if abs_corrs.max() > 0.3:
                max_idx = abs_corrs.argmax()
                sorted_corrs = np.sort(abs_corrs)[::-1]
                if len(sorted_corrs) > 1 and sorted_corrs[0] > 2 * sorted_corrs[1]:
                    monosemantic_features.append({
                        'feature_idx': int(f),
                        'fg': fg_names[max_idx],
                        'correlation': float(correlations[f, max_idx]),
                        'selectivity': float(sorted_corrs[0] / (sorted_corrs[1] + 1e-8))
                    })

        monosemantic_features = sorted(monosemantic_features, key=lambda x: -x['selectivity'])

        return {
            'correlations': correlations,
            'p_values': p_values,
            'top_features_per_fg': top_features_per_fg,
            'monosemantic_features': monosemantic_features,
            'n_samples': len(valid_indices),
            'n_features': n_features,
            'fg_names': fg_names,
        }

    def visualize(self, results: Dict, output_path: Path):
        """Visualize feature-FG correlations."""
        correlations = results['correlations']
        fg_names = results['fg_names']
        n_features = correlations.shape[0]
        n_fgs = correlations.shape[1]

        fig_w = max(16, n_fgs * 0.6)
        fig, axes = plt.subplots(2, 2, figsize=(fig_w, 14))

        # Plot 1: Top correlations heatmap (top 50 features by max correlation)
        ax = axes[0, 0]
        max_corrs = np.abs(correlations).max(axis=1)
        top_feature_idx = np.argsort(max_corrs)[-50:]

        heatmap_data = correlations[top_feature_idx]
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_xlabel('Functional Group')
        ax.set_ylabel('SAE Feature Index')
        ax.set_xticks(range(n_fgs))
        ax.set_xticklabels(fg_names, rotation=45, ha='right', fontsize=max(5, 9 - n_fgs // 10))
        ax.set_title('Top 50 SAE Features by FG Correlation')
        plt.colorbar(im, ax=ax, label='Spearman Correlation')

        # Plot 2: Max correlation distribution
        ax = axes[0, 1]
        ax.hist(max_corrs, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.3, color='red', linestyle='--', label='Significance threshold')
        ax.set_xlabel('Max |Correlation| with any FG')
        ax.set_ylabel('Number of Features')
        ax.set_title('Distribution of Feature-FG Correlations')
        ax.legend()

        # Plot 3: Feature count per FG
        ax = axes[1, 0]
        significant_counts = (np.abs(correlations) > 0.2).sum(axis=0)
        ax.bar(range(n_fgs), significant_counts, color='steelblue', edgecolor='black')
        ax.set_xticks(range(n_fgs))
        ax.set_xticklabels(fg_names, rotation=45, ha='right', fontsize=max(5, 9 - n_fgs // 10))
        ax.set_ylabel('Number of Correlated Features (|r|>0.2)')
        ax.set_title('SAE Features per Functional Group')

        # Plot 4: Monosemantic features
        ax = axes[1, 1]
        mono = results['monosemantic_features']
        if mono:
            fg_labels = [m['fg'] for m in mono[:15]]
            selectivities = [m['selectivity'] for m in mono[:15]]
            correlations_mono = [m['correlation'] for m in mono[:15]]

            colors = ['forestgreen' if c > 0 else 'tomato' for c in correlations_mono]
            y_pos = range(len(fg_labels))
            ax.barh(y_pos, selectivities, color=colors, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"F{m['feature_idx']} -> {m['fg']}" for m in mono[:15]])
            ax.set_xlabel('Selectivity (ratio of top/second correlation)')
            ax.set_title(f'Top {len(mono[:15])} Monosemantic Features')
        else:
            ax.text(0.5, 0.5, 'No monosemantic features found', ha='center', va='center')
            ax.set_title('Monosemantic Features')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


# ── Feature Direction Visualization ──────────────────────────────────────────

class FeatureDirectionVisualizer:
    """Visualize SAE decoder directions in wavenumber space."""

    def __init__(self, sae: BatchTopKSAE, model: SlotUnmixer, device: str = 'cuda'):
        self.sae = sae
        self.model = model
        self.device = device
        self.wavenumbers = get_wavenumbers()
        self.mean_embedding = None

    def set_mean_embedding(self, embeddings: torch.Tensor):
        self.mean_embedding = embeddings.mean(dim=0).to(self.device)

    def get_feature_spectrum(self, feature_idx: int, epsilon: float = 3.0) -> np.ndarray:
        """Get the spectral pattern associated with a feature direction."""
        decoder_directions = self.sae.get_decoder_directions()
        feature_direction = decoder_directions[feature_idx].to(self.device)

        with torch.no_grad():
            if self.mean_embedding is not None:
                s_mean = self.mean_embedding.unsqueeze(0).unsqueeze(0)
                s_pert = (self.mean_embedding + epsilon * feature_direction).unsqueeze(0).unsqueeze(0)
                pred_mean = self.model.prediction(s_mean)
                pred_pert = self.model.prediction(s_pert)
                spectrum = (pred_pert['components'][0, 0] - pred_mean['components'][0, 0]).cpu().numpy()
            else:
                slot_embedding = feature_direction.unsqueeze(0).unsqueeze(0)
                outputs = self.model.prediction(slot_embedding)
                spectrum = outputs['components'][0, 0].cpu().numpy()

        return spectrum

    def visualize_top_features(
        self,
        correlation_results: Dict,
        output_path: Path,
        n_features: int = 12
    ):
        """Visualize spectral patterns for top correlated features."""
        mono_features = correlation_results['monosemantic_features'][:n_features]

        if not mono_features:
            print("  No monosemantic features to visualize")
            return

        n_cols = 3
        n_rows = (len(mono_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for i, feature_info in enumerate(mono_features):
            ax = axes[i]
            feature_idx = feature_info['feature_idx']
            fg_name = feature_info['fg']
            corr = feature_info['correlation']

            spectrum = self.get_feature_spectrum(feature_idx)

            color = 'forestgreen' if corr > 0 else 'tomato'
            ax.plot(self.wavenumbers, spectrum, color=color, linewidth=1)
            ax.fill_between(self.wavenumbers, 0, spectrum, alpha=0.3, color=color)

            ax.set_title(f'Feature {feature_idx} -> {fg_name} (r={corr:.2f})')
            ax.set_ylabel('Intensity')
            ax.set_xlim(self.wavenumbers[-1], self.wavenumbers[0])
            ax.set_ylim(0, 1)

            # Mark known IR bands via FG_TO_BANDS lookup
            if fg_name in FG_TO_BANDS:
                for band_name in FG_TO_BANDS[fg_name]:
                    if band_name in KNOWN_IR_BANDS:
                        low, high = KNOWN_IR_BANDS[band_name]
                        ax.axvspan(low, high, alpha=0.2, color='blue')

        for j in range(len(mono_features), len(axes)):
            axes[j].set_visible(False)

        if len(mono_features) > 0:
            axes[-1].set_xlabel('Wavenumber (cm-1)')

        plt.suptitle('SAE Feature Spectral Patterns (Decoder Directions)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


# ── Feature Activation Analysis ──────────────────────────────────────────────

class FeatureActivationAnalyzer:
    """Analyze when features activate on real spectra."""

    def __init__(self, sae: BatchTopKSAE, model: SlotUnmixer, device: str = 'cuda'):
        self.sae = sae
        self.model = model
        self.device = device
        self.wavenumbers = get_wavenumbers()

    def analyze_feature_activations(
        self,
        dataset: MixtureDataset,
        feature_indices: List[int],
        top_n: int = 50,
    ) -> Dict:
        """Find samples that maximally activate specific features."""
        import heapq

        self.model.eval()
        self.sae.eval()

        # Min-heaps keyed by activation value (keep top_n largest)
        heaps: Dict[int, list] = {f: [] for f in feature_indices}

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )

        sample_offset = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing activations"):
                mixture = batch['mixture'].to(self.device)
                B = mixture.shape[0]

                tokens = self.model.tokenizer(mixture)['tokens']
                memory = self.model.encoder(tokens)
                routing = self.model.routing(memory)
                slots = routing['slots']  # [B, K, d_model]
                K = slots.shape[1]

                slots_flat = slots.reshape(B * K, -1)
                activations = self.sae.get_activations(slots_flat)
                activations = activations.reshape(B, K, -1)  # [B, K, d_latent]

                for b in range(B):
                    sample_idx = sample_offset + b
                    for slot_idx in range(K):
                        for f in feature_indices:
                            act_val = activations[b, slot_idx, f].item()
                            if act_val > 0:
                                entry = {
                                    'sample_idx': sample_idx,
                                    'slot_idx': slot_idx,
                                    'activation': act_val,
                                }
                                if len(heaps[f]) < top_n:
                                    heapq.heappush(heaps[f], (act_val, len(heaps[f]), entry))
                                elif act_val > heaps[f][0][0]:
                                    heapq.heapreplace(heaps[f], (act_val, sample_idx * K + slot_idx, entry))

                sample_offset += B

        # Extract sorted results (descending by activation)
        feature_activations = {}
        for f in feature_indices:
            items = [item for (_, _, item) in heaps[f]]
            feature_activations[f] = sorted(items, key=lambda x: -x['activation'])

        return feature_activations

    def visualize_max_activating_samples(
        self,
        feature_activations: Dict,
        correlation_results: Dict,
        output_path: Path,
        dataset: Optional[MixtureDataset] = None,
        n_samples_per_feature: int = 3
    ):
        """Visualize samples that maximally activate each feature."""
        mono_features = correlation_results['monosemantic_features'][:6]

        if not mono_features:
            return

        fig, axes = plt.subplots(len(mono_features), n_samples_per_feature + 1,
                                  figsize=(4 * (n_samples_per_feature + 1), 3 * len(mono_features)))

        for row, feature_info in enumerate(mono_features):
            feature_idx = feature_info['feature_idx']
            fg_name = feature_info['fg']

            activations = feature_activations.get(feature_idx, [])

            ax = axes[row, 0]
            ax.text(0.5, 0.5, f"Feature {feature_idx}\n-> {fg_name}\nr={feature_info['correlation']:.2f}",
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue'))
            ax.axis('off')

            for col, sample_info in enumerate(activations[:n_samples_per_feature]):
                ax = axes[row, col + 1]
                # Retrieve mixture from dataset if not stored inline
                if 'mixture' in sample_info:
                    mixture = sample_info['mixture']
                elif dataset is not None:
                    mixture = dataset[sample_info['sample_idx']]['mixture'].numpy()
                else:
                    continue
                ax.plot(self.wavenumbers, mixture, 'k-', linewidth=0.8)
                ax.set_title(f"Act={sample_info['activation']:.2f}")
                ax.set_xlim(self.wavenumbers[-1], self.wavenumbers[0])
                ax.set_ylim(0, 1)
                if row == len(mono_features) - 1:
                    ax.set_xlabel('Wavenumber (cm-1)')

        plt.suptitle('Max-Activating Samples per SAE Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


# ── Spectral Attribution Analysis ─────────────────────────────────────────────

class FeatureSpectralAttributionAnalyzer:
    """Gradient-based spectral attribution for SAE features."""

    def __init__(self, sae: BatchTopKSAE, model: SlotUnmixer, device: str = 'cuda'):
        self.sae = sae
        self.model = model
        self.device = device
        self.wavenumbers = get_wavenumbers()

    def compute_feature_gradient(self, mixture: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Compute gradient of SAE feature activation w.r.t. input spectrum."""
        mixture_input = mixture.clone().detach().requires_grad_(True)
        outputs = self.model(mixture=mixture_input.unsqueeze(0), stage=2)
        slots = outputs['slots']  # [1, K, d_model]

        # Encode all slots through SAE encoder (pre-activation, before sparsification)
        z_pre = self.sae.encode(slots[0])  # [K, d_latent]

        # Target: max pre-activation across slots for this feature
        target = z_pre[:, feature_idx].max()
        target.backward()

        return mixture_input.grad.detach()  # [spectrum_dim]

    def compute_feature_attribution(
        self,
        dataset: MixtureDataset,
        feature_idx: int,
        sample_indices: List[Dict],
        n_samples: int = 50,
    ) -> np.ndarray:
        """Average gradient-based saliency for a feature across its top-activating samples."""
        self.model.eval()
        saliencies = []

        for info in sample_indices[:n_samples]:
            sample = dataset[info['sample_idx']]
            mixture = sample['mixture'].to(self.device)
            grad = self.compute_feature_gradient(mixture, feature_idx)
            saliencies.append(grad.abs().cpu().numpy())

        return np.mean(saliencies, axis=0)  # [spectrum_dim]

    def analyze(
        self,
        dataset: MixtureDataset,
        correlation_results: Dict,
        feature_activations: Dict,
        n_features: int = 12,
        n_samples: int = 50,
    ) -> Tuple[Dict, Dict]:
        """Full spectral attribution analysis."""
        mono_features = correlation_results['monosemantic_features'][:n_features]
        per_feature = {}

        for feat in tqdm(mono_features, desc="Computing spectral attributions"):
            fidx = feat['feature_idx']
            activations = feature_activations.get(fidx, [])
            if not activations:
                continue
            saliency = self.compute_feature_attribution(dataset, fidx, activations, n_samples)
            per_feature[fidx] = {
                'saliency': saliency,
                'fg': feat['fg'],
                'correlation': feat['correlation'],
            }

        # Aggregate per FG
        per_fg = defaultdict(list)
        for fidx, data in per_feature.items():
            per_fg[data['fg']].append(data['saliency'])

        per_fg_mean = {}
        for fg_name, saliency_list in per_fg.items():
            per_fg_mean[fg_name] = np.mean(saliency_list, axis=0)

        return per_feature, per_fg_mean

    def visualize_per_feature(self, per_feature: Dict, output_path: Path, n_cols: int = 3):
        """Grid of per-feature saliency maps with known IR band overlays."""
        if not per_feature:
            print("  No feature attributions to visualize")
            return

        n_features = len(per_feature)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (fidx, data) in enumerate(per_feature.items()):
            ax = axes[i]
            saliency = data['saliency']
            fg_name = data['fg']
            corr = data['correlation']

            color = 'forestgreen' if corr > 0 else 'tomato'
            ax.plot(self.wavenumbers, saliency, color=color, linewidth=1)
            ax.fill_between(self.wavenumbers, 0, saliency, alpha=0.3, color=color)

            ax.set_title(f'Feature {fidx} -> {fg_name} (r={corr:.2f})')
            ax.set_ylabel('|grad|')
            ax.set_xlim(self.wavenumbers[-1], self.wavenumbers[0])

            # Overlay known IR bands
            if fg_name in FG_TO_BANDS:
                for band_name in FG_TO_BANDS[fg_name]:
                    if band_name in KNOWN_IR_BANDS:
                        low, high = KNOWN_IR_BANDS[band_name]
                        ax.axvspan(low, high, alpha=0.15, color='blue', label=band_name)

        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)

        if n_features > 0:
            for ax in axes[max(0, n_features - n_cols):n_features]:
                ax.set_xlabel('Wavenumber (cm$^{-1}$)')

        plt.suptitle('SAE Feature Spectral Attribution (Input Gradients)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_per_fg(self, per_fg: Dict, output_path: Path):
        """Per-FG aggregated spectral attribution with IR band validation."""
        if not per_fg:
            print("  No per-FG attributions to visualize")
            return

        n_fgs = len(per_fg)
        n_cols = min(3, n_fgs)
        n_rows = (n_fgs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for i, (fg_name, saliency) in enumerate(per_fg.items()):
            ax = axes[i]
            ax.plot(self.wavenumbers, saliency, color='steelblue', linewidth=1.2)
            ax.fill_between(self.wavenumbers, 0, saliency, alpha=0.3, color='steelblue')

            ax.set_title(f'{fg_name}')
            ax.set_ylabel('Mean |grad|')
            ax.set_xlim(self.wavenumbers[-1], self.wavenumbers[0])

            # Overlay expected IR bands for validation
            if fg_name in FG_TO_BANDS:
                for band_name in FG_TO_BANDS[fg_name]:
                    if band_name in KNOWN_IR_BANDS:
                        low, high = KNOWN_IR_BANDS[band_name]
                        ax.axvspan(low, high, alpha=0.2, color='orange', label=band_name)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7, loc='upper right')

        for j in range(n_fgs, len(axes)):
            axes[j].set_visible(False)

        if n_fgs > 0:
            for ax in axes[max(0, n_fgs - n_cols):n_fgs]:
                ax.set_xlabel('Wavenumber (cm$^{-1}$)')

        plt.suptitle('Per-FG Spectral Attribution (Aggregated)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


# ── Precision-Recall Validation ─────────────────────────────────────────

class PrecisionRecallValidator:
    """Precision-Recall feature validation for SAE latents vs functional groups."""

    def __init__(self, sae: BatchTopKSAE, model: SlotUnmixer, device: str = 'cuda'):
        self.sae = sae
        self.model = model
        self.device = torch.device(device)

    def collect_activations_with_labels(
        self,
        dataloader,
        fg_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect SAE latent activations and corresponding FG labels."""
        self.model.eval()
        self.sae.eval()

        z_list, fg_list = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting SAE activations + labels"):
                mixture = batch["mixture"].to(self.device)
                comp_indices = batch["indices"]  # [B, K]
                true_components = batch["components"].to(self.device)  # [B, K, spectrum_dim]

                tokens = self.model.tokenizer(mixture)["tokens"]
                memory = self.model.encoder(tokens)
                routing = self.model.routing(memory)
                slots = routing["slots"]  # [B, K, d]
                B, K, d = slots.shape

                # Hungarian matching: align slots to ground-truth components
                pred = self.model.prediction(slots)
                matched = match_slots_l2(pred["components"], true_components)
                aligned_indices = comp_indices.clone()
                for b in range(B):
                    aligned_indices[b] = comp_indices[b, matched[b]]

                slots_flat = slots.reshape(B * K, d)
                z = self.sae.get_activations(slots_flat)
                z_list.append(z.cpu().numpy())

                if fg_matrix is not None:
                    idx_flat = aligned_indices.reshape(-1).numpy()
                    fg_expanded = fg_matrix[idx_flat]
                    fg_list.append(fg_expanded)

        if not z_list:
            raise RuntimeError("No activations collected.")

        z_all = np.concatenate(z_list, axis=0)
        if fg_list:
            fg_all = np.concatenate(fg_list, axis=0)
        else:
            raise RuntimeError("No FG labels collected. Provide fg_matrix.")

        return z_all, fg_all

    @staticmethod
    def compute(
        z: np.ndarray,
        fg: np.ndarray,
        fg_names: List[str],
        thresholds_percentile: List[float] = (50, 75, 90, 95),
        min_support: int = 20,
    ) -> Dict:
        """
        Compute precision, recall, F1, and false-positive rate for all
        (latent k, functional group phi) pairs.
        """
        N, d_latent = z.shape
        _, n_fg = fg.shape

        results_by_threshold = {}

        for pct in thresholds_percentile:
            thresholds = np.zeros(d_latent)
            for k in range(d_latent):
                nonzero = z[:, k][z[:, k] > 0]
                if len(nonzero) > 0:
                    thresholds[k] = np.percentile(nonzero, pct)
                else:
                    thresholds[k] = float("inf")

            active = z > thresholds[np.newaxis, :]

            precision_matrix = np.full((d_latent, n_fg), np.nan)
            recall_matrix = np.full((d_latent, n_fg), np.nan)
            f1_matrix = np.full((d_latent, n_fg), np.nan)
            fpr_matrix = np.full((d_latent, n_fg), np.nan)

            for j in range(n_fg):
                fg_positive = fg[:, j] > 0.5
                fg_negative = ~fg_positive
                n_pos = fg_positive.sum()

                if n_pos < min_support:
                    continue

                for k in range(d_latent):
                    k_active = active[:, k]
                    n_active = k_active.sum()
                    if n_active == 0:
                        continue

                    tp = (k_active & fg_positive).sum()
                    prec = tp / n_active if n_active > 0 else 0.0
                    rec = tp / n_pos if n_pos > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    fpr = (k_active & fg_negative).sum() / fg_negative.sum() if fg_negative.sum() > 0 else 0.0

                    precision_matrix[k, j] = prec
                    recall_matrix[k, j] = rec
                    f1_matrix[k, j] = f1
                    fpr_matrix[k, j] = fpr

            valid_precision = np.nansum(precision_matrix >= 0.8)
            valid_recall = np.nansum(recall_matrix >= 0.8)
            valid_f1 = np.nansum(f1_matrix >= 0.5)

            results_by_threshold[pct] = {
                "precision": precision_matrix,
                "recall": recall_matrix,
                "f1": f1_matrix,
                "fpr": fpr_matrix,
                "thresholds": thresholds,
                "n_validated_precision_08": int(valid_precision),
                "n_validated_recall_08": int(valid_recall),
                "n_validated_f1_05": int(valid_f1),
            }

            print(f"  tau = {pct}th percentile: "
                  f"precision>=0.8: {valid_precision}, "
                  f"recall>=0.8: {valid_recall}, "
                  f"F1>=0.5: {valid_f1}")

        # Per-FG summary: sweep all thresholds, pick the one that maximizes F1
        pct_keys = [k for k in results_by_threshold if isinstance(k, (int, float))]
        if pct_keys:
            per_fg_summary = []
            for j, name in enumerate(fg_names):
                best_f1_overall = 0.0
                best_entry = {
                    "fg_name": name,
                    "best_precision": 0.0,
                    "best_recall": 0.0,
                    "best_f1": 0.0,
                    "best_latent": -1,
                    "best_threshold_pct": -1,
                    "n_associated_latents": 0,
                }
                for pct in pct_keys:
                    data = results_by_threshold[pct]
                    f1_col = data["f1"][:, j]
                    if np.all(np.isnan(f1_col)):
                        continue
                    k = int(np.nanargmax(f1_col))
                    f1_val = float(f1_col[k])
                    if f1_val > best_f1_overall:
                        best_f1_overall = f1_val
                        best_entry = {
                            "fg_name": name,
                            "best_precision": float(data["precision"][k, j]),
                            "best_recall": float(data["recall"][k, j]),
                            "best_f1": f1_val,
                            "best_latent": int(k),
                            "best_threshold_pct": int(pct),
                            "n_associated_latents": int(np.nansum(f1_col >= 0.5)),
                        }
                per_fg_summary.append(best_entry)
            results_by_threshold["per_fg_summary"] = per_fg_summary

        return results_by_threshold

    @staticmethod
    def visualize(results: Dict, fg_names: List[str], output_dir: Path):
        """Plot cumulative distributions and per-FG F1 bar chart."""
        # Plot 1: Cumulative distribution of precision/recall/F1
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for pct in (50, 75, 90, 95):
            if pct not in results:
                continue
            data = results[pct]
            for ax, metric, label in zip(
                axes,
                ("precision", "recall", "f1"),
                ("Precision", "Recall", "F1"),
            ):
                vals = data[metric].flatten()
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                sorted_v = np.sort(vals)[::-1]
                ax.plot(
                    np.arange(len(sorted_v)),
                    sorted_v,
                    label=f"tau = {pct}th pct",
                    linewidth=1.5,
                )
                ax.set_xlabel("Latent-FG pair index (sorted)", fontsize=10)
                ax.set_ylabel(label, fontsize=10)
                ax.set_title(f"Cumulative {label}", fontsize=12, fontweight="bold")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / "precision_recall_cumulative.pdf", dpi=300)
        fig.savefig(output_dir / "precision_recall_cumulative.png", dpi=150)
        plt.close(fig)

        # Plot 2: Per-FG best F1 (at 90th percentile)
        if "per_fg_summary" in results:
            summary = results["per_fg_summary"]
            fig, ax = plt.subplots(figsize=(12, 4))
            names = [s["fg_name"] for s in summary]
            f1_vals = [s.get("best_f1", s.get("max_f1", 0.0)) for s in summary]
            n_assoc = [s["n_associated_latents"] for s in summary]

            x = np.arange(len(names))
            bars = ax.bar(x, f1_vals, color="#2E86AB", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Best F1 Score", fontsize=11)
            ax.set_title("Best Latent-FG Match (F1) per Functional Group (tau = 90th pct)", fontsize=13, fontweight="bold")
            ax.axhline(y=0.5, color="#C73E1D", linestyle="--", alpha=0.6, label="F1 = 0.5 threshold")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            for i, (bar, n) in enumerate(zip(bars, n_assoc)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        str(n), ha="center", va="bottom", fontsize=8, color="#555")

            plt.tight_layout()
            fig.savefig(output_dir / "precision_recall_per_fg_f1.pdf", dpi=300)
            fig.savefig(output_dir / "precision_recall_per_fg_f1.png", dpi=150)
            plt.close(fig)


# ── Main Analysis Pipeline ───────────────────────────────────────────────────

def run_analysis(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = 'cuda',
    dataset_file: Optional[str] = None,
    use_full_dataset: bool = False,
    sae_latent_dim: int = 1024,
    sae_sparsity: float = 0.05,
    sae_k_fixed: Optional[int] = None,
    sae_epochs: int = 50,
    pretrained_sae: Optional[str] = None,
    precision_recall: bool = False,
    n_fgs: Optional[int] = None,
    spectral_attribution: bool = False,
    spectral_localization: bool = False,
    permutation_test: bool = False,
) -> Dict:
    """Run SAE feature analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    with open(config_path) as f:
        config_dict = json.load(f)
    cfg = load_config(**config_dict)

    model = SlotUnmixer(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load dataset
    print("Loading dataset...")
    mixture_dataset_file = dataset_file if dataset_file else cfg.data.mixture_dataset
    print(f"  Dataset: {mixture_dataset_file}")
    mixture_data = _load_mixture_dataset(cfg.data, mixture_dataset_file)
    n_samples = mixture_data['mixture'].shape[0]

    if use_full_dataset:
        train_indices = list(range(n_samples))
        test_indices = list(range(n_samples))
        print(f"  Using full dataset: {len(train_indices)} samples (train=test=all)")
    else:
        train_indices, _, test_indices = split_indices(n_samples, seed=cfg.seed)
        print(f"  Train split: {len(train_indices)} samples")
        print(f"  Test split: {len(test_indices)} samples")

    train_dataset = MixtureDataset(cfg.data, dataset_file=mixture_dataset_file, split_indices=train_indices)
    test_dataset = MixtureDataset(cfg.data, dataset_file=mixture_dataset_file, split_indices=test_indices)

    # Load FG names (auto-detect count from data if n_fgs not specified)
    fg_names = load_fg_names(n_fgs)
    # Clamp to actual data dimension
    fg_data = test_dataset.molecule_db.get('functional_groups', None)
    if fg_data is not None:
        actual_n_fgs = fg_data.shape[1] if fg_data.ndim == 2 else 0
        if len(fg_names) > actual_n_fgs:
            fg_names = fg_names[:actual_n_fgs]
    print(f"  Analyzing {len(fg_names)} functional groups")

    results = {}
    results['dataset'] = mixture_dataset_file
    results['n_train_samples'] = len(train_indices)
    results['n_test_samples'] = len(test_indices)
    results['fg_names'] = fg_names

    # Initialize SAE trainer
    trainer = SAETrainer(
        d_input=cfg.model.d_model,
        d_latent=sae_latent_dim,
        sparsity_percent=sae_sparsity,
        k_fixed=sae_k_fixed,
        device=device
    )

    # Collect embeddings
    print("\n" + "=" * 60)
    print("Collecting Slot Embeddings")
    print("=" * 60)
    if not pretrained_sae:
        train_embeddings, _ = trainer.collect_embeddings(model, train_dataset)
        print(f"  Collected {len(train_embeddings)} train embeddings")
    test_embeddings, test_metadata = trainer.collect_embeddings(model, test_dataset)
    print(f"  Collected {len(test_embeddings)} test embeddings")

    # Train or load SAE
    if pretrained_sae:
        print("\n" + "=" * 60)
        print(f"Loading Pretrained SAE from {pretrained_sae}")
        print("=" * 60)
        trainer.sae.load_state_dict(torch.load(pretrained_sae, map_location=device, weights_only=True))
        trainer.sae.eval()
        results['sae_training'] = {
            'pretrained': True,
            'pretrained_path': pretrained_sae,
            'n_test_embeddings': len(test_embeddings),
        }
        print(f"  Loaded pretrained SAE (analyze-only mode)")
    else:
        print("\n" + "=" * 60)
        print("Training SAE on Slot Embeddings")
        print("=" * 60)

        history = trainer.train(train_embeddings, n_epochs=sae_epochs)
        print(f"  Final reconstruction loss: {history['loss'][-1]:.6f}")
        print(f"  Final L0 sparsity: {history['L0'][-1]:.1f}")

        results['sae_training'] = {
            'pretrained': False,
            'n_train_embeddings': len(train_embeddings),
            'n_test_embeddings': len(test_embeddings),
            'd_latent': sae_latent_dim,
            'sparsity_percent': sae_sparsity,
            'final_loss': history['loss'][-1],
            'final_L0': history['L0'][-1],
        }

        sae_path = output_dir / 'trained_sae.pt'
        torch.save(trainer.sae.state_dict(), sae_path)
        print(f"  Saved SAE to {sae_path}")

        del train_embeddings

    # Feature-FG Correlation
    print("\n" + "=" * 60)
    print("Feature-FG Correlation Analysis")
    print("=" * 60)

    correlation_analyzer = FeatureFGCorrelationAnalyzer(trainer.sae, device)
    correlation_results = correlation_analyzer.analyze(test_embeddings, test_metadata, fg_names)

    if 'error' not in correlation_results:
        correlation_analyzer.visualize(correlation_results, output_dir / 'feature_fg_correlations.png')
        print(f"  Saved: feature_fg_correlations.png")

        print(f"\n  Top monosemantic features:")
        for feat in correlation_results['monosemantic_features'][:5]:
            print(f"    Feature {feat['feature_idx']} -> {feat['fg']}: r={feat['correlation']:.3f}, selectivity={feat['selectivity']:.1f}")

        results['feature_fg_correlation'] = {
            'n_monosemantic': len(correlation_results['monosemantic_features']),
            'top_features': correlation_results['monosemantic_features'],
            'top_features_per_fg': {k: v[:3] for k, v in correlation_results['top_features_per_fg'].items()},
        }
    else:
        print(f"  Skipped: {correlation_results['error']}")
        results['feature_fg_correlation'] = correlation_results

    # Feature Direction Visualization
    print("\n" + "=" * 60)
    print("Feature Direction Visualization")
    print("=" * 60)

    if 'error' not in correlation_results and correlation_results['monosemantic_features']:
        direction_viz = FeatureDirectionVisualizer(trainer.sae, model, device)
        direction_viz.set_mean_embedding(test_embeddings)
        direction_viz.visualize_top_features(
            correlation_results,
            output_dir / 'feature_spectral_patterns.png'
        )
        print(f"  Saved: feature_spectral_patterns.png")
    else:
        print("  Skipped: No monosemantic features found")

    # Max-Activating Samples
    print("\n" + "=" * 60)
    print("Max-Activating Sample Analysis")
    print("=" * 60)

    feature_activations = {}
    if 'error' not in correlation_results and correlation_results['monosemantic_features']:
        activation_analyzer = FeatureActivationAnalyzer(trainer.sae, model, device)

        # Collect more features if spectral attribution will need them
        n_act_features = 12 if spectral_attribution else 6
        top_feature_indices = [f['feature_idx'] for f in correlation_results['monosemantic_features'][:n_act_features]]
        feature_activations = activation_analyzer.analyze_feature_activations(
            test_dataset, top_feature_indices,
        )

        activation_analyzer.visualize_max_activating_samples(
            feature_activations,
            correlation_results,
            output_dir / 'max_activating_samples.png',
            dataset=test_dataset,
        )
        print(f"  Saved: max_activating_samples.png")

        results['feature_activations'] = {
            'features_analyzed': top_feature_indices,
            'samples_per_feature': {f: len(feature_activations[f]) for f in top_feature_indices}
        }
    else:
        print("  Skipped: No features to analyze")

    # Spectral Attribution Analysis (optional)
    if spectral_attribution and 'error' not in correlation_results and correlation_results['monosemantic_features']:
        print("\n" + "=" * 60)
        print("Spectral Attribution Analysis")
        print("=" * 60)

        attr_analyzer = FeatureSpectralAttributionAnalyzer(trainer.sae, model, device)
        per_feature_attr, per_fg_attr = attr_analyzer.analyze(
            test_dataset, correlation_results, feature_activations,
            n_features=12, n_samples=50,
        )

        attr_analyzer.visualize_per_feature(
            per_feature_attr, output_dir / 'feature_spectral_attributions.png'
        )
        print(f"  Saved: feature_spectral_attributions.png")

        attr_analyzer.visualize_per_fg(
            per_fg_attr, output_dir / 'fg_spectral_regions.png'
        )
        print(f"  Saved: fg_spectral_regions.png")

        # Save saliency data
        attr_save = {}
        for fidx, data in per_feature_attr.items():
            attr_save[f'feature_{fidx}_saliency'] = data['saliency']
        for fg_name, saliency in per_fg_attr.items():
            attr_save[f'fg_{fg_name}_saliency'] = saliency
        np.savez(output_dir / 'spectral_attributions.npz', **attr_save)

        results['spectral_attribution'] = {
            'n_features_analyzed': len(per_feature_attr),
            'n_fgs_analyzed': len(per_fg_attr),
            'fgs': list(per_fg_attr.keys()),
        }

    # Precision-Recall Validation (optional)
    if precision_recall:
        print("\n" + "=" * 60)
        print("Precision-Recall Feature Validation")
        print("=" * 60)

        # Build dataloader for batch-wise processing with Hungarian matching
        pr_dataloader = build_dataloader(test_dataset, cfg, shuffle=False)

        # Extract FG matrix from molecule database
        fg_matrix = None
        if hasattr(test_dataset, 'molecule_db'):
            fg_data = test_dataset.molecule_db.get('functional_groups')
            if fg_data is not None:
                fg_matrix = fg_data.numpy() if isinstance(fg_data, torch.Tensor) else fg_data

        if fg_matrix is None:
            print("  Skipped: No FG matrix available in molecule database")
        else:
            validator = PrecisionRecallValidator(trainer.sae, model, device)
            z, fg = validator.collect_activations_with_labels(pr_dataloader, fg_matrix=fg_matrix)
            print(f"  Collected {z.shape[0]} samples, {z.shape[1]} latents, {fg.shape[1]} FGs")

            pr_results = PrecisionRecallValidator.compute(z, fg, fg_names)
            PrecisionRecallValidator.visualize(pr_results, fg_names, output_dir)

            # Save numeric results
            save_data = {}
            for pct, data in pr_results.items():
                if isinstance(pct, (int, float)):
                    for key in ("precision", "recall", "f1", "fpr", "thresholds"):
                        save_data[f"pct{pct}_{key}"] = data[key]
            np.savez(output_dir / "precision_recall.npz", **save_data)

            if "per_fg_summary" in pr_results:
                with open(output_dir / "precision_recall_per_fg.json", "w") as f:
                    json.dump(pr_results["per_fg_summary"], f, indent=2)

            results['precision_recall'] = {
                'n_samples': z.shape[0],
                'per_fg_summary': pr_results.get('per_fg_summary', []),
            }

    # Spectral Localization Index (optional)
    if spectral_localization and 'error' not in correlation_results and correlation_results['monosemantic_features']:
        print("\n" + "=" * 60)
        print("Spectral Localization Index (SLI)")
        print("=" * 60)

        sli_analyzer = SpectralLocalizationAnalyzer(trainer.sae, model, device)
        sli_analyzer.set_mean_embedding(test_embeddings)
        sli_results = sli_analyzer.analyze(correlation_results)
        SpectralLocalizationAnalyzer.visualize(sli_results, output_dir)
        print(f"  Saved: spectral_localization.png/pdf")

        # Summary
        per_feat = sli_results['per_feature']
        if per_feat:
            mean_sli = np.mean([e['sli'] for e in per_feat])
            mean_baseline = np.mean([e['baseline_sli'] for e in per_feat])
            errors = [e['peak_error_cm1'] for e in per_feat if not np.isnan(e['peak_error_cm1'])]
            print(f"  Mean SLI: {mean_sli:.3f} (baseline: {mean_baseline:.3f}, ratio: {mean_sli/(mean_baseline+1e-6):.1f}×)")
            if errors:
                print(f"  Median peak error: {np.median(errors):.0f} cm⁻¹")

        results['spectral_localization'] = {
            'per_feature': sli_results['per_feature'],
            'per_fg': sli_results['per_fg'],
            'random_baseline_sli': sli_results['random_baseline_sli'],
        }

        with open(output_dir / 'spectral_localization.json', 'w') as f:
            json.dump(results['spectral_localization'], f, indent=2)

    # Permutation Test (optional)
    if permutation_test and 'error' not in correlation_results:
        print("\n" + "=" * 60)
        print("Permutation Test (Null Model)")
        print("=" * 60)

        perm_analyzer = PermutationTestAnalyzer(trainer.sae, device)
        perm_results = perm_analyzer.analyze(test_embeddings, test_metadata, fg_names)
        PermutationTestAnalyzer.visualize(perm_results, output_dir)
        print(f"  Saved: permutation_test.png/pdf")
        print(f"  Real: {perm_results['real_monosemantic_count']}, "
              f"Null: {perm_results['null_mean']:.1f}±{perm_results['null_std']:.1f}, "
              f"p={perm_results['p_value']:.4f}")

        results['permutation_test'] = perm_results

        with open(output_dir / 'permutation_test.json', 'w') as f:
            json.dump(perm_results, f, indent=2)

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


class SpectralLocalizationAnalyzer:
    """Quantify how well decoder directions align with known IR absorption bands."""

    def __init__(self, sae: BatchTopKSAE, model: SlotUnmixer, device: str = 'cuda',
                 mean_embedding: Optional[torch.Tensor] = None):
        self.sae = sae
        self.model = model
        self.device = device
        self.wavenumbers = get_wavenumbers()
        self.mean_embedding = mean_embedding  # [d_model], set via set_mean_embedding

    def set_mean_embedding(self, embeddings: torch.Tensor):
        """Compute and store mean slot embedding for perturbation-based analysis."""
        self.mean_embedding = embeddings.mean(dim=0).to(self.device)

    def _find_best_peak(self, spectrum: np.ndarray, wn: np.ndarray,
                        fg_name: str) -> float:
        """Find the most chemically-relevant peak in a decoded spectrum."""
        abs_spec = np.abs(spectrum)
        # Find peaks with prominence
        peak_indices, properties = find_peaks(abs_spec, prominence=abs_spec.max() * 0.05)

        if len(peak_indices) == 0:
            # Fallback to global argmax
            return float(wn[np.argmax(abs_spec)])

        prominences = properties['prominences']

        # Check if FG has expected bands
        bands = FG_TO_BANDS.get(fg_name, [])
        band_ranges = []
        tolerance = 100  # cm-1
        for band_name in bands:
            if band_name in KNOWN_IR_BANDS:
                lo, hi = KNOWN_IR_BANDS[band_name]
                band_ranges.append((lo - tolerance, hi + tolerance))

        if band_ranges:
            # Find peaks within expected bands
            in_band_mask = np.zeros(len(peak_indices), dtype=bool)
            for lo, hi in band_ranges:
                for i, pi in enumerate(peak_indices):
                    if lo <= wn[pi] <= hi:
                        in_band_mask[i] = True
            if in_band_mask.any():
                # Pick the most prominent peak within expected bands
                in_band_proms = prominences.copy()
                in_band_proms[~in_band_mask] = -1
                best_idx = peak_indices[np.argmax(in_band_proms)]
                return float(wn[best_idx])

        # No expected bands or no peaks in band: use most prominent peak,
        # but penalize C-H region (2850-3000 cm-1) by halving its prominence
        adjusted_proms = prominences.copy()
        for i, pi in enumerate(peak_indices):
            if 2850 <= wn[pi] <= 3000:
                adjusted_proms[i] *= 0.5
        best_idx = peak_indices[np.argmax(adjusted_proms)]
        return float(wn[best_idx])

    def _decode_feature_to_spectrum(self, feature_idx: int, epsilon: float = 3.0) -> np.ndarray:
        """Project SAE decoder direction through prediction head using perturbation from mean."""
        decoder_directions = self.sae.get_decoder_directions()
        direction = decoder_directions[feature_idx].to(self.device)

        with torch.no_grad():
            if self.mean_embedding is not None:
                s_mean = self.mean_embedding.unsqueeze(0).unsqueeze(0)
                s_perturbed = (self.mean_embedding + epsilon * direction).unsqueeze(0).unsqueeze(0)
                pred_mean = self.model.prediction(s_mean)
                pred_pert = self.model.prediction(s_perturbed)
                spectrum = (pred_pert['components'][0, 0] - pred_mean['components'][0, 0]).cpu().numpy()
            else:
                # Fallback: raw direction (less accurate)
                slot = direction.unsqueeze(0).unsqueeze(0)
                pred = self.model.prediction(slot)
                spectrum = pred['components'][0, 0].cpu().numpy()
        return spectrum

    def analyze(self, correlation_results: Dict, n_random: int = 100) -> Dict:
        """Compute SLI and peak accuracy for monosemantic features."""
        mono_features = correlation_results['monosemantic_features']
        wn = self.wavenumbers
        total_range = wn[-1] - wn[0]

        results_per_feature = []
        results_per_fg = defaultdict(list)

        for feat in mono_features:
            fidx = feat['feature_idx']
            fg_name = feat['fg']

            spectrum = self._decode_feature_to_spectrum(fidx)
            energy = spectrum ** 2
            total_energy = energy.sum()
            if total_energy < 1e-12:
                continue

            # Find prominent peaks, preferring those near expected bands
            peak_wn = self._find_best_peak(spectrum, wn, fg_name)

            # Compute SLI for this feature's assigned FG
            bands = FG_TO_BANDS.get(fg_name, [])
            band_energy = 0.0
            band_centers = []
            band_widths = 0.0
            for band_name in bands:
                if band_name in KNOWN_IR_BANDS:
                    lo, hi = KNOWN_IR_BANDS[band_name]
                    mask = (wn >= lo) & (wn <= hi)
                    band_energy += energy[mask].sum()
                    band_centers.append((lo + hi) / 2)
                    band_widths += (hi - lo)

            sli = float(band_energy / total_energy) if total_energy > 0 else 0.0
            baseline_sli = band_widths / total_range if band_widths > 0 else 0.0

            # Peak error: distance to nearest band center
            if band_centers:
                peak_error = min(abs(peak_wn - c) for c in band_centers)
            else:
                peak_error = float('nan')

            entry = {
                'feature_idx': fidx,
                'fg': fg_name,
                'correlation': feat['correlation'],
                'sli': sli,
                'baseline_sli': baseline_sli,
                'sli_ratio': sli / baseline_sli if baseline_sli > 0 else float('nan'),
                'peak_wn': float(peak_wn),
                'peak_error_cm1': float(peak_error),
            }
            results_per_feature.append(entry)
            results_per_fg[fg_name].append(entry)

        # Random baseline: SLI for random decoder directions
        n_latents = self.sae.get_decoder_directions().shape[0]
        random_slis = []
        rng = np.random.RandomState(42)
        rand_indices = rng.choice(n_latents, size=min(n_random, n_latents), replace=False)
        for ridx in rand_indices:
            spectrum = self._decode_feature_to_spectrum(int(ridx))
            energy = spectrum ** 2
            total_energy = energy.sum()
            if total_energy < 1e-12:
                continue
            # Compute mean SLI across all FG bands
            fg_slis = []
            for fg_name, band_names in FG_TO_BANDS.items():
                band_energy = 0.0
                for bn in band_names:
                    if bn in KNOWN_IR_BANDS:
                        lo, hi = KNOWN_IR_BANDS[bn]
                        mask = (wn >= lo) & (wn <= hi)
                        band_energy += energy[mask].sum()
                fg_slis.append(float(band_energy / total_energy))
            random_slis.append(np.mean(fg_slis))

        # Per-FG summary
        per_fg_summary = {}
        for fg_name, entries in results_per_fg.items():
            slis = [e['sli'] for e in entries]
            errors = [e['peak_error_cm1'] for e in entries if not np.isnan(e['peak_error_cm1'])]
            per_fg_summary[fg_name] = {
                'mean_sli': float(np.mean(slis)),
                'mean_peak_error_cm1': float(np.mean(errors)) if errors else float('nan'),
                'n_features': len(entries),
                'baseline_sli': entries[0]['baseline_sli'] if entries else 0.0,
            }

        return {
            'per_feature': results_per_feature,
            'per_fg': per_fg_summary,
            'random_baseline_sli': float(np.mean(random_slis)) if random_slis else 0.0,
            'random_baseline_std': float(np.std(random_slis)) if random_slis else 0.0,
        }

    @staticmethod
    def visualize(results: Dict, output_dir: Path):
        """Plot SLI bar chart and peak error distribution."""
        per_feature = results['per_feature']
        if not per_feature:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: SLI per feature vs baseline
        ax = axes[0]
        labels = [f"F{e['feature_idx']}\n→{e['fg']}" for e in per_feature[:20]]
        slis = [e['sli'] for e in per_feature[:20]]
        baselines = [e['baseline_sli'] for e in per_feature[:20]]
        x = np.arange(len(labels))
        ax.bar(x - 0.15, slis, 0.3, label='Learned SLI', color='#2E86AB')
        ax.bar(x + 0.15, baselines, 0.3, label='Chance SLI', color='#CCCCCC')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('Spectral Localization Index')
        ax.set_title('SLI: Energy in Expected IR Band')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: SLI ratio (learned / chance)
        ax = axes[1]
        ratios = [e['sli_ratio'] for e in per_feature[:20] if not np.isnan(e['sli_ratio'])]
        ratio_labels = [f"F{e['feature_idx']}" for e in per_feature[:20] if not np.isnan(e['sli_ratio'])]
        colors = ['#2E86AB' if r > 1 else '#C73E1D' for r in ratios]
        ax.bar(range(len(ratios)), ratios, color=colors)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(ratio_labels, rotation=45, fontsize=8)
        ax.set_ylabel('SLI / Chance SLI')
        ax.set_title('Spectral Localization Enrichment')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Peak error histogram
        ax = axes[2]
        errors = [e['peak_error_cm1'] for e in per_feature if not np.isnan(e['peak_error_cm1'])]
        if errors:
            ax.hist(errors, bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
            ax.axvline(x=np.median(errors), color='red', linestyle='--',
                      label=f'Median: {np.median(errors):.0f} cm⁻¹')
            ax.set_xlabel('Peak Error (cm⁻¹)')
            ax.set_ylabel('Count')
            ax.set_title('Distance from Peak to Nearest Expected Band')
            ax.legend()

        plt.tight_layout()
        fig.savefig(output_dir / 'spectral_localization.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'spectral_localization.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)


# ── Causal Ablation Analysis ─────────────────────────────────────────────────

class PermutationTestAnalyzer:
    """Permutation test: shuffle FG labels and re-compute correlations to establish null distribution."""

    def __init__(self, sae: BatchTopKSAE, device: str = 'cuda'):
        self.sae = sae
        self.device = device

    def analyze(
        self,
        embeddings: torch.Tensor,
        metadata: List[Dict],
        fg_names: List[str],
        n_permutations: int = 500,
        mono_threshold: float = 0.3,
        selectivity_threshold: float = 2.0,
    ) -> Dict:
        """Run permutation test."""
        n_fgs = len(fg_names)

        with torch.no_grad():
            activations = self.sae.get_activations(embeddings.to(self.device))
            activations = activations.cpu().numpy()

        # Build FG matrix
        fg_matrix = []
        valid_indices = []
        for i, meta in enumerate(metadata):
            if meta['component_fgs'] is not None:
                row = meta['component_fgs']
                padded = np.zeros(n_fgs)
                padded[:min(len(row), n_fgs)] = row[:n_fgs]
                fg_matrix.append(padded)
                valid_indices.append(i)

        if not valid_indices:
            return {'error': 'No FG labels'}

        fg_matrix = np.array(fg_matrix)
        act_valid = activations[valid_indices]
        n_features = act_valid.shape[1]

        # Real monosemantic count
        real_count = self._count_monosemantic(act_valid, fg_matrix, mono_threshold, selectivity_threshold)

        # Permutation null
        rng = np.random.RandomState(42)
        null_counts = []
        for _ in tqdm(range(n_permutations), desc="Permutation test"):
            perm = rng.permutation(len(fg_matrix))
            fg_shuffled = fg_matrix[perm]
            count = self._count_monosemantic(act_valid, fg_shuffled, mono_threshold, selectivity_threshold)
            null_counts.append(count)

        null_counts = np.array(null_counts)
        p_value = float((null_counts >= real_count).mean())

        return {
            'real_monosemantic_count': real_count,
            'null_mean': float(null_counts.mean()),
            'null_std': float(null_counts.std()),
            'null_max': int(null_counts.max()),
            'p_value': p_value,
            'n_permutations': n_permutations,
            'enrichment': real_count / (null_counts.mean() + 1e-6),
        }

    @staticmethod
    def _count_monosemantic(activations, fg_matrix, threshold, selectivity):
        n_features = activations.shape[1]
        n_fgs = fg_matrix.shape[1]
        count = 0
        for f in range(n_features):
            if activations[:, f].std() < 1e-8:
                continue
            corrs = np.zeros(n_fgs)
            for g in range(n_fgs):
                if fg_matrix[:, g].std() > 1e-8:
                    r, _ = spearmanr(activations[:, f], fg_matrix[:, g])
                    corrs[g] = abs(r) if not np.isnan(r) else 0
            sorted_c = np.sort(corrs)[::-1]
            if sorted_c[0] > threshold and len(sorted_c) > 1 and sorted_c[0] > selectivity * sorted_c[1]:
                count += 1
        return count

    @staticmethod
    def visualize(results: Dict, output_dir: Path):
        """Plot permutation test results."""
        if 'error' in results:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        null_mean = results['null_mean']
        null_std = results['null_std']
        real = results['real_monosemantic_count']

        # Approximate null distribution
        x = np.arange(max(0, int(null_mean - 3 * null_std)), int(null_mean + 4 * null_std) + 1)
        from scipy.stats import norm
        y = norm.pdf(x, null_mean, null_std + 1e-6)
        ax.fill_between(x, y, alpha=0.3, color='gray', label='Null distribution')
        ax.plot(x, y, color='gray')
        ax.axvline(x=real, color='#C73E1D', linewidth=2, linestyle='-',
                  label=f'Observed: {real} (p={results["p_value"]:.4f})')
        ax.set_xlabel('Number of Monosemantic Features')
        ax.set_ylabel('Density')
        ax.set_title(f'Permutation Test: {real} vs {null_mean:.1f}±{null_std:.1f} by chance\n'
                    f'Enrichment: {results["enrichment"]:.1f}×')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / 'permutation_test.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'permutation_test.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)


# ── Single-Sample Decomposition Dashboard ────────────────────────────────────

# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the BatchTopK SAE on frozen slot embeddings and analyze its features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/sae")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sae-latent-dim", type=int, default=2048)
    parser.add_argument("--sae-k-fixed", type=int, default=50, help="Active features per sample")
    parser.add_argument("--sae-epochs", type=int, default=50)
    parser.add_argument("--pretrained-sae", type=str, default=None,
                        help="Path to a trained SAE (skip training, analyze only)")
    args = parser.parse_args()

    run_analysis(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        device=args.device,
        sae_latent_dim=args.sae_latent_dim,
        sae_sparsity=0.05,
        sae_k_fixed=args.sae_k_fixed,
        sae_epochs=args.sae_epochs,
        pretrained_sae=args.pretrained_sae,
        precision_recall=True,
        spectral_attribution=True,
        spectral_localization=True,
        permutation_test=True,
    )
