"""
Shared data loading utilities for all baseline models.
Ensures exact same train/test splits as slot_ir.
"""

import random
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader


class UnmixDataset(Dataset):
    """Unified dataset for spectral unmixing matching slot_ir format."""

    def _load_molecule_database(self) -> Dict[str, np.ndarray]:
        """Load pure component spectra database."""
        db_path = self.data_dir / "molecule_spectra_database.h5"
        if not db_path.exists():
            raise FileNotFoundError(f"Molecule database not found: {db_path}")

        with h5py.File(db_path, "r") as f:
            return {
                "gas": np.array(f["gas_spectra"][:]),
                "liq": np.array(f["liq_spectra"][:]),
                "functional_groups": np.array(f["functional_groups"][:]),
                "smiles": np.array(f["smiles"][:], dtype=str),
            }

    def _load_mixture_dataset(self) -> Dict[str, np.ndarray]:
        """Load mixture dataset."""
        filename = "mix_ml_instances.h5" if self.dataset_type == "binary" else "mix_ternary_ml_instances.h5"
        mix_path = self.data_dir / filename

        if not mix_path.exists():
            raise FileNotFoundError(f"Mixture dataset not found: {mix_path}")

        with h5py.File(mix_path, "r") as f:
            return {
                "mixture": np.array(f["mixture_spectra"][:]),
                "component_indices": np.array(f["component_indices"][:]),
                "component_smiles": [
                    tuple(raw.decode("utf-8") for raw in pair)
                    for pair in f["component_smiles"][:]
                ],
            }

    def _create_splits(self) -> np.ndarray:
        """Create train/val/test splits (80/10/10) with fixed seed."""
        n_samples = len(self.mixture_data["mixture"])
        train_idx, val_idx, test_idx = split_indices_compatible(
            n_samples, train_ratio=0.8, val_ratio=0.1, seed=self.seed,
        )

        if self.split == "train":
            return np.array(train_idx)
        elif self.split == "val":
            return np.array(val_idx)
        else:  # test
            return np.array(test_idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __init__(
        self,
        data_dir: str = "../data/processed_unmix",
        dataset_type: str = "binary",  # "binary" or "ternary"
        split: str = "train",  # "train", "val", or "test"
        seed: int = 42,
        apply_scaling: bool = True,  # Apply global scale factor
        scale_factor: Optional[float] = None,  # Pre-computed scale factor
    ):
        """Args:
        data_dir: Path to processed_unmix directory
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.seed = seed
        self.apply_scaling = apply_scaling

        # Load molecule database
        self.molecule_db = self._load_molecule_database()

        # Load mixture dataset
        self.mixture_data = self._load_mixture_dataset()

        # Compute or use provided scale factor
        if apply_scaling:
            if scale_factor is None:
                self.scale_factor = compute_global_scale_factor(str(data_dir))
            else:
                self.scale_factor = scale_factor
        else:
            self.scale_factor = 1.0

        # Create split indices (same as slot_ir: 80/10/10)
        self.indices = self._create_splits()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Returns:
        Dictionary with:
        """
        actual_idx = self.indices[idx]

        # Get mixture spectrum
        mixture = self.mixture_data["mixture"][actual_idx].astype(np.float32)

        # Get component indices
        comp_indices = self.mixture_data["component_indices"][actual_idx]

        # Get ground truth components (liquid phase)
        components = self.molecule_db["liq"][comp_indices].astype(np.float32)

        # Apply scaling and clamp to [0, 1] (matching src/data.py .clamp(max=1.0))
        if self.apply_scaling:
            mixture = np.clip(mixture * self.scale_factor, 0, 1)
            components = np.clip(components * self.scale_factor, 0, 1)

        return {
            "mixture": mixture,
            "components": components,
            "component_indices": comp_indices,
            "n_components": len(comp_indices),
        }


def get_unmix_dataloaders(
    data_dir: str = "../data/processed_unmix",
    dataset_type: str = "binary",
    batch_size: int = 32,
    num_workers: int = 0,  # Changed default to 0 to avoid "too many open files" issue
    seed: int = 42,
    apply_scaling: bool = True,
    scale_factor: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders with exact same splits as slot_ir."""
    # Compute scale factor once if not provided
    if apply_scaling and scale_factor is None:
        scale_factor = compute_global_scale_factor(data_dir)
    
    train_dataset = UnmixDataset(data_dir, dataset_type, "train", seed, apply_scaling, scale_factor)
    val_dataset = UnmixDataset(data_dir, dataset_type, "val", seed, apply_scaling, scale_factor)
    test_dataset = UnmixDataset(data_dir, dataset_type, "test", seed, apply_scaling, scale_factor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,  # pin_memory requires num_workers > 0
        persistent_workers=False,  # Don't keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=False,
    )

    return train_loader, val_loader, test_loader


def split_indices_compatible(
    num_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Create train/val/test splits using random.Random (matching slot_ir)."""
    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def compute_global_scale_factor(
    data_dir: str = "../data/processed_unmix",
    global_intensity_scale: float = 1.0,
) -> float:
    """Compute global intensity scale factor (99th percentile) across all data."""
    data_dir = Path(data_dir)

    print("[Scaling] Computing global scale factor from all data types...")

    # ── Single-phase: gas and liquid counted SEPARATELY (not per-molecule max) ──
    db_path = data_dir / "molecule_spectra_database.h5"
    if not db_path.exists():
        raise FileNotFoundError(f"Molecule database not found: {db_path}")

    with h5py.File(db_path, "r") as f:
        gas_raw = np.array(f["gas_spectra"][:])
        liq_raw = np.array(f["liq_spectra"][:])

    gas_maxes = np.max(gas_raw, axis=1)   # [N_mol]
    liq_maxes = np.max(liq_raw, axis=1)   # [N_mol]

    # Concatenate gas + liquid maxes (NOT per-molecule max)
    all_single_maxes = np.concatenate([gas_maxes, liq_maxes])

    # Filter out zeros and NaNs (matching src/data.py)
    valid_mask = (~np.isnan(all_single_maxes)) & (all_single_maxes > 0)
    num_filtered = int(np.sum(~valid_mask))
    if num_filtered > 0:
        print(f"  Filtered {num_filtered} zero/NaN single-phase entries")
    single_phase_maxes = all_single_maxes[valid_mask]

    print(f"  Single-phase (gas+liquid): {len(single_phase_maxes)} spectra "
          f"({len(gas_maxes)} gas + {len(liq_maxes)} liq, {num_filtered} filtered)")

    # ── Mixture datasets ──
    all_mixture_maxes = []

    for mix_file, label in [
        ("mix_ml_instances.h5", "Binary mixtures"),
        ("mix_ternary_ml_instances.h5", "Ternary mixtures"),
        ("mix_drugs_ml_instances.h5", "Drug mixtures"),
    ]:
        mix_path = data_dir / mix_file
        if mix_path.exists():
            with h5py.File(mix_path, "r") as f:
                mix_spectra = np.array(f["mixture_spectra"][:])
                mix_maxes = np.max(mix_spectra, axis=1)
                all_mixture_maxes.append(mix_maxes)
                print(f"  {label} ({mix_file}): {len(mix_maxes)} samples")

    # ── Combine all ──
    if all_mixture_maxes:
        mixture_maxes_combined = np.concatenate(all_mixture_maxes)
        combined_maxes = np.concatenate([single_phase_maxes, mixture_maxes_combined])
    else:
        print("  Warning: No mixture datasets found, using only single-phase data")
        combined_maxes = single_phase_maxes

    # 99th percentile (matching torch.quantile(x, 0.99))
    threshold = float(np.percentile(combined_maxes, 99))
    print(f"  99th percentile threshold: {threshold:.6f}")

    scale_factor = global_intensity_scale / threshold if threshold > 0 else global_intensity_scale
    print(f"  Global scale factor: {scale_factor:.6f}")
    print(f"  Total samples considered: {len(combined_maxes)}")

    return scale_factor
