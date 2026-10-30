from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import AugmentationConfig, DataConfig, TrainingConfig

# Global scale factor cache (computed once, used by all datasets)
_global_scale_factor: Optional[float] = None
_molecule_db_cache: Dict[Path, Dict[str, torch.Tensor]] = {}
_mixture_dataset_cache: Dict[Path, Dict[str, torch.Tensor]] = {}


def _load_tensor_file(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    tensor = torch.load(path)
    if isinstance(tensor, torch.Tensor):
        return tensor.float()
    raise ValueError(f"Expected torch.Tensor at {path}, got {type(tensor)}")


def _load_molecule_database(cfg: DataConfig, database_file: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Load (and cache) molecule spectra database with raw intensities."""
    if database_file is None:
        database_file = "molecule_spectra_database.h5"
    db_path = cfg.processed_unmix_dir / database_file
    if db_path in _molecule_db_cache:
        return _molecule_db_cache[db_path]

    if not db_path.exists():
        raise FileNotFoundError(f"Missing molecule database at {db_path}")

    with h5py.File(db_path, "r") as f:
        molecule_data = {
            "gas": torch.from_numpy(f["gas_spectra"][:]).float(),
            "liq": torch.from_numpy(f["liq_spectra"][:]).float(),
            "functional_groups": torch.from_numpy(f["functional_groups"][:]).float(),
            "smiles": np.array(f["smiles"][:], dtype=str).tolist(),
        }
    _molecule_db_cache[db_path] = molecule_data
    return molecule_data


def _load_mixture_dataset(cfg: DataConfig, dataset_file: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Load (and cache) mixture dataset from processed_unmix directory."""
    filename = dataset_file or cfg.mixture_dataset
    dataset_path = cfg.processed_unmix_dir / filename
    if dataset_path in _mixture_dataset_cache:
        return _mixture_dataset_cache[dataset_path]
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing mixture dataset at {dataset_path}")
    
    with h5py.File(dataset_path, "r") as f:
        mixture = torch.from_numpy(f["mixture_spectra"][:]).float()
        component_indices = torch.from_numpy(f["component_indices"][:]).long()
        component_smiles = [
            tuple(raw.decode("utf-8") for raw in pair)
            for pair in f["component_smiles"][:]
        ]
        dataset = {
            "mixture": mixture,
            "component_indices": component_indices,
            "component_smiles": component_smiles,
        }
    _mixture_dataset_cache[dataset_path] = dataset
    return dataset


def get_global_scale_factor(cfg: DataConfig, force_recompute: bool = False) -> float:
    """Compute and cache global scale factor from all data types."""
    global _global_scale_factor
    
    if _global_scale_factor is not None and not force_recompute:
        return _global_scale_factor
    
    if not cfg.use_global_intensity_scale:
        _global_scale_factor = 1.0
        return _global_scale_factor
    
    print("[Scaling] Computing global scale factor from all data types...")
    
    # Load raw single-phase data from molecule database (gas + liquid)
    # These are used in Stage 1 (gas-liquid translation)
    molecule_db = _load_molecule_database(cfg)
    gas_raw = molecule_db["gas"]
    liq_raw = molecule_db["liq"]
    
    # Per-spectrum counting: treat gas and liquid as separate spectra,
    # matching how the model processes them independently in Stage 1.
    # Filter out zero-max entries (liquid-only molecules have gas = all zeros).
    gas_maxes = gas_raw.max(dim=1)[0]
    liq_maxes = liq_raw.max(dim=1)[0]

    # Concatenate gas and liquid maxes, then filter out zeros and NaNs
    all_single_maxes = torch.cat([gas_maxes, liq_maxes])
    valid_mask = (~torch.isnan(all_single_maxes)) & (all_single_maxes > 0)
    if not valid_mask.all():
        num_filtered = (~valid_mask).sum().item()
        print(f"  Filtered {num_filtered} zero/NaN single-phase entries")
    single_phase_maxes = all_single_maxes[valid_mask]

    print(f"  Single-phase (gas+liquid): {len(single_phase_maxes)} spectra "
          f"({len(gas_maxes)} gas + {len(liq_maxes)} liq, {(~valid_mask).sum().item()} filtered)")
    
    # Collect max values from all mixture datasets
    all_mixture_maxes = []
    
    # Binary mixtures (2-component) - Stage 2 training data
    binary_dataset = "mix_ml_instances.h5"
    if cfg.mixture_dataset != binary_dataset:
        # Only load binary if it's not the current training dataset
        try:
            binary_mixture_data = _load_mixture_dataset(cfg, binary_dataset)
            binary_maxes = binary_mixture_data["mixture"].max(dim=1)[0]
            all_mixture_maxes.append(binary_maxes)
            print(f"  Binary mixtures ({binary_dataset}): {len(binary_maxes)} samples")
        except FileNotFoundError:
            print(f"  Warning: Binary mixture dataset not found: {binary_dataset}")
    
    # Ternary mixtures (3-component) - Stage 2 training data
    ternary_dataset = "mix_ternary_ml_instances.h5"
    if cfg.mixture_dataset != ternary_dataset:
        # Only load ternary if it's not the current training dataset
        try:
            ternary_mixture_data = _load_mixture_dataset(cfg, ternary_dataset)
            ternary_maxes = ternary_mixture_data["mixture"].max(dim=1)[0]
            all_mixture_maxes.append(ternary_maxes)
            print(f"  Ternary mixtures ({ternary_dataset}): {len(ternary_maxes)} samples")
        except FileNotFoundError:
            print(f"  Warning: Ternary mixture dataset not found: {ternary_dataset}")
    
    # Current training dataset (could be binary or ternary)
    try:
        current_mixture_data = _load_mixture_dataset(cfg, cfg.mixture_dataset)
        current_maxes = current_mixture_data["mixture"].max(dim=1)[0]
        all_mixture_maxes.append(current_maxes)
        # Determine type based on filename
        if "ternary" in cfg.mixture_dataset.lower():
            print(f"  Ternary mixtures ({cfg.mixture_dataset}): {len(current_maxes)} samples")
        else:
            print(f"  Binary mixtures ({cfg.mixture_dataset}): {len(current_maxes)} samples")
    except FileNotFoundError:
        print(f"  Warning: Training mixture dataset not found: {cfg.mixture_dataset}")
    
    # Drug mixtures (optional, for evaluation)
    try:
        drug_mixture_data = _load_mixture_dataset(cfg, cfg.drug_mixture_dataset)
        drug_maxes = drug_mixture_data["mixture"].max(dim=1)[0]
        all_mixture_maxes.append(drug_maxes)
        print(f"  Drug mixtures ({cfg.drug_mixture_dataset}): {len(drug_maxes)} samples")
    except FileNotFoundError:
        print(f"  Warning: Drug mixture dataset not found: {cfg.drug_mixture_dataset}")
    
    # Combine all distributions: single-phase + all mixture types
    if all_mixture_maxes:
        mixture_maxes_combined = torch.cat(all_mixture_maxes, dim=0)
        combined_maxes = torch.cat([single_phase_maxes, mixture_maxes_combined], dim=0)
    else:
        # Fallback: only use single-phase if no mixtures found
        print("  Warning: No mixture datasets found, using only single-phase data")
        combined_maxes = single_phase_maxes
    
    # Compute 99th percentile threshold from all data
    threshold = torch.quantile(combined_maxes, 0.99).item()
    print(f"  99th percentile threshold: {threshold:.6f}")
    
    # Compute scale factor
    _global_scale_factor = cfg.global_intensity_scale / threshold if threshold > 0 else cfg.global_intensity_scale
    print(f"  Global scale factor: {_global_scale_factor:.6f}")
    print(f"  Total samples considered: {len(combined_maxes)}")
    
    return _global_scale_factor


class PureSpectrumDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        split_indices: Optional[List[int]] = None,
        include_fg: bool = True,
        n_functional_groups: Optional[int] = None,
        apply_normalization: bool = True,
    ) -> None:
        processed = cfg.processed_dir
        gas_raw = _load_tensor_file(processed / "gas_spectra.pt")
        liq_raw = _load_tensor_file(processed / "liq_spectra.pt")
        
        if apply_normalization:
            # Use global scale factor (same for train/val/test)
            scale_factor = get_global_scale_factor(cfg)
            
            # Apply scale factor and clamp to [0, 1]
            self.gas = (gas_raw * scale_factor).clamp(max=1.0)
            self.liq = (liq_raw * scale_factor).clamp(max=1.0)
        else:
            self.gas = gas_raw
            self.liq = liq_raw
            
        if include_fg:
            self.func_groups = _load_tensor_file(processed / "functional_group_labels_selected.pt")
            if (
                n_functional_groups is not None
                and self.func_groups.ndim == 2
                and self.func_groups.shape[1] >= n_functional_groups
            ):
                self.func_groups = self.func_groups[:, :n_functional_groups]
        else:
            self.func_groups = None

        if split_indices is None:
            self.indices = list(range(self.gas.shape[0]))
        else:
            self.indices = split_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        sample = {"gas": self.gas[i], "liq": self.liq[i], "index": i}
        if self.func_groups is not None:
            sample["fg"] = self.func_groups[i]
        return sample


class MixtureDataset(Dataset):
    """Dataset wrapping binary/ternary mixtures stored in processed_unmix/*.h5 files."""

    def __init__(
        self,
        cfg: DataConfig,
        dataset_file: Optional[str] = None,
        split_indices: Optional[List[int]] = None,
        augment: bool = False,
        aug_cfg: Optional[AugmentationConfig] = None,
    ) -> None:
        self.cfg = cfg
        self.dataset_file = dataset_file or cfg.mixture_dataset
        self.scale_factor = get_global_scale_factor(cfg)

        # Augmentation setup
        self.augment = augment
        self.aug_cfg = aug_cfg
        if augment and aug_cfg is not None and aug_cfg.enable:
            from .augmentation import augment_sample
            self._augment_fn = augment_sample

        # Determine which molecule database to use based on the mixture dataset
        # Drug mixtures use combined_molecule_database.h5 (contains both training and drug molecules)
        # Other mixtures use molecule_spectra_database.h5 (training molecules only)
        if "drugs" in self.dataset_file.lower():
            molecule_db_file = "combined_molecule_database.h5"
        else:
            molecule_db_file = "molecule_spectra_database.h5"

        self.molecule_db = _load_molecule_database(cfg, molecule_db_file)
        self.mixture_data = _load_mixture_dataset(cfg, self.dataset_file)

        all_indices = list(range(self.mixture_data["mixture"].shape[0]))
        self.indices = all_indices if split_indices is None else list(split_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_idx = self.indices[idx]
        comp_indices = self.mixture_data["component_indices"][dataset_idx]

        # Target components are always clean (unaugmented)
        components = (self.molecule_db["liq"][comp_indices] * self.scale_factor).clamp(max=1.0)

        # Use real mixture data (augmentation is applied in SyntheticMixtureDataset, not here)
        mixture = (self.mixture_data["mixture"][dataset_idx] * self.scale_factor).clamp(max=1.0)

        sample = {
            "mixture": mixture.clone(),
            "components": components.clone(),
            "indices": comp_indices.clone(),
            "smiles": self.mixture_data["component_smiles"][dataset_idx],
        }
        return sample

    def get_smiles(self, dataset_idx: int) -> List[str]:
        """Return SMILES strings for a given dataset index (respecting splits)."""
        base_idx = self.indices[dataset_idx]
        return list(self.mixture_data["component_smiles"][base_idx])


class SyntheticMixtureDataset(Dataset):
    """Generate synthetic mixtures by randomly combining K liquid spectra."""

    def __init__(
        self,
        cfg: DataConfig,
        num_components: int,
        num_samples: int,
        seed: int = 42,
        source: str = "train",
        num_drug_train_samples: int = 0,
        include_s1_drugs: bool = False,
        exclude_molecule_indices: Optional[set] = None,
        generate_all_combinations: bool = False,
        sampling_strategy: str = "random",
        hard_negative_temperature: float = 0.5,
        aug_cfg: Optional[AugmentationConfig] = None,
    ) -> None:
        """Args:
        cfg: Data configuration.
        """
        if source not in ("train", "drug", "drug_train", "stratified"):
            raise ValueError(f"Unknown source: {source!r}")

        self.cfg = cfg
        self.source = source
        self.num_components = num_components
        self.scale_factor = get_global_scale_factor(cfg)
        self.rng = np.random.default_rng(seed)
        self.generate_all = generate_all_combinations and source == "train"
        self.sampling_strategy = sampling_strategy
        self.temperature = hard_negative_temperature

        # Augmentation setup (always use augment_sample)
        self.aug_cfg = aug_cfg
        if aug_cfg is None:
            from .config import AugmentationConfig
            self.aug_cfg = AugmentationConfig()
        from .augmentation import augment_sample
        self._augment_fn = augment_sample

        # Log augmentation settings
        if self.aug_cfg.shift_percent == 0 or self.aug_cfg.shift_range == 0:
            print(f"[SyntheticDataset] Using clean synthesis (no augmentation)")
        else:
            print(f"[SyntheticDataset] Using shift_windows augmentation "
                  f"(percent={self.aug_cfg.shift_percent:.1%}, width={self.aug_cfg.shift_window_width}, "
                  f"range={self.aug_cfg.shift_range:.1f})")

        # ── Pool loading ──────────────────────────────────────────────
        self.pools: Dict[str, torch.Tensor] = {}       # name -> [N, L]
        self.pool_smiles: Dict[str, List[str]] = {}     # name -> list[str]

        # Training pool (needed for "train", "drug_train", "stratified")
        if source in ("train", "drug_train", "stratified"):
            molecule_db = _load_molecule_database(cfg)
            train_liq = molecule_db["liq"]
            train_smiles = molecule_db["smiles"]

            if source == "train":
                # Filter out excluded molecules (e.g., held-out)
                all_indices = set(range(train_liq.shape[0]))
                if exclude_molecule_indices:
                    avail = sorted(all_indices - exclude_molecule_indices)
                else:
                    avail = sorted(all_indices)
                self.pools["train"] = train_liq[avail]
                self.pool_smiles["train"] = [train_smiles[i] for i in avail]
            else:
                self.pools["train"] = train_liq
                self.pool_smiles["train"] = train_smiles

            self.functional_groups = molecule_db.get("functional_groups")
        else:
            self.functional_groups = None

        # Drug pool (needed for "drug", "drug_train", "stratified")
        if source in ("drug", "drug_train", "stratified"):
            training_db = _load_molecule_database(cfg, "molecule_spectra_database.h5")
            training_smiles_set = set(training_db["smiles"])

            combined_db = _load_molecule_database(cfg, "combined_molecule_database.h5")
            combined_liq = combined_db["liq"]
            combined_smiles = combined_db["smiles"]

            if include_s1_drugs:
                # Include ALL 115 drug molecules (both U and S1).
                # Load drug mixture dataset to identify all drug SMILES.
                drug_mixture_data = _load_mixture_dataset(cfg, cfg.drug_mixture_dataset)
                drug_comp_indices = drug_mixture_data["component_indices"]  # [N, K]
                # Get unique molecule indices used in drug mixtures
                unique_drug_mol_idx = set(int(x) for row in drug_comp_indices for x in row)
                # These indices refer to combined_molecule_database
                drug_idx = sorted(unique_drug_mol_idx)
                print(f"[SyntheticDataset] include_s1_drugs=True: using all {len(drug_idx)} drug molecules")
            else:
                drug_idx = [
                    i for i, s in enumerate(combined_smiles)
                    if s not in training_smiles_set
                ]
            if len(drug_idx) == 0:
                raise ValueError("No drug molecules found in combined database")

            self.pools["drug"] = combined_liq[drug_idx]
            self.pool_smiles["drug"] = [combined_smiles[i] for i in drug_idx]

        # Print pool sizes
        for name, pool in self.pools.items():
            print(f"[SyntheticDataset] Pool '{name}': {pool.shape[0]} molecules")

        # ── Inverse-frequency weights for training molecules ─────────
        self.train_freq_weights: Optional[np.ndarray] = None
        if (self.aug_cfg.balance_rare_molecules
                and source in ("drug_train", "stratified", "train")
                and "train" in self.pools):
            from collections import Counter
            mixture_data = _load_mixture_dataset(cfg, cfg.mixture_dataset)
            component_indices = mixture_data["component_indices"]  # [N, K]
            mol_count: Counter = Counter()
            for row in component_indices:
                for mol_idx in row.tolist():
                    mol_count[int(mol_idx)] += 1

            n_train = self.pools["train"].shape[0]
            weights = np.empty(n_train, dtype=np.float64)
            for i in range(n_train):
                count = mol_count.get(i, 0)
                weights[i] = 1.0 / max(count, 1)
            weights /= weights.sum()
            self.train_freq_weights = weights
            print(f"[SyntheticDataset] Inverse-frequency weighting enabled for {n_train} training molecules "
                  f"(min_w={weights.min():.6f}, max_w={weights.max():.6f})")

        # ── Sample generation ─────────────────────────────────────────
        self.stratified_samples = None   # used by "stratified"
        self.all_combinations = None     # used by "train" + generate_all
        self.use_streaming = False

        if source == "train":
            # available_indices for sampling strategies (indices into training pool)
            self.available_indices = list(range(self.pools["train"].shape[0]))

            if len(self.available_indices) < num_components:
                raise ValueError(
                    f"Not enough molecules ({len(self.available_indices)}) to form "
                    f"{num_components}-component mixtures"
                )

            self._init_sampling_strategy()

            if self.generate_all:
                from itertools import combinations
                from math import comb

                n = len(self.available_indices)
                total_combinations = comb(n, num_components)
                self.num_samples = total_combinations
                memory_threshold = 100_000_000
                if total_combinations > memory_threshold:
                    print(f"[SyntheticDataset] Using streaming for {total_combinations:,} "
                          f"{num_components}-combinations from {n} molecules")
                    self.use_streaming = True
                else:
                    print(f"[SyntheticDataset] Generating all {total_combinations:,} "
                          f"{num_components}-combinations from {n} molecules...")
                    self.all_combinations = list(combinations(self.available_indices, num_components))
                    print(f"[SyntheticDataset] Generated {len(self.all_combinations):,} unique combinations")
            else:
                self.num_samples = num_samples

        elif source == "drug":
            n_drug = self.pools["drug"].shape[0]
            if n_drug < num_components:
                raise ValueError(
                    f"Not enough drug molecules ({n_drug}) for "
                    f"{num_components}-component drug-only mixtures"
                )
            self.num_samples = num_samples
            print(f"[SyntheticDataset] Mode: drug-only (all components from drugs)")

        elif source == "drug_train":
            self.num_samples = num_samples
            print(f"[SyntheticDataset] Mode: drug+training (1 drug + {num_components - 1} training per mixture)")

        elif source == "stratified":
            from itertools import combinations
            from math import comb

            if num_components != 2:
                raise ValueError(
                    f"Stratified sampling currently only supports binary (K=2), got K={num_components}"
                )

            n_drugs = self.pools["drug"].shape[0]
            total_drug_drug = comb(n_drugs, num_components)
            print(f"[SyntheticDataset] Generating all {total_drug_drug:,} drug+drug combinations...")

            all_drug_combos = list(combinations(range(n_drugs), num_components))
            self.stratified_samples = [
                ("drug_drug", combo) for combo in all_drug_combos
            ]

            if num_drug_train_samples > 0:
                print(f"[SyntheticDataset] Adding {num_drug_train_samples:,} drug+training samples...")
                n_train = self.pools["train"].shape[0]
                for _ in range(num_drug_train_samples):
                    d_idx = self.rng.choice(n_drugs)
                    t_idx = self.rng.choice(n_train, p=self.train_freq_weights)
                    self.stratified_samples.append(("drug_training", (d_idx, t_idx)))

            self.rng.shuffle(self.stratified_samples)
            self.num_samples = len(self.stratified_samples)
            print(f"[SyntheticDataset] Stratified mode: {total_drug_drug:,} drug+drug + "
                  f"{num_drug_train_samples:,} drug+training = {self.num_samples:,} total samples")

    def __len__(self) -> int:
        return self.num_samples

    # ── Sampling strategies (source="train" only) ─────────────────

    def _init_sampling_strategy(self):
        """Initialize data structures for different sampling strategies."""
        if self.sampling_strategy == "random":
            self.similarity_matrix = None
            self.functional_similarity = None
            print(f"[SyntheticDataset] Using random sampling strategy")

        elif self.sampling_strategy == "hard_spectral":
            print(f"[SyntheticDataset] Computing spectral similarity matrix for hard negative sampling...")
            spectra_subset = self.pools["train"][self.available_indices]
            spectra_norm = spectra_subset / (spectra_subset.norm(dim=1, keepdim=True) + 1e-8)
            self.similarity_matrix = None
            self.spectra_norm = spectra_norm
            print(f"[SyntheticDataset] Using hard_spectral sampling (temperature={self.temperature})")

        elif self.sampling_strategy == "hard_functional":
            if self.functional_groups is None:
                print(f"[Warning] Functional groups not available, falling back to random sampling")
                self.sampling_strategy = "random"
                self.functional_similarity = None
            else:
                print(f"[SyntheticDataset] Computing functional group similarity for hard negative sampling...")
                fg_subset = self.functional_groups[self.available_indices]
                self.functional_similarity = self._compute_functional_similarity(fg_subset)
                print(f"[SyntheticDataset] Using hard_functional sampling (temperature={self.temperature})")

        elif self.sampling_strategy == "diverse":
            if self.functional_groups is None:
                print(f"[Warning] Functional groups not available, falling back to random sampling")
                self.sampling_strategy = "random"
                self.functional_similarity = None
            else:
                print(f"[SyntheticDataset] Computing functional group similarity for diverse sampling...")
                fg_subset = self.functional_groups[self.available_indices]
                self.functional_similarity = self._compute_functional_similarity(fg_subset)
                print(f"[SyntheticDataset] Using diverse sampling strategy")
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _compute_functional_similarity(self, fg_matrix: torch.Tensor) -> torch.Tensor:
        """Compute Jaccard similarity between functional group vectors (vectorized)."""
        fg_binary = (fg_matrix > 0).float()
        dot_product = torch.mm(fg_binary, fg_binary.T)
        sums = fg_binary.sum(dim=1, keepdim=True)
        union = sums + sums.T - dot_product
        jaccard = dot_product / (union + 1e-8)
        return jaccard

    def _sample_with_strategy(self, anchor_idx: Optional[int] = None) -> np.ndarray:
        """Sample K molecules using the specified strategy."""
        k = self.num_components

        if self.sampling_strategy == "random" or self.generate_all:
            if self.train_freq_weights is not None:
                w = self.train_freq_weights[self.available_indices]
                w = w / w.sum()
                return self.rng.choice(self.available_indices, size=k, replace=False, p=w)
            return self.rng.choice(self.available_indices, size=k, replace=False)

        elif self.sampling_strategy == "hard_spectral":
            first_idx = self.rng.choice(len(self.available_indices))
            selected = [self.available_indices[first_idx]]

            for _ in range(k - 1):
                anchor_spec = self.spectra_norm[first_idx]
                similarities = (self.spectra_norm * anchor_spec).sum(dim=1)
                mask = torch.ones(len(self.available_indices), dtype=torch.bool)
                for s in selected:
                    mask[self.available_indices.index(s)] = False
                probs = torch.softmax(similarities[mask] / self.temperature, dim=0).numpy()
                available_masked = [self.available_indices[i] for i, m in enumerate(mask) if m]
                next_idx = self.rng.choice(available_masked, p=probs)
                selected.append(next_idx)

            return np.array(selected)

        elif self.sampling_strategy == "hard_functional":
            first_idx = self.rng.choice(len(self.available_indices))
            selected = [self.available_indices[first_idx]]

            for _ in range(k - 1):
                similarities = self.functional_similarity[first_idx]
                mask = torch.ones(len(self.available_indices), dtype=torch.bool)
                for s in selected:
                    mask[self.available_indices.index(s)] = False
                probs = torch.softmax(similarities[mask] / self.temperature, dim=0).numpy()
                available_masked = [self.available_indices[i] for i, m in enumerate(mask) if m]
                next_idx = self.rng.choice(available_masked, p=probs)
                selected.append(next_idx)

            return np.array(selected)

        elif self.sampling_strategy == "diverse":
            first_idx = self.rng.choice(len(self.available_indices))
            selected = [self.available_indices[first_idx]]

            for _ in range(k - 1):
                similarities = self.functional_similarity[first_idx]
                mask = torch.ones(len(self.available_indices), dtype=torch.bool)
                for s in selected:
                    mask[self.available_indices.index(s)] = False
                dissimilarities = 1.0 - similarities[mask]
                probs = torch.softmax(dissimilarities / self.temperature, dim=0).numpy()
                available_masked = [self.available_indices[i] for i, m in enumerate(mask) if m]
                next_idx = self.rng.choice(available_masked, p=probs)
                selected.append(next_idx)

            return np.array(selected)

    def _nth_combination(self, n: int, k: int, index: int) -> tuple:
        """Get the index-th combination of k elements from range(n) using combinatorial number system."""
        from math import comb

        result = []
        for i in range(k):
            j = k - i - 1
            while j < n:
                c = comb(j, k - i)
                if c > index:
                    break
                j += 1
            j -= 1
            result.append(self.available_indices[j])
            index -= comb(j, k - i)

        return tuple(result)

    # ── __getitem__ ───────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        k = self.num_components

        # ── Determine component spectra and SMILES ────────────────
        if self.source == "train":
            if self.generate_all:
                if self.use_streaming:
                    comp_idx = self._nth_combination(
                        len(self.available_indices), k, idx
                    )
                    comp_idx = torch.tensor(comp_idx, dtype=torch.long)
                else:
                    comp_idx = torch.tensor(self.all_combinations[idx], dtype=torch.long)
            else:
                comp_idx = torch.from_numpy(self._sample_with_strategy()).long()

            components_raw = self.pools["train"][comp_idx]
            smiles_list = tuple(self.pool_smiles["train"][i] for i in comp_idx.tolist())
            indices_out = comp_idx.clone()

        elif self.source == "drug":
            drug_pool = self.pools["drug"]
            sel = self.rng.choice(drug_pool.shape[0], size=k, replace=False)
            components_raw = drug_pool[sel]
            smiles_list = tuple(self.pool_smiles["drug"][i] for i in sel)
            indices_out = torch.arange(k).long()

        elif self.source == "drug_train":
            components_list = []
            smiles_tmp = []
            drug_pool = self.pools["drug"]
            train_pool = self.pools["train"]
            # 1 drug component
            d_idx = self.rng.choice(drug_pool.shape[0])
            components_list.append(drug_pool[d_idx])
            smiles_tmp.append(self.pool_smiles["drug"][d_idx])
            # K-1 training components
            for _ in range(k - 1):
                t_idx = self.rng.choice(train_pool.shape[0], p=self.train_freq_weights)
                components_list.append(train_pool[t_idx])
                smiles_tmp.append(self.pool_smiles["train"][t_idx])
            components_raw = torch.stack(components_list)
            smiles_list = tuple(smiles_tmp)
            indices_out = torch.arange(k).long()

        elif self.source == "stratified":
            sample_type, indices = self.stratified_samples[idx]
            drug_pool = self.pools["drug"]
            if sample_type == "drug_drug":
                components_list = [drug_pool[i] for i in indices]
                smiles_list = tuple(self.pool_smiles["drug"][i] for i in indices)
            elif sample_type == "drug_training":
                d_idx, t_idx = indices
                components_list = [drug_pool[d_idx], self.pools["train"][t_idx]]
                smiles_list = (self.pool_smiles["drug"][d_idx], self.pool_smiles["train"][t_idx])
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")
            components_raw = torch.stack(components_list)
            indices_out = torch.arange(k).long()

        # ── Common tail: scale, augment, return ───────────────────
        components = (components_raw * self.scale_factor).clamp(max=1.0)

        components_np = components.numpy()
        mixture_np = self._augment_fn(components_np, self.aug_cfg, self.rng)
        mixture = torch.from_numpy(mixture_np.astype(np.float32)).clamp(0, 1)

        return {
            "mixture": mixture.clone(),
            "components": components.clone(),
            "indices": indices_out,
            "smiles": smiles_list,
        }


def build_dataloader(
    dataset: Dataset,
    cfg: TrainingConfig,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )


def split_indices(
    num_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def kfold_split_indices(
    num_samples: int,
    k: int = 5,
    seed: int = 42,
) -> List[List[int]]:
    """Split sample indices into *k* non-overlapping folds."""
    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)
    fold_size = num_samples // k
    remainder = num_samples % k
    folds: List[List[int]] = []
    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)
        folds.append(indices[start:end])
        start = end
    return folds


def compute_molecule_holdout(
    cfg: DataConfig,
    train_indices: List[int],
    seed: int = 42,
) -> Tuple[set, set]:
    """Compute which molecules to hold out from Stage 2 mixture training."""
    from collections import Counter

    n_holdout = cfg.n_holdout_molecules
    if n_holdout <= 0:
        return set(), set()

    mixture_data = _load_mixture_dataset(cfg, cfg.mixture_dataset)
    component_indices = mixture_data["component_indices"]  # [N, K]

    mol_count: Counter = Counter()
    for idx in train_indices:
        for mol_idx in component_indices[idx].tolist():
            mol_count[int(mol_idx)] += 1
    all_mols = sorted(mol_count.keys())
    # Sort by frequency (ascending), break ties by index for reproducibility
    rng = random.Random(seed)
    shuffled = list(all_mols)
    rng.shuffle(shuffled)
    sorted_mols = sorted(shuffled, key=lambda m: mol_count[m])

    holdout = set(sorted_mols[:n_holdout])
    kept = set(all_mols) - holdout

    print(f"[Holdout] {n_holdout} molecules held out from Stage 2 training")
    print(f"  Kept (S12): {len(kept)}, Held-out (become S1): {len(holdout)}")

    return kept, holdout


def filter_indices_by_molecules(
    cfg: DataConfig,
    indices: List[int],
    kept_molecules: set,
) -> List[int]:
    """Keep only instances where ALL components are in ``kept_molecules``."""
    mixture_data = _load_mixture_dataset(cfg, cfg.mixture_dataset)
    component_indices = mixture_data["component_indices"]

    filtered = []
    for idx in indices:
        comps = component_indices[idx].tolist()
        if all(int(c) in kept_molecules for c in comps):
            filtered.append(idx)

    n_removed = len(indices) - len(filtered)
    print(f"[Holdout] Filtered training instances: {len(filtered)} kept, {n_removed} removed")
    return filtered


class AutoencoderDataset(PureSpectrumDataset):
    """Stage 1 dataset that feeds liquid as both input and target (autoencoder control)."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        sample = {"gas": self.liq[i], "liq": self.liq[i], "index": i}
        if self.func_groups is not None:
            sample["fg"] = self.func_groups[i]
        return sample


class GasAutoencoderDataset(PureSpectrumDataset):
    """Stage 1 dataset that feeds gas as both input and target (gas->gas autoencoder control)."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        sample = {"gas": self.gas[i], "liq": self.gas[i], "index": i}
        if self.func_groups is not None:
            sample["fg"] = self.func_groups[i]
        return sample


class ShuffledPairsDataset(PureSpectrumDataset):
    """Stage 1 dataset with randomized gas-liquid pairings (negative control)."""

    def __init__(self, cfg: DataConfig, split_indices=None, shuffle_seed: int = 123, **kwargs):
        super().__init__(cfg, split_indices, **kwargs)
        rng = np.random.default_rng(shuffle_seed)
        self.liq_perm = rng.permutation(len(self.indices))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        j = self.indices[self.liq_perm[idx]]
        sample = {"gas": self.gas[i], "liq": self.liq[j], "index": i}
        if self.func_groups is not None:
            sample["fg"] = self.func_groups[i]
        return sample
