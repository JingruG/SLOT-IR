"""Unified preprocessing script for IR spectra data.
This script handles:
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import logging

from rdkit import Chem

# Tier-based functional group selection - All 4 tiers explicitly defined
# Updated to include aromatic ring detection for better coverage
# ── Functional group tiers ────────────────────────────────────────────────────
# Tier 1: Strong SAE validation (r>0.5 at encoder layer 6), sufficient samples
# Tier 2: Moderate validation (r>0.3) or spectrally important
# Tier 3: Merged groups (combine spectrally similar FGs)
# Tier 4: Kept for completeness but low validation signal
#
# Changes from v1:
#   - Ar SMARTS broadened: [cR1] -> [c] (any aromatic C, was 14 -> ~5000+ molecules)
#   - Added merged groups: Amide, Sulfonamide, Halide (union of sub-groups)
#   - Dropped: Ar-O-Ar, Peroxide, C=S, P=O, P=S, NCS, S=O (too rare or noisy)
#   - Reordered by SAE evidence strength
TIER_1 = ['NH', 'OH', 'Ar', 'SH', 'CHO', 'N=N', 'COOH', 'COO', 'NH2', 'SO3H', 'C#N']  # 11: strong signal
TIER_2 = ['Ketone', 'N', 'SO2', 'C=N', 'C-F', 'C-O-C', 'NO2', 'NCO', 'SCN', 'Thioether']  # 10: moderate signal
TIER_3 = ['Amide', 'Sulfonamide', 'Halide', 'Ar-Het',  # 4 merged groups
          'CONH2', 'CONH', 'CON',                        # amide sub-types (kept for fine-grained analysis)
          'SO2NH', 'SO2NH2', 'SO2N',                      # sulfonamide sub-types
          'C-Cl', 'C-Br', 'C-I',                          # halide sub-types
          'Ar-N', 'Ar-NH', 'Ar-O', 'Ar-S']               # aromatic heteroatom sub-types
TIER_4 = ['C=C', 'C#C', 'Ar-O-C', 'Ar-O-Ar', 'C=S', 'P=O', 'P=S',  # rare/low-signal
          'NCS', 'Peroxide', 'S=O']

# All groups (44 total - all tiers)
ALL_GROUPS = TIER_1 + TIER_2 + TIER_3 + TIER_4

# Selected groups (32 total - Tier 1, 2, 3) - kept for backward compatibility
SELECTED_GROUPS = TIER_1 + TIER_2 + TIER_3


class SMILESProcessor:
    """SMILES processor for functional group extraction"""
    def __init__(self):
        self.functional_groups = {
            # --- Acids and Derivatives (High Priority) ---
            'COOH': '[CX3](=O)[OX2H1]',                        # Carboxylic acid
            'SO3H': '[S](=O)(=O)[OH]',                         # Sulfonic acid
            'COO': '[CX3](=O)[OX2H0][#6]',                     # Ester
            'CONH2': '[CX3](=O)[NX3;H2]',                      # Primary amide
            'CONH': '[CX3](=O)[NX3;H1][#6]',                   # Secondary amide
            'CON': '[CX3](=O)[NX3;H0]([#6])[#6]',              # Tertiary amide
            
            # --- Sulfonamides (High Priority) ---
            'SO2NH2': '[SX4](=O)(=O)[NX3;H2]',                # Primary sulfonamide
            'SO2NH': '[SX4](=O)(=O)[NX3;H1][#6]',             # Secondary sulfonamide
            'SO2N': '[SX4](=O)(=O)[NX3;H0]([#6])[#6]',        # Tertiary sulfonamide
            
            # --- Carbonyls (Specific Types) ---
            'CHO': '[CX3H1](=O)[#6]',                          # More specific aldehyde pattern (remove ,H)
            'Ketone': '[#6][CX3](=O)[#6]',                     # Ketone (including in heterocycles)
            
            # --- Phosphorus Groups ---
            'P=S': '[PX4](=[SX1])',                            # Thiophosphate
            'P=O': '[PX4](=[OX1])',                            # Phosphate
            
            # --- Nitrogen Groups ---
            'NO2': '[NX3+](=O)[O-]',                           # Nitro group
            'C#N': '[CX2]#[NX1]',                              # Nitrile
            'NH2': '[NX3;H2;!$(NC=O);!$(NS(=O)(=O))][#6]',    # Primary amine
            'NH': '[NX3;H1;!$(NC=O);!$(NS(=O)(=O))]([#6])[#6]', # Secondary amine
            'N': '[NX3;H0;!$(NC=O);!$(NS(=O)(=O))]([#6])([#6])[#6]', # Tertiary amine
            'NCS': '[N]=[C]=[S]',                              # Isothiocyanate
            'NCO': '[NX3][CX3]=[OX1]',                         # Isocyanate/Imide carbonyl
            'C=N': '[CX3]=[NX2]',                              # Imine
            'N=N': '[NX2]=[NX2]',                              # Azo group
            
            # --- Oxygen Groups ---
            'OH': '[OX2H;!$(O[CX3](=O));!$(OS(=O)(=O))]',      # Hydroxyl (not in COOH or SO3H)
            'Ar-O-Ar': '[OX2](c)(c)',                          # Diaryl ether
            'Ar-O-C': '[OX2](c)[CX4]',                         # Aryl-alkyl ether
            'C-O-C': '[OD2]([#6;!$(C=O)])[#6;!$(C=O)]',        # Ether (aliphatic)
            'Peroxide': '[OX2][OX2]',                          # Peroxide linkage
            
            # --- Sulfur Groups ---
            'SCN': '[SX2][CX2]#[NX1]',                         # Thiocyanate (changed to avoid overlap with C#N)
            'Thioether': '[SX2]([#6])[#6]',                    # Thioether
            'C=S': '[CX3]=[SX1]',                              # Thiocarbonyl (including thiocarbamate)
            'SH': '[SX2H]',                                    # Thiol
            'S=O': '[SX3]=[OX1]',                              # Sulfoxide
            'SO2': '[SX4](=O)=O',                              # Sulfone
            
            # --- Unsaturation ---
            'C=C': '[CX3]=[CX3]',                              # Alkene
            'C#C': '[CX2]#[CX2]',                              # Alkyne

            # --- Aromatic Systems ---
            # IR signatures: C-H aromatic stretch 3000-3100 cm⁻¹, ring modes 1450-1600 cm⁻¹
            'Ar': '[c]',                                       # Any aromatic carbon (broadened from [cR1])
            'Ar-N': '[nX2]',                                   # Pyridine-type nitrogen (sp2, in ring)
            'Ar-NH': '[nX3H1]',                                # Pyrrole-type NH (sp3-like, in ring)
            'Ar-O': '[oX2]',                                   # Furan-type oxygen (in 5-membered ring)
            'Ar-S': '[sX2]',                                   # Thiophene-type sulfur (in ring)

            # --- Merged Groups (union of sub-types) ---
            # These capture the dominant IR signature shared by the sub-types.
            # Amide: C(=O)N in any form (primary, secondary, tertiary, + isocyanate/imide)
            'Amide': '[CX3](=O)[NX3]',
            # Sulfonamide: S(=O)(=O)N in any form
            'Sulfonamide': '[SX4](=O)(=O)[NX3]',
            # Halide: any C-halogen bond (union of C-F, C-Cl, C-Br, C-I)
            'Halide': '[#6][F,Cl,Br,I]',
            # Aromatic heteroatom: any heteroatom in aromatic ring
            'Ar-Het': '[n,o,s]',

            # --- Halogens (handled separately with custom logic) ---
        }
        
        # Add halogen patterns separately for special handling
        self.halogen_patterns = {
            'C-F': '[#6][F]',
            'C-Cl': '[#6][Cl]', 
            'C-Br': '[#6][Br]',
            'C-I': '[#6][I]'
        }
        
        # Combine all functional groups for the vector
        self.all_groups = list(self.functional_groups.keys()) + list(self.halogen_patterns.keys())
        self.fg_names = self.all_groups
        self.n_functional_groups = len(self.fg_names)
        self.fg_map = {name: i for i, name in enumerate(self.fg_names)}

    def count_halogen_bonds(self, mol, halogen_symbol):
        """Count C-X bonds where X is a halogen, handling CF3, CCl3, etc correctly."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == halogen_symbol:
                # Each halogen forms one C-X bond
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'C':
                        count += 1
        return count

    def extract_functional_groups_with_counts(self, smiles):
        """Extract functional groups with counts."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_functional_groups, dtype=float)
        
        fg_vector = np.zeros(self.n_functional_groups, dtype=float)

        # --- Process SMARTS-based functional groups ---
        for fg_name, smarts_pattern in self.functional_groups.items():
            if smarts_pattern is None:
                continue
                
            try:
                pattern = Chem.MolFromSmarts(smarts_pattern)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    fg_vector[self.fg_map[fg_name]] = len(matches)
            except Exception as e:
                logging.warning(f"Error processing {fg_name} with pattern {smarts_pattern}: {e}")
                continue

        # --- Process Halogens with Custom Logic ---
        for halogen_name, smarts_pattern in self.halogen_patterns.items():
            halogen_symbol = halogen_name.split('-')[1]  # Extract F, Cl, Br, I
            count = self.count_halogen_bonds(mol, halogen_symbol)
            fg_vector[self.fg_map[halogen_name]] = count

        return fg_vector

    def get_functional_group_names(self):
        """Get list of functional group names."""
        return self.fg_names

    def smiles_to_functional_groups(self, smiles_list):
        """Convert list of SMILES to functional group count matrix."""
        fg_matrix = []
        for smiles in smiles_list:
            fg_vector = self.extract_functional_groups_with_counts(smiles)
            fg_matrix.append(fg_vector)
        return np.array(fg_matrix)


def is_valid(obj):
    """Check if object is valid (not None and recursively valid)."""
    if obj is None:
        return False
    if isinstance(obj, dict):
        return all(is_valid(v) for v in obj.values())
    if isinstance(obj, list):
        return all(is_valid(v) for v in obj)
    return True


def load_json(path):
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# SMILES to exclude from all datasets
EXCLUDED_SMILES = {'CCS(=O)(=O)[O-]'}


def load_drug_smiles(raw_data_dir: str) -> set:
    """Load drug molecule SMILES from liq_drugs_ml.json for exclusion."""
    drug_path = Path(raw_data_dir) / "liq_drugs_ml.json"
    if not drug_path.exists():
        logging.warning(f"Drug file not found: {drug_path}. No drug exclusion applied.")
        return set()
    with open(drug_path, "r") as f:
        drug_data = json.load(f)
    drug_smiles = {r["smiles"] for r in drug_data if isinstance(r.get("smiles"), str)}
    return drug_smiles


def process_raw_data(raw_data_dir, save_dir="processed", filter_zero_fg=False):
    """Complete preprocessing pipeline:
    1. Load and merge raw data
    """
    raw_path = Path(raw_data_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Unified Preprocessing Pipeline ===")
    print(f"Raw data directory: {raw_path}")
    print(f"Save directory: {save_path}")

    # Initialize processor
    smiles_processor = SMILESProcessor()

    # Load drug SMILES for exclusion (all 115 drug molecules must be kept out of training)
    drug_smiles = load_drug_smiles(raw_data_dir)
    print(f"  Drug SMILES to exclude: {len(drug_smiles)}")
    
    # Metadata for spectra (not used for processing, just for saving metadata)
    wavenumber_range = (400, 4000)
    n_points = 1250
    
    # Step 1: Load raw data (non-drug molecules only)
    # Drug-like molecules are held out for test data only
    print("\n1. Loading raw data (non-drug molecules only)...")
    src_gas = raw_path / "gas_ml.json"
    src_liq = raw_path / "liq_ml.json"

    # Load gas data (exclude drug molecules and hardcoded exclusions)
    gas_data = load_json(src_gas)
    gas_merged = [r for r in gas_data if is_valid(r) and isinstance(r.get("smiles"), str)
                  and r.get("smiles") not in EXCLUDED_SMILES and r.get("smiles") not in drug_smiles]
    print(f"  Gas data: {len(gas_merged)} samples (non-drug only)")

    # Load liquid data (exclude drug molecules and hardcoded exclusions)
    liq_data = load_json(src_liq)
    liq_merged = [r for r in liq_data if is_valid(r) and isinstance(r.get("smiles"), str)
                  and r.get("smiles") not in EXCLUDED_SMILES and r.get("smiles") not in drug_smiles]
    print(f"  Liquid data: {len(liq_merged)} samples (non-drug only)")
    print(f"  Note: {len(drug_smiles)} drug molecules excluded from training (test holdout)")
    
    # Step 2: Pair gas and liquid data by SMILES
    print("\n2. Pairing gas and liquid data...")
    liq_map = {r["smiles"]: r for r in liq_merged if is_valid(r) and isinstance(r.get("smiles"), str)}
    paired = []
    for g in gas_merged:
        smi = g.get("smiles")
        if smi in liq_map and is_valid(g) and is_valid(liq_map[smi]):
            paired.append({
                "smiles": smi,
                "gas": {"job_uuid": g.get("job_uuid"), "intensity": g.get("intensity")},
                "liq": {"job_uuid": liq_map[smi].get("job_uuid"), "intensity": liq_map[smi].get("intensity")},
            })
    
    # Drop any record containing None (EXCLUDED_SMILES already filtered in gas_merged/liq_merged)
    paired = [p for p in paired if is_valid(p)]
    print(f"  Paired data: {len(paired)} samples")
    
    # Save paired data
    save_json(save_path / "paired_ml.json", paired)
    print(f"  ✓ Saved paired data to {save_path}/paired_ml.json")
    
    # Step 3: Extract functional groups and load spectra
    print("\n3. Extracting functional groups and loading spectra...")
    gas_spectra = []
    liquid_spectra = []
    fg_labels = []
    smiles_list = []
    
    for i, item in enumerate(paired):
        try:
            # Load spectra directly (raw data, no processing)
            gas_spectrum = np.array(item['gas']['intensity'], dtype=np.float32)
            liq_spectrum = np.array(item['liq']['intensity'], dtype=np.float32)
            
            # Validate spectra (check for None/NaN)
            if np.any(np.isnan(gas_spectrum)) or np.any(np.isnan(liq_spectrum)):
                logging.warning(f"Skipping sample {i} due to NaN values in spectrum")
                continue
            
            gas_spectra.append(gas_spectrum)
            liquid_spectra.append(liq_spectrum)
            
            # Extract functional groups
            smiles = item['smiles']
            fg_vector = smiles_processor.extract_functional_groups_with_counts(smiles)
            fg_labels.append(fg_vector)
            smiles_list.append(smiles)

        except Exception as e:
            logging.warning(f"Error processing sample {i}: {e}")
            continue
    
    # Convert to numpy array for filtering
    fg_labels_array = np.array(fg_labels)
    
    # Step 4: Handle samples with 0 functional groups
    print("\n4. Handling samples with 0 functional groups...")
    original_count = len(fg_labels_array)
    if filter_zero_fg:
        non_zero_mask = np.sum(fg_labels_array, axis=1) > 0
        print("  Filtering enabled: removing samples with zero detected functional groups.")
    else:
        non_zero_mask = np.ones(len(fg_labels_array), dtype=bool)
        print("  Filtering disabled: retaining all samples (including zero-FG cases).")
    
    fg_labels_filtered = fg_labels_array[non_zero_mask]
    gas_spectra_filtered = [gas_spectra[i] for i in range(len(gas_spectra)) if non_zero_mask[i]]
    liquid_spectra_filtered = [liquid_spectra[i] for i in range(len(liquid_spectra)) if non_zero_mask[i]]
    smiles_list_filtered = [smiles_list[i] for i in range(len(smiles_list)) if non_zero_mask[i]]
    
    removed_count = original_count - len(fg_labels_filtered)
    removed_pct = (removed_count / original_count * 100) if original_count > 0 else 0.0
    if filter_zero_fg:
        print(f"  Removed {removed_count} samples ({removed_pct:.1f}%) with 0 functional groups")
    else:
        print("  Removed 0 samples (filtering skipped).")
    print(f"  Remaining samples: {len(fg_labels_filtered)}")
    
    # Step 5: Organize groups by tiers
    print("\n5. Organizing functional groups by tiers...")
    all_fg_names = smiles_processor.get_functional_group_names()
    
    # Verify all tier groups exist in the original groups
    all_tier_groups = set(TIER_1 + TIER_2 + TIER_3 + TIER_4)
    missing_groups = all_tier_groups - set(all_fg_names)
    if missing_groups:
        logging.warning(f"Some tier groups not found in original groups: {missing_groups}")

    # Create tier mapping for each group
    tier_mapping = {}
    for name in TIER_1:
        tier_mapping[name] = 1
    for name in TIER_2:
        tier_mapping[name] = 2
    for name in TIER_3:
        tier_mapping[name] = 3
    for name in TIER_4:
        tier_mapping[name] = 4

    # Precompute name-to-index for O(1) lookup (instead of repeated .index() calls)
    fg_name_to_idx = {name: i for i, name in enumerate(all_fg_names)}

    # Create mapping from original indices to new indices for ALL groups (ordered by tier)
    old_to_new_mapping_all = {}
    all_old_indices = []
    for new_idx, group_name in enumerate(ALL_GROUPS):
        if group_name in fg_name_to_idx:
            old_idx = fg_name_to_idx[group_name]
            old_to_new_mapping_all[old_idx] = new_idx
            all_old_indices.append(old_idx)
        else:
            logging.warning(f"Functional group '{group_name}' not found in original groups. Skipping.")

    # Also create mapping for selected groups (for backward compatibility)
    old_to_new_mapping_selected = {}
    selected_old_indices = []
    for new_idx, group_name in enumerate(SELECTED_GROUPS):
        if group_name in fg_name_to_idx:
            old_idx = fg_name_to_idx[group_name]
            old_to_new_mapping_selected[old_idx] = new_idx
            selected_old_indices.append(old_idx)
        else:
            logging.warning(f"Functional group '{group_name}' not found in original groups. Skipping.")
    
    print(f"  Original groups: {len(all_fg_names)}")
    print(f"  All groups (Tier 1-4): {len(ALL_GROUPS)} groups")
    print(f"  Tier 1: {len(TIER_1)} groups")
    print(f"  Tier 2: {len(TIER_2)} groups")
    print(f"  Tier 3: {len(TIER_3)} groups")
    print(f"  Tier 4: {len(TIER_4)} groups")
    print(f"  Selected groups (Tier 1-3): {len(SELECTED_GROUPS)} groups (for backward compatibility)")
    
    # Save all groups in main labels file (ordered by tier: Tier 1, 2, 3, 4)
    fg_labels_all = fg_labels_filtered[:, all_old_indices]
    
    # Also save selected groups version for backward compatibility
    fg_labels_selected = fg_labels_filtered[:, selected_old_indices]
    
    print(f"  All labels shape: {fg_labels_all.shape} (all {len(ALL_GROUPS)} groups)")
    print(f"  Selected labels shape: {fg_labels_selected.shape} (selected {len(SELECTED_GROUPS)} groups)")
    
    # No scaling applied - save raw intensities exactly as loaded
    print(f"\n6. Saving raw intensities (no scaling/normalization)...")
    gas_spectra_array = np.array(gas_spectra_filtered)
    liquid_spectra_array = np.array(liquid_spectra_filtered)
    
    print(f"  Gas spectra range: [{np.min(gas_spectra_array):.6f}, {np.max(gas_spectra_array):.6f}]")
    print(f"  Liquid spectra range: [{np.min(liquid_spectra_array):.6f}, {np.max(liquid_spectra_array):.6f}]")
    
    # Convert to tensors (raw intensities, no modification)
    gas_spectra = torch.tensor(gas_spectra_array, dtype=torch.float32)
    liquid_spectra = torch.tensor(liquid_spectra_array, dtype=torch.float32)
    
    # Use all groups as the main labels file (all 39 groups, ordered by tier)
    fg_labels = torch.tensor(fg_labels_all, dtype=torch.float32)
    
    print(f"  Processed {len(fg_labels)} valid samples (after filtering)")
    
    # Step 7: Save processed data
    print("\n7. Saving processed data...")
    
    # Save tensors - main labels file now contains all 39 groups
    torch.save(gas_spectra, save_path / "gas_spectra.pt")
    torch.save(liquid_spectra, save_path / "liq_spectra.pt")
    torch.save(fg_labels, save_path / "functional_group_labels.pt")
    print(f"  ✓ Saved gas spectra: {gas_spectra.shape}")
    print(f"  ✓ Saved liquid spectra: {liquid_spectra.shape}")
    print(f"  ✓ Saved functional group labels: {fg_labels.shape} (all {len(ALL_GROUPS)} groups)")
    
    # Also save selected groups version for backward compatibility
    fg_labels_selected_tensor = torch.tensor(fg_labels_selected, dtype=torch.float32)
    torch.save(fg_labels_selected_tensor, save_path / "functional_group_labels_selected.pt")
    print(f"  ✓ Saved selected group labels: {fg_labels_selected_tensor.shape} (selected {len(SELECTED_GROUPS)} groups, for backward compatibility)")
    
    # Create and save metadata
    metadata = {
        'n_samples': len(fg_labels),
        'n_functional_groups': len(ALL_GROUPS),
        'n_original_functional_groups': smiles_processor.n_functional_groups,
        'functional_group_names': ALL_GROUPS,  # All groups (Tier 1-4), ordered by tier
        'original_functional_group_names': smiles_processor.get_functional_group_names(),
        'selected_functional_group_names': SELECTED_GROUPS,  # Selected groups (Tier 1-3) for backward compatibility
        'tier_1_groups': TIER_1,
        'tier_2_groups': TIER_2,
        'tier_3_groups': TIER_3,
        'tier_4_groups': TIER_4,
        'tier_mapping': tier_mapping,  # Maps group name -> tier number
        'group_mapping': {str(old_idx): new_idx for old_idx, new_idx in old_to_new_mapping_all.items()},
        'selected_group_mapping': {str(old_idx): new_idx for old_idx, new_idx in old_to_new_mapping_selected.items()},
        'selected_old_indices': selected_old_indices,
        'all_old_indices': all_old_indices,
        'wavenumber_range': wavenumber_range,
        'n_wavenumbers': n_points,
        'smiles_list': smiles_list_filtered,
        'raw_intensities': True,
        'no_scaling': True,
        'standardization_applied_at_training': True,
        'samples_with_zero_groups_removed': removed_count,
        'original_sample_count': original_count,
        'filter_zero_fg_applied': filter_zero_fg
    }
    
    save_json(save_path / "metadata.json", metadata)
    print(f"  ✓ Saved metadata: {metadata['n_functional_groups']} functional groups (all tiers)")
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Final dataset: {len(fg_labels)} samples")
    print(f"Functional groups: {len(ALL_GROUPS)} total (from {smiles_processor.n_functional_groups} original)")
    print(f"  - Tier 1: {len(TIER_1)} groups")
    print(f"  - Tier 2: {len(TIER_2)} groups")
    print(f"  - Tier 3: {len(TIER_3)} groups")
    print(f"  - Tier 4: {len(TIER_4)} groups")
    print(f"  Total: {len(ALL_GROUPS)} groups (all tiers available for training)")
    print(f"Spectral dimensions: {n_points}")
    print(f"Data saved to: {save_path}")
    
    return gas_spectra, liquid_spectra, fg_labels, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified preprocessing for IR spectra data")
    parser.add_argument("raw_data_dir", type=str, help="Directory containing raw gas/liquid JSON files")
    parser.add_argument("--save-dir", type=str, default="processed",
                        help="Directory to save processed tensors/metadata (default: processed)")
    parser.add_argument("--filter-zero-fg", action="store_true",
                        help="Remove samples with zero detected functional groups")
    
    args = parser.parse_args()
    
    print("Starting unified preprocessing...")
    print(f"Raw data: {args.raw_data_dir}")
    print(f"Output: {args.save_dir}")
    print(f"Filter zero-FG samples: {'Yes' if args.filter_zero_fg else 'No'}")

    gas_spectra, liquid_spectra, fg_labels, metadata = process_raw_data(
        args.raw_data_dir, args.save_dir, filter_zero_fg=args.filter_zero_fg
    )

    # Print comprehensive summary for paper
    print("\n")
    print("=" * 80)
    print("PREPROCESS.PY - DATASET SUMMARY FOR PAPER")
    print("=" * 80)

    n_samples = len(fg_labels)
    n_wavenumbers = gas_spectra.shape[1] if gas_spectra.dim() > 1 else 1250

    print(f"\n### Single-Phase Dataset Statistics")
    print(f"")
    print(f"| Statistic | Value |")
    print(f"|-----------|-------|")
    print(f"| Total paired samples | {n_samples:,} |")
    print(f"| Spectrum length | {n_wavenumbers} points |")
    print(f"| Wavenumber range | 400-4000 cm⁻¹ |")
    print(f"| Resolution | {(4000-400)/n_wavenumbers:.2f} cm⁻¹ |")
    print(f"| Functional groups | {len(ALL_GROUPS)} |")

    # Intensity statistics
    gas_np = gas_spectra.numpy()
    liq_np = liquid_spectra.numpy()

    print(f"\n### Raw Intensity Statistics")
    print(f"")
    print(f"| Phase | Min | Max | Mean | Std |")
    print(f"|-------|-----|-----|------|-----|")
    print(f"| Gas | {gas_np.min():.6f} | {gas_np.max():.6f} | {gas_np.mean():.6f} | {gas_np.std():.6f} |")
    print(f"| Liquid | {liq_np.min():.6f} | {liq_np.max():.6f} | {liq_np.mean():.6f} | {liq_np.std():.6f} |")

    # Per-sample max statistics
    gas_maxes = gas_np.max(axis=1)
    liq_maxes = liq_np.max(axis=1)

    print(f"\n### Per-Sample Max Intensity Distribution")
    print(f"")
    print(f"| Phase | Median | 95th pct | 99th pct | Max |")
    print(f"|-------|--------|----------|----------|-----|")
    print(f"| Gas | {np.median(gas_maxes):.6f} | {np.percentile(gas_maxes, 95):.6f} | {np.percentile(gas_maxes, 99):.6f} | {gas_maxes.max():.6f} |")
    print(f"| Liquid | {np.median(liq_maxes):.6f} | {np.percentile(liq_maxes, 95):.6f} | {np.percentile(liq_maxes, 99):.6f} | {liq_maxes.max():.6f} |")

    # Functional group statistics
    fg_np = fg_labels.numpy()
    fg_counts = (fg_np > 0).sum(axis=0)
    fg_per_sample = (fg_np > 0).sum(axis=1)

    print(f"\n### Functional Group Statistics")
    print(f"")
    print(f"| Statistic | Value |")
    print(f"|-----------|-------|")
    print(f"| Groups per sample (mean) | {fg_per_sample.mean():.2f} |")
    print(f"| Groups per sample (median) | {np.median(fg_per_sample):.0f} |")
    print(f"| Groups per sample (range) | {fg_per_sample.min():.0f} - {fg_per_sample.max():.0f} |")
    print(f"| Samples with 0 groups | {(fg_per_sample == 0).sum()} |")

    # Top functional groups
    print(f"\n### Top 10 Functional Groups by Frequency")
    print(f"")
    print(f"| Rank | Group | Count | Percentage |")
    print(f"|------|-------|-------|------------|")
    sorted_idx = np.argsort(fg_counts)[::-1]
    for rank, idx in enumerate(sorted_idx[:10], 1):
        group_name = ALL_GROUPS[idx]
        count = int(fg_counts[idx])
        pct = 100.0 * count / n_samples
        print(f"| {rank} | {group_name} | {count:,} | {pct:.1f}% |")

    print(f"\n" + "=" * 80)
    print("✅ Preprocessing complete!")