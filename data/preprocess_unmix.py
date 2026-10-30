#!/usr/bin/env python3
"""Create training instances for mixture spectrum unmixing with optimized storage."""

import json
import torch
import numpy as np
import h5py
from typing import Dict, List
import os
from tqdm import tqdm

from preprocess import (
    is_valid, EXCLUDED_SMILES, load_drug_smiles, SMILESProcessor, ALL_GROUPS
)


def reorder_fg_vector(fg_vector: np.ndarray, original_names: List[str], target_names: List[str]) -> np.ndarray:
    """Reorder a functional group vector from original ordering to target ordering."""
    name_to_idx = {name: i for i, name in enumerate(original_names)}
    reordered = np.zeros(len(target_names), dtype=np.float32)
    for new_idx, group_name in enumerate(target_names):
        if group_name in name_to_idx:
            reordered[new_idx] = fg_vector[name_to_idx[group_name]]
    return reordered


def load_processed_data():
    """Load the processed single-phase data (NON-DRUG molecules only)."""
    print("Loading processed data...")

    # Load drug SMILES to exclude from training database
    drug_smiles = load_drug_smiles('250716')

    smiles_processor = SMILESProcessor()
    
    # Load paired data (single-phase spectra) - contains RAW intensity data
    paired_data = json.load(open('processed/paired_ml.json'))
    
    # Load functional group labels
    functional_group_labels = torch.load('processed/functional_group_labels.pt')
    
    # Load metadata for functional group names
    metadata = json.load(open('processed/metadata.json'))
    functional_group_names = metadata['functional_group_names']
    
    # Create lookup dictionaries
    # Use RAW spectra from paired_ml.json for unmixing (preserves physical relationships)
    smiles_to_index = {}
    smiles_to_gas_spectrum = {}
    smiles_to_liq_spectrum = {}
    smiles_to_functional_groups = {}
    
    # Note: Some samples in paired_data may have been filtered out during .pt file creation
    # (if baseline correction or processing failed). We need to match by SMILES.
    # Use smiles_list from metadata which matches the .pt file indices
    processed_smiles_list = metadata['smiles_list']
    print(f"Using SMILES list from metadata ({len(processed_smiles_list)} items)")
    
    # Create mapping from SMILES to paired_data index
    paired_smiles_to_idx = {record['smiles']: i for i, record in enumerate(paired_data)}
    
    excluded_count = 0
    drug_excluded_count = 0
    paired_count = 0
    for pt_idx, smiles in enumerate(processed_smiles_list):
        # Exclude specific SMILES
        if smiles in EXCLUDED_SMILES:
            excluded_count += 1
            continue
        # Exclude drug molecules from training database
        if smiles in drug_smiles:
            drug_excluded_count += 1
            continue
        if smiles in paired_smiles_to_idx:
            paired_idx = paired_smiles_to_idx[smiles]
            paired_record = paired_data[paired_idx]
            
            # Validate record (check for None values)
            if not is_valid(paired_record):
                continue
            
            # Validate gas and liquid data
            gas_intensity = paired_record.get('gas', {}).get('intensity')
            liq_intensity = paired_record.get('liq', {}).get('intensity')
            
            if gas_intensity is None or liq_intensity is None:
                continue
            
            # Convert to numpy arrays
            gas_spectrum = np.array(gas_intensity, dtype=np.float32)
            liq_spectrum = np.array(liq_intensity, dtype=np.float32)
            
            # Check for NaN or empty arrays
            if (np.any(np.isnan(gas_spectrum)) or np.any(np.isnan(liq_spectrum)) or
                len(gas_spectrum) == 0 or len(liq_spectrum) == 0):
                continue
            
            smiles_to_index[smiles] = pt_idx
            smiles_to_gas_spectrum[smiles] = gas_spectrum
            smiles_to_liq_spectrum[smiles] = liq_spectrum
            smiles_to_functional_groups[smiles] = functional_group_labels[pt_idx].numpy()
            paired_count += 1
    
    if excluded_count > 0:
        print(f"Excluded {excluded_count} molecules with hardcoded excluded SMILES from database")
    if drug_excluded_count > 0:
        print(f"Excluded {drug_excluded_count} drug molecules from paired training data")

    print(f"Loaded {paired_count} paired (gas+liquid) records")
    
    # Load liquid-only molecules from liq_ml.json for mixture datasets
    # These are needed for ternary mixtures where some components only have liquid spectra
    liquid_only_count = 0
    try:
        liq_data = json.load(open('250716/liq_ml.json'))
        print(f"Loading liquid-only molecules from liq_ml.json...")
        
        # Get spectrum length from first paired molecule (if available) or from liquid data
        if smiles_to_liq_spectrum:
            spectrum_length = len(list(smiles_to_liq_spectrum.values())[0])
        elif liq_data and liq_data[0].get('intensity'):
            spectrum_length = len(liq_data[0]['intensity'])
        else:
            spectrum_length = 1250  # Default fallback
        
        # Create mapping from SMILES to liq_data index (with validation)
        liq_smiles_to_idx = {}
        for i, record in enumerate(liq_data):
            # Validate record
            if not is_valid(record):
                continue
            
            smiles = record.get('smiles')
            intensity = record.get('intensity')
            
            # Check for valid SMILES and intensity
            if not isinstance(smiles, str) or intensity is None:
                continue
            
            # Validate intensity is not empty
            if len(intensity) == 0:
                continue
            
            liq_smiles_to_idx[smiles] = i
        
        # Add liquid-only molecules (not already in paired data, not drug molecules)
        for smiles, liq_idx in liq_smiles_to_idx.items():
            if smiles in EXCLUDED_SMILES:
                continue
            if smiles in drug_smiles:
                continue
            if smiles not in smiles_to_index:  # Not in paired data
                record = liq_data[liq_idx]
                intensity = record.get('intensity')
                
                # Convert to numpy array and validate
                liq_spectrum = np.array(intensity, dtype=np.float32)
                
                # Check for NaN or invalid data
                if np.any(np.isnan(liq_spectrum)) or len(liq_spectrum) == 0:
                    continue
                
                # Validate spectrum length matches expected
                if len(liq_spectrum) != spectrum_length:
                    continue
                
                # Assign a unique index (starting after paired indices)
                new_idx = len(smiles_to_index)
                smiles_to_index[smiles] = new_idx
                smiles_to_liq_spectrum[smiles] = liq_spectrum

                # Gas spectrum: use zeros (not available, but needed for database structure)
                gas_spectrum = np.zeros(spectrum_length, dtype=np.float32)
                smiles_to_gas_spectrum[smiles] = gas_spectrum

                # Functional groups: compute from SMILES
                fg_vector = smiles_processor.extract_functional_groups_with_counts(smiles)
                # Reorder to match the tier-based ordering used in functional_group_names
                fg_reordered = reorder_fg_vector(
                    fg_vector, smiles_processor.get_functional_group_names(), functional_group_names
                )
                smiles_to_functional_groups[smiles] = fg_reordered

                liquid_only_count += 1
        
        print(f"Added {liquid_only_count} liquid-only molecules (for mixture datasets)")
    except FileNotFoundError:
        print("Warning: liq_ml.json not found, skipping liquid-only molecules")
    except Exception as e:
        print(f"Warning: Error loading liquid-only molecules: {e}")
    
    print(f"Using raw intensities (no scaling applied)")
    print(f"Total loaded: {len(smiles_to_index)} molecules ({paired_count} paired + {liquid_only_count} liquid-only)")
    print(f"NOTE: Drug molecules are NOT included (loaded separately for OOD test)")
    print(f"Functional group names: {functional_group_names}")

    return {
        'smiles_to_index': smiles_to_index,
        'smiles_to_gas_spectrum': smiles_to_gas_spectrum,
        'smiles_to_liq_spectrum': smiles_to_liq_spectrum,
        'smiles_to_functional_groups': smiles_to_functional_groups,
        'functional_group_names': functional_group_names,
    }


def load_drug_molecules(functional_group_names: List[str], spectrum_length: int = 1250) -> Dict:
    """Load drug-like molecules separately for OOD test evaluation."""
    print("Loading drug molecules for OOD test evaluation...")

    smiles_processor = SMILESProcessor()

    smiles_to_index = {}
    smiles_to_gas_spectrum = {}
    smiles_to_liq_spectrum = {}
    smiles_to_functional_groups = {}

    try:
        gas_drugs_data = json.load(open('250716/gas_drugs_ml.json'))
        liq_drugs_data = json.load(open('250716/liq_drugs_ml.json'))

        # Create mapping from drug SMILES to data
        gas_drugs_map = {}
        for record in gas_drugs_data:
            if not is_valid(record):
                continue
            smiles = record.get('smiles')
            if isinstance(smiles, str) and smiles not in EXCLUDED_SMILES:
                gas_drugs_map[smiles] = record

        liq_drugs_map = {}
        for record in liq_drugs_data:
            if not is_valid(record):
                continue
            smiles = record.get('smiles')
            if isinstance(smiles, str) and smiles not in EXCLUDED_SMILES:
                liq_drugs_map[smiles] = record

        # Add drug molecules (require liquid spectrum for unmixing)
        all_drug_smiles = set(gas_drugs_map.keys()) | set(liq_drugs_map.keys())
        drug_count = 0

        for smiles in all_drug_smiles:
            # Check if we have liquid spectrum (required for unmixing)
            if smiles not in liq_drugs_map:
                continue

            liq_record = liq_drugs_map[smiles]
            liq_intensity = liq_record.get('intensity')
            if liq_intensity is None:
                continue

            liq_spectrum = np.array(liq_intensity, dtype=np.float32)
            if np.any(np.isnan(liq_spectrum)) or len(liq_spectrum) != spectrum_length:
                continue

            # Get gas spectrum if available, otherwise use zeros
            if smiles in gas_drugs_map:
                gas_record = gas_drugs_map[smiles]
                gas_intensity = gas_record.get('intensity')
                if gas_intensity is not None:
                    gas_spectrum = np.array(gas_intensity, dtype=np.float32)
                    if np.any(np.isnan(gas_spectrum)) or len(gas_spectrum) != spectrum_length:
                        gas_spectrum = np.zeros(spectrum_length, dtype=np.float32)
                else:
                    gas_spectrum = np.zeros(spectrum_length, dtype=np.float32)
            else:
                gas_spectrum = np.zeros(spectrum_length, dtype=np.float32)

            # Assign index and store
            new_idx = len(smiles_to_index)
            smiles_to_index[smiles] = new_idx
            smiles_to_gas_spectrum[smiles] = gas_spectrum
            smiles_to_liq_spectrum[smiles] = liq_spectrum

            # Compute functional groups from SMILES
            fg_vector = smiles_processor.extract_functional_groups_with_counts(smiles)
            fg_reordered = reorder_fg_vector(
                fg_vector, smiles_processor.get_functional_group_names(), functional_group_names
            )
            smiles_to_functional_groups[smiles] = fg_reordered
            drug_count += 1

        print(f"Loaded {drug_count} drug molecules for OOD test")

    except FileNotFoundError as e:
        print(f"Warning: Drug data files not found ({e})")
    except Exception as e:
        print(f"Warning: Error loading drug molecules: {e}")

    return {
        'smiles_to_index': smiles_to_index,
        'smiles_to_gas_spectrum': smiles_to_gas_spectrum,
        'smiles_to_liq_spectrum': smiles_to_liq_spectrum,
        'smiles_to_functional_groups': smiles_to_functional_groups,
        'functional_group_names': functional_group_names,
    }


def create_molecule_spectra_database(processed_data: Dict, output_file: str, database_name: str = "training"):
    """Create a database of individual molecule spectra."""
    print(f"Creating {database_name} molecule spectra database...")

    smiles_list = list(processed_data['smiles_to_gas_spectrum'].keys())
    if len(smiles_list) == 0:
        print(f"  Warning: No molecules to store in {database_name} database")
        return smiles_list

    spectrum_length = len(processed_data['smiles_to_gas_spectrum'][smiles_list[0]])
    num_functional_groups = len(processed_data['functional_group_names'])

    with h5py.File(output_file, 'w') as f:
        # Store metadata
        f.attrs['num_molecules'] = len(smiles_list)
        f.attrs['spectrum_length'] = spectrum_length
        f.attrs['functional_group_names'] = json.dumps(processed_data['functional_group_names'])
        f.attrs['raw_intensities'] = True
        f.attrs['no_scaling'] = True
        f.attrs['database_type'] = database_name

        # Create datasets
        smiles_dataset = f.create_dataset('smiles',
                                        shape=(len(smiles_list),),
                                        dtype=h5py.string_dtype(encoding='utf-8'))

        gas_spectra = f.create_dataset('gas_spectra',
                                     shape=(len(smiles_list), spectrum_length),
                                     dtype='float32')

        liq_spectra = f.create_dataset('liq_spectra',
                                     shape=(len(smiles_list), spectrum_length),
                                     dtype='float32')

        functional_groups = f.create_dataset('functional_groups',
                                           shape=(len(smiles_list), num_functional_groups),
                                           dtype='float32')

        # Fill datasets
        for i, smiles in enumerate(tqdm(smiles_list, desc=f"Storing {database_name} molecule spectra")):
            smiles_dataset[i] = smiles
            gas_spectra[i] = np.array(processed_data['smiles_to_gas_spectrum'][smiles], dtype=np.float32)
            liq_spectra[i] = np.array(processed_data['smiles_to_liq_spectrum'][smiles], dtype=np.float32)
            functional_groups[i] = np.array(processed_data['smiles_to_functional_groups'][smiles], dtype=np.float32)

    print(f"Created {database_name} database: {len(smiles_list)} molecules")
    return smiles_list


def create_mixture_training_instances(mixture_file: str, molecule_db_file: str, output_file: str, is_test_only: bool = False):
    """Create training instances for mixture data using molecule database."""
    print(f"Processing {mixture_file}...")
    
    # Load mixture data
    mixture_data = json.load(open(mixture_file))
    
    # Load molecule database
    with h5py.File(molecule_db_file, 'r') as f:
        smiles_list = f['smiles'][:].astype(str).tolist()
        smiles_to_index = {smiles: i for i, smiles in enumerate(smiles_list)}
        functional_group_names = json.loads(f.attrs['functional_group_names'])
    
    # Filter mixture data to only include molecules with single-phase data
    valid_instances = []
    skipped_count = 0
    excluded_count = 0
    num_components = None  # Will be determined from first valid instance
    
    for record in tqdm(mixture_data, desc="Processing mixture records"):
        # Validate record (check for None values)
        if not is_valid(record):
            skipped_count += 1
            continue
        
        # Check for valid SMILES
        component_smiles_list = record.get('smiles')
        if component_smiles_list is None or not isinstance(component_smiles_list, list) or len(component_smiles_list) == 0:
            skipped_count += 1
            continue

        # Exclude mixtures containing excluded SMILES
        if any(smiles in EXCLUDED_SMILES for smiles in component_smiles_list):
            excluded_count += 1
            continue

        # Validate intensity data
        intensity = record.get('intensity')
        if intensity is None:
            skipped_count += 1
            continue

        # Convert to numpy array and validate
        mixture_spectrum = np.array(intensity, dtype=np.float32)

        # Check for NaN or empty spectrum
        if np.any(np.isnan(mixture_spectrum)) or len(mixture_spectrum) == 0:
            skipped_count += 1
            continue

        # Check if all components have single-phase data
        all_components_available = True
        component_indices = []

        for smiles in component_smiles_list:
            if not isinstance(smiles, str):
                all_components_available = False
                break
            if smiles in smiles_to_index:
                component_indices.append(smiles_to_index[smiles])
            else:
                all_components_available = False
                break

        if all_components_available:
            # Determine number of components from first valid instance
            if num_components is None:
                num_components = len(component_indices)
                print(f"  Detected {num_components} components per mixture")

            valid_instances.append({
                'mixture_spectrum': mixture_spectrum,
                'component_indices': component_indices,
                'component_smiles': component_smiles_list
            })
        else:
            skipped_count += 1

    if excluded_count > 0:
        print(f"Excluded {excluded_count} mixture records containing excluded SMILES")

    # Deduplicate: keep only the first instance per unique combination
    seen_combos = set()
    unique_instances = []
    duplicate_count = 0
    for inst in valid_instances:
        combo_key = tuple(sorted(inst['component_indices']))
        if combo_key not in seen_combos:
            seen_combos.add(combo_key)
            unique_instances.append(inst)
        else:
            duplicate_count += 1
    valid_instances = unique_instances

    print(f"Valid instances before dedup: {len(valid_instances) + duplicate_count}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Unique instances: {len(valid_instances)}")
    print(f"Skipped {skipped_count} records (missing single-phase data)")
    
    # Handle empty dataset case
    if len(valid_instances) == 0:
        print(f"No valid instances found, creating empty dataset...")
        # Default to 2 components for empty dataset (backward compatibility)
        num_components = num_components or 2
        with h5py.File(output_file, 'w') as f:
            f.attrs['num_instances'] = 0
            f.attrs['molecule_db_file'] = molecule_db_file
            f.attrs['functional_group_names'] = json.dumps(functional_group_names)
            f.attrs['raw_intensities'] = True
            f.attrs['no_scaling'] = True
            f.attrs['num_components'] = num_components
            f.attrs['is_test_only'] = is_test_only
            # Create empty datasets
            f.create_dataset('mixture_spectra', shape=(0, 1250), dtype='float32', maxshape=(None, 1250))
            f.create_dataset('component_indices', shape=(0, num_components), dtype='int32', maxshape=(None, num_components))
            f.create_dataset('component_smiles', shape=(0, num_components), dtype=h5py.string_dtype(encoding='utf-8'), maxshape=(None, num_components))
        print(f"Saved empty {'test' if is_test_only else 'training'} instances to {output_file}")
        return valid_instances

    # Save training instances
    with h5py.File(output_file, 'w') as f:
        # Store metadata
        f.attrs['num_instances'] = len(valid_instances)
        f.attrs['molecule_db_file'] = molecule_db_file
        f.attrs['functional_group_names'] = json.dumps(functional_group_names)
        f.attrs['raw_intensities'] = True
        f.attrs['no_scaling'] = True
        f.attrs['num_components'] = num_components
        f.attrs['is_test_only'] = is_test_only
        
        # Create datasets
        num_instances = len(valid_instances)
        spectrum_length = len(valid_instances[0]['mixture_spectrum'])
        
        # Mixture spectra
        mixture_spectra = f.create_dataset('mixture_spectra', 
                                         shape=(num_instances, spectrum_length), 
                                         dtype='float32')
        
        # Component references (indices into molecule database)
        # Shape is dynamic based on number of components detected
        component_indices = f.create_dataset('component_indices', 
                                            shape=(num_instances, num_components), 
                                            dtype='int32')
        
        # Component SMILES (for reference)
        component_smiles = f.create_dataset('component_smiles', 
                                           shape=(num_instances, num_components), 
                                           dtype=h5py.string_dtype(encoding='utf-8'))
        
        # Fill datasets
        for i, instance in enumerate(tqdm(valid_instances, desc="Writing training instances")):
            mixture_spectra[i] = instance['mixture_spectrum']
            
            # Fill component data (supports variable component counts)
            for j in range(num_components):
                component_indices[i, j] = instance['component_indices'][j]
                component_smiles[i, j] = instance['component_smiles'][j]
    
    split_type = "test" if is_test_only else "training"
    print(f"Saved {split_type} instances to {output_file}")
    print(f"  Instances: {num_instances}, Components per instance: {num_components}")
    return valid_instances


def analyze_storage_efficiency(molecule_db_file: str, training_files: List[str]):
    """Analyze storage efficiency of the optimized approach."""
    print("\n=== STORAGE EFFICIENCY ANALYSIS ===")
    
    # Molecule database size
    mol_db_size = os.path.getsize(molecule_db_file) / (1024 * 1024)  # MB
    print(f"Molecule database size: {mol_db_size:.1f} MB")
    
    total_training_size = 0
    total_instances = 0
    
    for training_file in training_files:
        if os.path.exists(training_file):
            size_mb = os.path.getsize(training_file) / (1024 * 1024)
            total_training_size += size_mb
            
            with h5py.File(training_file, 'r') as f:
                num_instances = f.attrs['num_instances']
                total_instances += num_instances
                print(f"{training_file}: {num_instances} instances, {size_mb:.1f} MB")
    
    total_size = mol_db_size + total_training_size
    print(f"\nTotal storage: {total_size:.1f} MB")
    print(f"Total training instances: {total_instances}")
    print(f"Average per instance: {total_size/total_instances*1024:.1f} KB")
    
    # Compare with naive approach (estimate)
    spectrum_length = 1250
    functional_groups_length = 39
    naive_size_per_instance = (3 * spectrum_length + functional_groups_length) * 4 / 1024  # KB
    naive_total_size = naive_size_per_instance * total_instances / 1024  # MB
    
    print(f"\nEstimated naive storage: {naive_total_size:.1f} MB")
    print(f"Storage efficiency: {naive_total_size/total_size:.1f}x reduction")


def main():
    """Main function to create optimized training instances."""
    print("Creating optimized training instances for mixture spectrum unmixing...")
    print("\n" + "=" * 60)
    print("DATA ORGANIZATION")
    print("=" * 60)
    print("1. Training database: Non-drug molecules only (gas_ml + liq_ml)")
    print("2. Drug database: Drug molecules only (for OOD test lookup)")
    print("3. Mixtures: mix_ml (train), mix_ternary_ml (train), mix_drugs_ml (test)")
    print("=" * 60)

    os.makedirs('processed_unmix', exist_ok=True)

    # Step 1: Load and create NON-DRUG molecule database (for training)
    print("\n[1/4] Creating training molecule database (non-drug only)...")
    processed_data = load_processed_data()
    molecule_db_file = 'processed_unmix/molecule_spectra_database.h5'
    smiles_list = create_molecule_spectra_database(processed_data, molecule_db_file, "training")

    # Step 2: Load and create DRUG molecule database (for OOD test)
    print("\n[2/4] Creating drug molecule database (for OOD test)...")
    drug_db_file = 'processed_unmix/drug_molecule_database.h5'
    drug_data = load_drug_molecules(processed_data['functional_group_names'])
    drug_smiles_list = create_molecule_spectra_database(drug_data, drug_db_file, "drug_test")

    # Step 3: Create combined database for mix_drugs_ml (contains both non-drug and drug molecules)
    # This is needed because mix_drugs_ml may contain mixtures of drug + non-drug molecules
    print("\n[3/4] Creating combined database for OOD test...")
    combined_db_file = 'processed_unmix/combined_molecule_database.h5'
    combined_data = {
        'smiles_to_index': {},
        'smiles_to_gas_spectrum': {},
        'smiles_to_liq_spectrum': {},
        'smiles_to_functional_groups': {},
        'functional_group_names': processed_data['functional_group_names'],
    }
    # Add non-drug molecules first
    for smiles in processed_data['smiles_to_index']:
        idx = len(combined_data['smiles_to_index'])
        combined_data['smiles_to_index'][smiles] = idx
        combined_data['smiles_to_gas_spectrum'][smiles] = processed_data['smiles_to_gas_spectrum'][smiles]
        combined_data['smiles_to_liq_spectrum'][smiles] = processed_data['smiles_to_liq_spectrum'][smiles]
        combined_data['smiles_to_functional_groups'][smiles] = processed_data['smiles_to_functional_groups'][smiles]
    # Add drug molecules
    for smiles in drug_data['smiles_to_index']:
        if smiles not in combined_data['smiles_to_index']:
            idx = len(combined_data['smiles_to_index'])
            combined_data['smiles_to_index'][smiles] = idx
            combined_data['smiles_to_gas_spectrum'][smiles] = drug_data['smiles_to_gas_spectrum'][smiles]
            combined_data['smiles_to_liq_spectrum'][smiles] = drug_data['smiles_to_liq_spectrum'][smiles]
            combined_data['smiles_to_functional_groups'][smiles] = drug_data['smiles_to_functional_groups'][smiles]
    create_molecule_spectra_database(combined_data, combined_db_file, "combined")

    # Step 4: Process mixture datasets
    print("\n[4/4] Processing mixture datasets...")

    # Training mixtures use the training database (non-drug only)
    # OOD test mixtures use the combined database (non-drug + drug)
    datasets = [
        # (mixture_file, output_file, molecule_db, is_test_only)
        ('250716/mix_ml.json', 'processed_unmix/mix_ml_instances.h5', molecule_db_file, False),
        ('250716/mix_ternary_ml.json', 'processed_unmix/mix_ternary_ml_instances.h5', molecule_db_file, False),
        ('250716/mix_drugs_ml.json', 'processed_unmix/mix_drugs_ml_instances.h5', combined_db_file, True),
    ]

    training_files = []

    for mixture_file, output_file, db_file, is_test_only in datasets:
        if os.path.exists(mixture_file):
            split_type = "TEST ONLY" if is_test_only else "TRAINING"
            print(f"\n[{split_type}] {mixture_file}")
            print(f"  Using molecule database: {db_file}")
            training_instances = create_mixture_training_instances(mixture_file, db_file, output_file, is_test_only)
            training_files.append(output_file)
        else:
            print(f"Warning: {mixture_file} not found, skipping...")

    # Analyze storage efficiency
    analyze_storage_efficiency(molecule_db_file, training_files)

    # Print comprehensive summary for paper
    print_paper_summary(
        molecule_db_file, drug_db_file, combined_db_file,
        len(smiles_list), len(drug_smiles_list), len(combined_data['smiles_to_index']),
        training_files
    )


def print_paper_summary(
    molecule_db_file: str,
    drug_db_file: str,
    combined_db_file: str,
    n_training_molecules: int,
    n_drug_molecules: int,
    n_combined_molecules: int,
    training_files: List[str]
):
    """Print comprehensive summary statistics for paper."""

    print("\n")
    print("=" * 80)
    print("PREPROCESS_UNMIX.PY - DATASET SUMMARY FOR PAPER")
    print("=" * 80)

    # Load data for statistics
    print("\nLoading data for statistics...")

    with h5py.File(molecule_db_file, 'r') as f:
        train_gas = f['gas_spectra'][:]
        train_liq = f['liq_spectra'][:]
        n_train = f.attrs['num_molecules']

    with h5py.File(drug_db_file, 'r') as f:
        drug_liq = f['liq_spectra'][:]
        n_drug = f.attrs['num_molecules']

    # Load mixture datasets
    mixture_stats = {}
    for training_file in training_files:
        if os.path.exists(training_file):
            with h5py.File(training_file, 'r') as f:
                mix_spectra = f['mixture_spectra'][:]
                n_instances = f.attrs['num_instances']
                n_components = f.attrs.get('num_components', 2)
                is_test_only = f.attrs.get('is_test_only', False)
                mixture_stats[training_file] = {
                    'spectra': mix_spectra,
                    'n_instances': n_instances,
                    'n_components': n_components,
                    'is_test_only': is_test_only
                }

    print(f"\n### Molecule Database Statistics")
    print(f"")
    print(f"| Database | Molecules | Purpose |")
    print(f"|----------|-----------|---------|")
    print(f"| Training (non-drug) | {n_training_molecules:,} | Spectral library for baselines |")
    print(f"| Drug (OOD test) | {n_drug_molecules:,} | OOD evaluation reference |")
    print(f"| Combined | {n_combined_molecules:,} | Drug mixture component lookup |")

    print(f"\n### Mixture Dataset Statistics")
    print(f"")
    print(f"| Dataset | Instances | Components | Type |")
    print(f"|---------|-----------|------------|------|")
    for filepath, stats in mixture_stats.items():
        name = os.path.basename(filepath).replace('_instances.h5', '')
        dtype = "Test only" if stats['is_test_only'] else "Train/Val/Test"
        print(f"| {name} | {stats['n_instances']:,} | {stats['n_components']} | {dtype} |")

    # Compute intensity statistics
    print(f"\n### Raw Intensity Statistics (Single-Phase)")
    print(f"")
    print(f"| Data | Min | Max | Mean |")
    print(f"|------|-----|-----|------|")
    print(f"| Training Gas | {train_gas.min():.6f} | {train_gas.max():.6f} | {train_gas.mean():.6f} |")
    print(f"| Training Liquid | {train_liq.min():.6f} | {train_liq.max():.6f} | {train_liq.mean():.6f} |")
    print(f"| Drug Liquid | {drug_liq.min():.6f} | {drug_liq.max():.6f} | {drug_liq.mean():.6f} |")

    print(f"\n### Raw Intensity Statistics (Mixtures)")
    print(f"")
    print(f"| Dataset | Min | Max | Mean |")
    print(f"|---------|-----|-----|------|")
    for filepath, stats in mixture_stats.items():
        name = os.path.basename(filepath).replace('_instances.h5', '')
        spectra = stats['spectra']
        print(f"| {name} | {spectra.min():.6f} | {spectra.max():.6f} | {spectra.mean():.6f} |")

    # Compute global scale factor (same logic as src/data.py)
    print(f"\n### Global Scaling Parameters")

    # Single-phase maxes
    gas_maxes = train_gas.max(axis=1)
    liq_maxes = train_liq.max(axis=1)
    single_phase_maxes = np.maximum(gas_maxes, liq_maxes)

    # Mixture maxes
    all_maxes = [single_phase_maxes]
    for filepath, stats in mixture_stats.items():
        mix_maxes = stats['spectra'].max(axis=1)
        all_maxes.append(mix_maxes)

    combined_maxes = np.concatenate(all_maxes)
    threshold = np.percentile(combined_maxes, 99)
    scale_factor = 1.0 / threshold if threshold > 0 else 1.0

    print(f"")
    print(f"| Parameter | Value |")
    print(f"|-----------|-------|")
    print(f"| 99th percentile threshold | {threshold:.6f} |")
    print(f"| Global scale factor | {scale_factor:.6f} |")
    print(f"| Target range after scaling | [0, 1] (clamped) |")
    print(f"| Total samples for scaling | {len(combined_maxes):,} |")

    # After scaling statistics
    print(f"\n### After Scaling + Clamping (max=1.0)")
    print(f"")
    print(f"| Data | Min | Max | Mean | % Clamped |")
    print(f"|------|-----|-----|------|-----------|")

    def scaled_stats(data, name):
        scaled = np.clip(data * scale_factor, 0, 1.0)
        n_clamped = (data * scale_factor > 1.0).sum()
        pct_clamped = 100.0 * n_clamped / data.size
        return f"| {name} | {scaled.min():.6f} | {scaled.max():.6f} | {scaled.mean():.6f} | {pct_clamped:.2f}% |"

    print(scaled_stats(train_gas, "Training Gas"))
    print(scaled_stats(train_liq, "Training Liquid"))
    print(scaled_stats(drug_liq, "Drug Liquid"))
    for filepath, stats in mixture_stats.items():
        name = os.path.basename(filepath).replace('_instances.h5', '')
        print(scaled_stats(stats['spectra'], name))

    # Per-sample max distribution
    print(f"\n### Per-Sample Max Intensity Percentiles (Raw)")
    print(f"")
    print(f"| Data | 50th | 75th | 95th | 99th | Max |")
    print(f"|------|------|------|------|------|-----|")

    def percentile_row(maxes, name):
        return f"| {name} | {np.percentile(maxes, 50):.4f} | {np.percentile(maxes, 75):.4f} | {np.percentile(maxes, 95):.4f} | {np.percentile(maxes, 99):.4f} | {maxes.max():.4f} |"

    print(percentile_row(single_phase_maxes, "Single-phase"))
    for filepath, stats in mixture_stats.items():
        name = os.path.basename(filepath).replace('_instances.h5', '')
        mix_maxes = stats['spectra'].max(axis=1)
        print(percentile_row(mix_maxes, name))

    print(f"\n" + "=" * 80)
    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    main()