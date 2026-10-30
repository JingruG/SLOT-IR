#!/usr/bin/env python3
"""Per-molecule evaluation by molecule visibility category (used by run_holdout.py).

Usage (from slot_ir/):
    python -m analysis.generalization_test --checkpoint CKPT --config configs/generalization/stage2_binary_holdout.json --output_dir OUT --seed 42
    python -m analysis.generalization_test aggregate --input_dirs OUT1 OUT2 --output_dir OUT_AGG
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import h5py
import numpy as np
import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import (
    get_global_scale_factor,
    split_indices,
    compute_molecule_holdout,
)
from src.evaluate import _load_model, _per_sample_metrics
from src.matching import LiquidIRDatabase, RefinementMatcher

# Category names ordered by expected difficulty
BINARY_CATEGORY_ORDER = [
    "S12_S12", "S12_S1", "S1_S1", "S12_U", "S1_U", "U_U",
]

TERNARY_CATEGORY_ORDER = [
    "S12_S12_S12", "S12_S12_S1", "S12_S1_S1", "S1_S1_S1",
    "S12_S12_U", "S12_S1_U", "S1_S1_U",
    "S12_U_U", "S1_U_U", "U_U_U",
]


def _make_labels(order):
    return {cat: f"({', '.join(cat.split('_'))})" for cat in order}


BINARY_CATEGORY_LABELS = _make_labels(BINARY_CATEGORY_ORDER)
TERNARY_CATEGORY_LABELS = _make_labels(TERNARY_CATEGORY_ORDER)


# ---------------------------------------------------------------------------
# Step 1: Molecule categorization
# ---------------------------------------------------------------------------

def build_molecule_categories(
    data_dir: Path,
    n_holdout: int = 0,
    seed: int = 42,
    include_drug_data: bool = False,
) -> Dict[str, Set[int]]:
    """Categorize every molecule index into S12, S1, or U."""
    # Load molecule_spectra_database to get Stage 1 molecule count
    mol_db_path = data_dir / "molecule_spectra_database.h5"
    with h5py.File(mol_db_path, "r") as f:
        n_stage1 = f["liq_spectra"].shape[0]

    # Load mix_ml_instances.h5 and identify training split molecule indices
    mix_path = data_dir / "mix_ml_instances.h5"
    with h5py.File(mix_path, "r") as f:
        component_indices = f["component_indices"][:]
        n_mix = component_indices.shape[0]

    train_idx, _, _ = split_indices(n_mix, seed=seed)

    # All molecules appearing in training instances (before holdout)
    all_train_mols = set()
    for idx in train_idx:
        for mol_idx in component_indices[idx]:
            all_train_mols.add(int(mol_idx))

    # Apply molecule holdout
    if n_holdout > 0:
        from src.config import DataConfig
        # Build a minimal DataConfig pointing to this data_dir
        tmp_cfg = DataConfig(processed_unmix_dir=data_dir, n_holdout_molecules=n_holdout)
        kept, holdout = compute_molecule_holdout(tmp_cfg, train_idx, seed=seed)
        s12 = kept
    else:
        s12 = all_train_mols

    # Include drug training molecules in S12 when drug data is used in training
    if include_drug_data:
        drug_mix_path = data_dir / "mix_drugs_ml_instances.h5"
        with h5py.File(drug_mix_path, "r") as f:
            drug_comp_indices = f["component_indices"][:]
            n_drug_mix = drug_comp_indices.shape[0]
        drug_train_idx, _, _ = split_indices(n_drug_mix, seed=seed)
        drug_train_mols = set()
        for idx in drug_train_idx:
            for mol_idx in drug_comp_indices[idx]:
                drug_train_mols.add(int(mol_idx))
        s12 = s12 | drug_train_mols

    # S1: Stage 1 molecules NOT in S12
    all_stage1 = set(range(n_stage1))
    s1 = all_stage1 - s12

    # U: combined_molecule_database indices >= n_stage1 (drug-only molecules)
    # not seen in Stage 2 training
    combined_db_path = data_dir / "combined_molecule_database.h5"
    with h5py.File(combined_db_path, "r") as f:
        n_combined = f["liq_spectra"].shape[0]
    u = set(range(n_stage1, n_combined)) - s12

    print(f"Molecule categorization (n_holdout={n_holdout}, include_drug_data={include_drug_data}):")
    print(f"  S12 (seen in Stage 1+2 training):       {len(s12)}")
    print(f"  S1  (Stage 1 only / held out):          {len(s1)}")
    print(f"  U   (Unseen in Stage 1):                {len(u)}")
    print(f"  Total:                                  {len(s12) + len(s1) + len(u)}")

    return {"S12": s12, "S1": s1, "U": u}


def classify_molecule(mol_idx: int, categories: Dict[str, Set[int]]) -> str:
    """Return the category label for a single molecule index."""
    if mol_idx in categories["S12"]:
        return "S12"
    elif mol_idx in categories["S1"]:
        return "S1"
    elif mol_idx in categories["U"]:
        return "U"
    else:
        raise ValueError(f"Molecule index {mol_idx} not found in any category")


_CAT_RANK = {"S12": 0, "S1": 1, "U": 2}


def classify_mixture(mol_indices, categories: Dict[str, Set[int]]) -> str:
    """
    Return the category label for a mixture (binary or ternary).
    Components are sorted by visibility rank (S12 < S1 < U).
    """
    cats = [classify_molecule(int(m), categories) for m in mol_indices]
    cats.sort(key=lambda c: _CAT_RANK[c])
    return "_".join(cats)


# ---------------------------------------------------------------------------
# Step 2: Load mixture data by category
# ---------------------------------------------------------------------------

def load_binary_test_by_category(
    data_dir: Path,
    categories: Dict[str, Set[int]],
    liq_spectra: torch.Tensor,
    scale_factor: float,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Load all binary mixture test-split instances, grouped by category."""
    mix_path = data_dir / "mix_ml_instances.h5"
    with h5py.File(mix_path, "r") as f:
        mixture_spectra = torch.from_numpy(f["mixture_spectra"][:]).float()
        component_indices = f["component_indices"][:]
        n_mix = mixture_spectra.shape[0]

    _, _, test_idx = split_indices(n_mix, seed=seed)

    cat_order = BINARY_CATEGORY_ORDER
    cat_labels = BINARY_CATEGORY_LABELS

    # Group test instances by category
    by_category: Dict[str, List[int]] = defaultdict(list)
    for idx in test_idx:
        try:
            cat = classify_mixture(component_indices[idx], categories)
            by_category[cat].append(idx)
        except ValueError:
            continue

    print(f"\nBinary test split categorization:")
    for cat in cat_order:
        count = len(by_category.get(cat, []))
        if count > 0:
            print(f"  {cat_labels[cat]}: {count} instances")

    results = {}
    for cat, instance_indices in by_category.items():
        if cat not in cat_order:
            continue

        sel_idx = np.array(instance_indices)
        sel_mix = (mixture_spectra[sel_idx] * scale_factor).clamp(max=1.0)
        sel_comp_idx = component_indices[sel_idx]

        comp_a = (liq_spectra[sel_comp_idx[:, 0]] * scale_factor).clamp(max=1.0)
        comp_b = (liq_spectra[sel_comp_idx[:, 1]] * scale_factor).clamp(max=1.0)
        sel_components = torch.stack([comp_a, comp_b], dim=1)

        results[cat] = {
            "mixture_spectra": sel_mix,
            "component_spectra": sel_components,
            "component_indices": sel_comp_idx,
            "source": "binary",
        }

    return results


def load_drug_mixtures_by_category(
    data_dir: Path,
    categories: Dict[str, Set[int]],
    liq_spectra: torch.Tensor,
    scale_factor: float,
    split_seed: Optional[int] = None,
) -> Dict[str, Dict]:
    """Load drug mixtures from mix_drugs_ml_instances.h5 and group by category."""
    drug_mix_path = data_dir / "mix_drugs_ml_instances.h5"
    if not drug_mix_path.exists():
        print(f"Drug mixture file not found: {drug_mix_path}")
        return {}

    with h5py.File(drug_mix_path, "r") as f:
        mixture_spectra = torch.from_numpy(f["mixture_spectra"][:]).float()
        component_indices = f["component_indices"][:]

    # Optionally restrict to test split only
    if split_seed is not None:
        _, _, test_idx = split_indices(len(mixture_spectra), seed=split_seed)
        eligible = set(test_idx)
        print(f"\nDrug data split (seed={split_seed}): using {len(test_idx)} test instances out of {len(mixture_spectra)}")
    else:
        eligible = None

    cat_order = BINARY_CATEGORY_ORDER
    cat_labels = BINARY_CATEGORY_LABELS

    # Group instances by category
    by_category: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(mixture_spectra)):
        if eligible is not None and i not in eligible:
            continue
        try:
            cat = classify_mixture(component_indices[i], categories)
            by_category[cat].append(i)
        except ValueError:
            continue

    print(f"\nDrug mixture categorization:")
    for cat in cat_order:
        count = len(by_category.get(cat, []))
        if count > 0:
            print(f"  {cat_labels[cat]}: {count} instances")

    results = {}
    for cat, instance_indices in by_category.items():
        if cat not in cat_order:
            continue

        sel_idx = np.array(instance_indices)
        sel_mix = (mixture_spectra[sel_idx] * scale_factor).clamp(max=1.0)
        sel_comp_idx = component_indices[sel_idx]

        comp_a = (liq_spectra[sel_comp_idx[:, 0]] * scale_factor).clamp(max=1.0)
        comp_b = (liq_spectra[sel_comp_idx[:, 1]] * scale_factor).clamp(max=1.0)
        sel_components = torch.stack([comp_a, comp_b], dim=1)

        results[cat] = {
            "mixture_spectra": sel_mix,
            "component_spectra": sel_components,
            "component_indices": sel_comp_idx,
            "source": "drug",
        }

    return results


def load_ternary_test_by_category(
    data_dir: Path,
    categories: Dict[str, Set[int]],
    liq_spectra: torch.Tensor,
    scale_factor: float,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Load ternary mixture test-split instances, grouped by category."""
    ternary_path = data_dir / "mix_ternary_ml_instances.h5"
    if not ternary_path.exists():
        print(f"Ternary mixture file not found: {ternary_path}")
        return {}

    with h5py.File(ternary_path, "r") as f:
        mixture_spectra = torch.from_numpy(f["mixture_spectra"][:]).float()
        component_indices = f["component_indices"][:]
        n_mix = mixture_spectra.shape[0]

    _, _, test_idx = split_indices(n_mix, seed=seed)

    cat_order = TERNARY_CATEGORY_ORDER
    cat_labels = TERNARY_CATEGORY_LABELS

    by_category: Dict[str, List[int]] = defaultdict(list)
    for idx in test_idx:
        try:
            cat = classify_mixture(component_indices[idx], categories)
            by_category[cat].append(idx)
        except ValueError:
            continue

    print(f"\nTernary test split categorization:")
    for cat in cat_order:
        count = len(by_category.get(cat, []))
        if count > 0:
            print(f"  {cat_labels[cat]}: {count} instances")

    results = {}
    for cat, instance_indices in by_category.items():
        if cat not in cat_order:
            continue

        sel_idx = np.array(instance_indices)
        sel_mix = (mixture_spectra[sel_idx] * scale_factor).clamp(max=1.0)
        sel_comp_idx = component_indices[sel_idx]

        comps = []
        for k in range(3):
            comps.append((liq_spectra[sel_comp_idx[:, k]] * scale_factor).clamp(max=1.0))
        sel_components = torch.stack(comps, dim=1)

        results[cat] = {
            "mixture_spectra": sel_mix,
            "component_spectra": sel_components,
            "component_indices": sel_comp_idx,
            "source": "ternary",
        }

    return results


# ---------------------------------------------------------------------------
# Step 3: Evaluation
# ---------------------------------------------------------------------------

def evaluate_category(
    model: torch.nn.Module,
    data: Dict,
    device: torch.device,
    categories: Dict[str, Set[int]],
    batch_size: int = 32,
    database: Optional[LiquidIRDatabase] = None,
    smiles_list: Optional[List[str]] = None,
    peak_matcher: Optional[RefinementMatcher] = None,
) -> Dict:
    """Evaluate the model on a single category of mixtures."""
    mixture = data["mixture_spectra"]
    components = data["component_spectra"]
    comp_indices = data["component_indices"]
    n_samples = len(mixture)
    n_components = components.shape[1]

    all_metrics: List[Dict[str, float]] = []
    total_top1 = 0
    total_top5 = 0
    total_top10 = 0
    total_top50 = 0
    total_components = 0

    # Per-molecule-category metrics
    per_mol_cat_cosines: Dict[str, List[float]] = defaultdict(list)
    per_mol_cat_mses: Dict[str, List[float]] = defaultdict(list)
    per_mol_cat_top1: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_top5: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_top10: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_top50: Dict[str, List[int]] = defaultdict(list)

    # Per-individual-molecule tracking (keyed by SMILES)
    # Each entry: list of dicts with {co_cat, top1, top5, top10, top50, cosine}
    per_molecule_detail: Dict[str, List[Dict]] = defaultdict(list)

    # Per-slot metrics (slot bias analysis)
    per_slot_cosines: Dict[int, List[float]] = defaultdict(list)
    per_slot_mses: Dict[int, List[float]] = defaultdict(list)
    per_slot_top1: Dict[int, List[int]] = defaultdict(list)
    per_slot_top5: Dict[int, List[int]] = defaultdict(list)
    per_slot_top10: Dict[int, List[int]] = defaultdict(list)
    per_slot_top50: Dict[int, List[int]] = defaultdict(list)

    # Mixture-level match tracking (all components correct at top-1)
    mixture_top1_matches: List[int] = []

    # Refinement matching accumulators (binary only)
    use_refinement = peak_matcher is not None and n_components == 2
    refine_comp_hits = [0, 0, 0, 0]   # total component hits at [top1, top5, top10, top50]
    refine_pair_hits = [0, 0, 0, 0]   # total pair hits at [top1, top5, top10, top50]
    refine_total_components = 0
    refine_total_samples = 0
    refine_recon_cosines: List[float] = []
    # Per-molecule-category refinement hits
    per_mol_cat_refine_top1: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_refine_top5: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_refine_top10: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_refine_top50: Dict[str, List[int]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_mix = mixture[start:end].to(device)
            batch_comp = components[start:end].to(device)

            outputs = model(mixture=batch_mix, stage=2)
            pred_components = outputs["components"]  # [B, K, L]

            batch_metrics = _per_sample_metrics(pred_components, batch_comp)
            all_metrics.extend(batch_metrics)

            # Per-component matching using Hungarian assignment
            # We query the database per-component so we can track accuracy by molecule category
            if database is not None and smiles_list is not None:
                for bi, m in enumerate(batch_metrics):
                    sample_idx = start + bi
                    pred_order = m["matched_pred_order"]
                    target_order = m["matched_target_order"]

                    # Pre-collect all component categories for co-component tracking
                    comp_cats = []
                    comp_smiles = []
                    for tgt_ki in target_order:
                        mol_idx = int(comp_indices[sample_idx, tgt_ki])
                        comp_cats.append(classify_molecule(mol_idx, categories))
                        comp_smiles.append(smiles_list[mol_idx])

                    sample_top1_hits = []
                    for ki, tgt_ki in enumerate(target_order):
                        mol_idx = int(comp_indices[sample_idx, tgt_ki])
                        mol_cat = comp_cats[ki]
                        gt_smiles = comp_smiles[ki]

                        # Co-component categories (all other components in this mixture)
                        co_cats = [comp_cats[j] for j in range(len(target_order)) if j != ki]
                        co_smiles = [comp_smiles[j] for j in range(len(target_order)) if j != ki]

                        # Unmixing metrics
                        per_mol_cat_cosines[mol_cat].append(m["per_component_cosines"][ki])
                        per_mol_cat_mses[mol_cat].append(m["per_component_mses"][ki])

                        # Database matching for this predicted slot
                        pred_slot_idx = pred_order[ki]
                        query = pred_components[bi, pred_slot_idx]
                        matches = database.find_best_match(query, top_k=50)
                        top1_hit = 1 if matches and matches[0][0] == gt_smiles else 0
                        top5_hit = 1 if gt_smiles in {s for s, _, _ in matches[:5]} else 0
                        top10_hit = 1 if gt_smiles in {s for s, _, _ in matches[:10]} else 0
                        top50_hit = 1 if gt_smiles in {s for s, _, _ in matches} else 0

                        sample_top1_hits.append(top1_hit)

                        per_mol_cat_top1[mol_cat].append(top1_hit)
                        per_mol_cat_top5[mol_cat].append(top5_hit)
                        per_mol_cat_top10[mol_cat].append(top10_hit)
                        per_mol_cat_top50[mol_cat].append(top50_hit)

                        total_top1 += top1_hit
                        total_top5 += top5_hit
                        total_top10 += top10_hit
                        total_top50 += top50_hit
                        total_components += 1

                        # Per-slot tracking
                        per_slot_cosines[pred_slot_idx].append(m["per_component_cosines"][ki])
                        per_slot_mses[pred_slot_idx].append(m["per_component_mses"][ki])
                        per_slot_top1[pred_slot_idx].append(top1_hit)
                        per_slot_top5[pred_slot_idx].append(top5_hit)
                        per_slot_top10[pred_slot_idx].append(top10_hit)
                        per_slot_top50[pred_slot_idx].append(top50_hit)

                        # Per-individual-molecule detail tracking
                        per_molecule_detail[gt_smiles].append({
                            "mol_cat": mol_cat,
                            "co_cats": co_cats,
                            "co_smiles": co_smiles,
                            "cosine": m["per_component_cosines"][ki],
                            "top1": top1_hit,
                            "top5": top5_hit,
                            "top10": top10_hit,
                            "top50": top50_hit,
                        })

                    # Mixture-level: all components correct at top-1
                    mixture_top1_matches.append(int(all(h == 1 for h in sample_top1_hits)))

                    # Refinement matching for this sample (binary only)
                    if use_refinement:
                        gt_smi_list = [
                            smiles_list[int(comp_indices[sample_idx, t])]
                            for t in range(n_components)
                        ]
                        gt_set = set(gt_smi_list)
                        try:
                            ref_result = peak_matcher.match_components_joint(
                                pred_components[bi],
                                batch_mix[bi],
                                ground_truth_smiles=gt_smi_list,
                                top_k=50,  # Keep 50 for comprehensive analysis (includes top-50 metrics)
                            )
                        except Exception:
                            continue

                        joint_top_k = ref_result["joint_top_k"]
                        refine_recon_cosines.append(ref_result["reconstruction_cosine"])
                        refine_total_components += n_components
                        refine_total_samples += 1

                        cutoffs = [1, 5, 10, 50]
                        refine_dicts = [
                            per_mol_cat_refine_top1, per_mol_cat_refine_top5,
                            per_mol_cat_refine_top10, per_mol_cat_refine_top50,
                        ]
                        # Pre-compute paired molecule sets at each cutoff
                        paired_at = []
                        for cutoff in cutoffs:
                            s = set()
                            for p, _ in joint_top_k[:cutoff]:
                                s.update(p)
                            paired_at.append(s)

                        for ci, cutoff in enumerate(cutoffs):
                            # Pair correct
                            pair_ok = any(set(p) == gt_set for p, _ in joint_top_k[:cutoff])
                            refine_pair_hits[ci] += int(pair_ok)
                            # Per-component hits
                            for tgt_ki in range(n_components):
                                mol_idx = int(comp_indices[sample_idx, tgt_ki])
                                gt_smi = smiles_list[mol_idx]
                                hit = 1 if gt_smi in paired_at[ci] else 0
                                refine_comp_hits[ci] += hit
                                # Per-molecule-category tracking
                                mol_cat = classify_molecule(mol_idx, categories)
                                refine_dicts[ci][mol_cat].append(hit)
            else:
                # Only unmixing metrics (no matching)
                for bi, m in enumerate(batch_metrics):
                    sample_idx = start + bi
                    pred_order = m["matched_pred_order"]
                    target_order = m["matched_target_order"]
                    for ki, tgt_ki in enumerate(target_order):
                        mol_idx = int(comp_indices[sample_idx, tgt_ki])
                        mol_cat = classify_molecule(mol_idx, categories)
                        per_mol_cat_cosines[mol_cat].append(m["per_component_cosines"][ki])
                        per_mol_cat_mses[mol_cat].append(m["per_component_mses"][ki])
                        # Per-slot tracking
                        pred_slot_idx = pred_order[ki]
                        per_slot_cosines[pred_slot_idx].append(m["per_component_cosines"][ki])
                        per_slot_mses[pred_slot_idx].append(m["per_component_mses"][ki])

    # Aggregate
    metric_keys = ["component_cosine", "component_mse"]
    agg = {}
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Per-molecule-category aggregation
    per_mol_cat = {}
    for cat_label in ["S12", "S1", "U"]:
        if cat_label in per_mol_cat_cosines:
            cos_vals = per_mol_cat_cosines[cat_label]
            mse_vals = per_mol_cat_mses[cat_label]
            entry = {
                "n": len(cos_vals),
                "cosine": {"mean": float(np.mean(cos_vals)), "std": float(np.std(cos_vals))},
                "mse": {"mean": float(np.mean(mse_vals)), "std": float(np.std(mse_vals))},
            }
            if cat_label in per_mol_cat_top1:
                entry["top1_accuracy"] = float(np.mean(per_mol_cat_top1[cat_label]))
                entry["top5_accuracy"] = float(np.mean(per_mol_cat_top5[cat_label]))
                entry["top10_accuracy"] = float(np.mean(per_mol_cat_top10[cat_label]))
                entry["top50_accuracy"] = float(np.mean(per_mol_cat_top50[cat_label]))
            if cat_label in per_mol_cat_refine_top1:
                entry["refine_top1_accuracy"] = float(np.mean(per_mol_cat_refine_top1[cat_label]))
                entry["refine_top5_accuracy"] = float(np.mean(per_mol_cat_refine_top5[cat_label]))
                entry["refine_top10_accuracy"] = float(np.mean(per_mol_cat_refine_top10[cat_label]))
                entry["refine_top50_accuracy"] = float(np.mean(per_mol_cat_refine_top50[cat_label]))
            per_mol_cat[cat_label] = entry

    # Per-slot aggregation (slot bias)
    per_slot = {}
    for slot_idx in sorted(per_slot_cosines.keys()):
        cos_vals = per_slot_cosines[slot_idx]
        mse_vals = per_slot_mses[slot_idx]
        entry = {
            "n": len(cos_vals),
            "cosine": {"mean": float(np.mean(cos_vals)), "std": float(np.std(cos_vals))},
            "mse": {"mean": float(np.mean(mse_vals)), "std": float(np.std(mse_vals))},
        }
        if slot_idx in per_slot_top1:
            entry["top1_accuracy"] = float(np.mean(per_slot_top1[slot_idx]))
            entry["top5_accuracy"] = float(np.mean(per_slot_top5[slot_idx]))
            entry["top10_accuracy"] = float(np.mean(per_slot_top10[slot_idx]))
            entry["top50_accuracy"] = float(np.mean(per_slot_top50[slot_idx]))
        per_slot[str(slot_idx)] = entry

    result = {
        "n_samples": n_samples,
        "unmixing": agg,
        "per_molecule_category": per_mol_cat,
        "per_slot": per_slot,
        "per_sample": all_metrics,
        "per_molecule_detail": dict(per_molecule_detail),
    }

    if total_components > 0:
        result["matching"] = {
            "top1_accuracy": float(total_top1) / total_components,
            "top5_accuracy": float(total_top5) / total_components,
            "top10_accuracy": float(total_top10) / total_components,
            "top50_accuracy": float(total_top50) / total_components,
            "mixture_match_rate": float(np.mean(mixture_top1_matches)) if mixture_top1_matches else 0.0,
        }

    if refine_total_components > 0:
        recon_arr = np.array(refine_recon_cosines)
        result["refinement_matching"] = {
            "component_top1_accuracy": float(refine_comp_hits[0]) / refine_total_components,
            "component_top5_accuracy": float(refine_comp_hits[1]) / refine_total_components,
            "component_top10_accuracy": float(refine_comp_hits[2]) / refine_total_components,
            "component_top50_accuracy": float(refine_comp_hits[3]) / refine_total_components,
            "pair_top1_accuracy": float(refine_pair_hits[0]) / refine_total_samples,
            "pair_top5_accuracy": float(refine_pair_hits[1]) / refine_total_samples,
            "pair_top10_accuracy": float(refine_pair_hits[2]) / refine_total_samples,
            "pair_top50_accuracy": float(refine_pair_hits[3]) / refine_total_samples,
            "reconstruction_cosine_mean": float(recon_arr.mean()),
            "reconstruction_cosine_std": float(recon_arr.std()),
        }

    return result


# ---------------------------------------------------------------------------
# Step 4: Visualization and reporting
# ---------------------------------------------------------------------------

def _build_per_component_grid(
    results: Dict[str, Dict],
) -> Dict[str, Dict[str, Dict]]:
    """Build a 2D grid[row_cat][col_cat] of per-component metrics."""
    tag_order = ["S12", "S1", "U"]
    cell_n: Dict[tuple, int] = {}
    cell_sums: Dict[tuple, Dict[str, float]] = {}

    for cat_key, r in results.items():
        per_mol = r.get("per_molecule_category", {})
        parts = cat_key.split("_")

        for comp_cat, mc in per_mol.items():
            remaining = list(parts)
            remaining.remove(comp_cat)
            other = remaining[0]

            key = (comp_cat, other)
            n = mc["n"]

            if key not in cell_n:
                cell_n[key] = 0
                cell_sums[key] = {}
            cell_n[key] += n

            for metric, accessor in [
                ("cosine_mean", lambda m: m["cosine"]["mean"]),
                ("top1", lambda m: m.get("top1_accuracy")),
                ("top5", lambda m: m.get("top5_accuracy")),
                ("top10", lambda m: m.get("top10_accuracy")),
                ("top50", lambda m: m.get("top50_accuracy")),
                ("refine_top1", lambda m: m.get("refine_top1_accuracy")),
                ("refine_top5", lambda m: m.get("refine_top5_accuracy")),
                ("refine_top10", lambda m: m.get("refine_top10_accuracy")),
                ("refine_top50", lambda m: m.get("refine_top50_accuracy")),
            ]:
                val = accessor(mc)
                if val is not None:
                    cell_sums[key][metric] = cell_sums[key].get(metric, 0) + n * val

    grid: Dict[str, Dict[str, Dict]] = {
        t: {t2: None for t2 in tag_order} for t in tag_order
    }
    for (row, col), count in cell_n.items():
        sums = cell_sums[(row, col)]
        entry = {"n": count}
        if "cosine_mean" in sums:
            entry["cosine"] = {"mean": sums["cosine_mean"] / count}
        for k in ["top1", "top5", "top10", "top50",
                   "refine_top1", "refine_top5", "refine_top10", "refine_top50"]:
            if k in sums:
                entry[f"{k}_accuracy"] = sums[k] / count
        grid[row][col] = entry

    return grid


def _build_per_component_grid_3d(
    results: Dict[str, Dict],
) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """Build a 3D grid[row][co1][co2] of per-component metrics for ternary."""
    tag_order = ["S12", "S1", "U"]
    cell_n: Dict[tuple, int] = {}
    cell_sums: Dict[tuple, Dict[str, float]] = {}

    for cat_key, r in results.items():
        per_mol = r.get("per_molecule_category", {})
        parts = cat_key.split("_")

        for comp_cat, mc in per_mol.items():
            remaining = list(parts)
            remaining.remove(comp_cat)
            remaining.sort(key=lambda c: _CAT_RANK[c])
            co1, co2 = remaining[0], remaining[1]

            key = (comp_cat, co1, co2)
            n = mc["n"]

            if key not in cell_n:
                cell_n[key] = 0
                cell_sums[key] = {}
            cell_n[key] += n

            for metric, accessor in [
                ("top1", lambda m: m.get("top1_accuracy")),
                ("top5", lambda m: m.get("top5_accuracy")),
                ("top10", lambda m: m.get("top10_accuracy")),
                ("top50", lambda m: m.get("top50_accuracy")),
            ]:
                val = accessor(mc)
                if val is not None:
                    cell_sums[key][metric] = cell_sums[key].get(metric, 0) + n * val

    grid = {
        t1: {t2: {t3: None for t3 in tag_order} for t2 in tag_order}
        for t1 in tag_order
    }
    for (row, co1, co2), count in cell_n.items():
        sums = cell_sums[(row, co1, co2)]
        entry = {"n": count}
        if "top1" in sums:
            entry["top1_accuracy"] = sums["top1"] / count
        if "top5" in sums:
            entry["top5_accuracy"] = sums["top5"] / count
        if "top10" in sums:
            entry["top10_accuracy"] = sums["top10"] / count
        if "top50" in sums:
            entry["top50_accuracy"] = sums["top50"] / count
        grid[row][co1][co2] = entry

    return grid


def generate_plots(
    results: Dict[str, Dict],
    output_dir: Path,
    category_order: List[str] = None,
    category_labels: Dict[str, str] = None,
    suffix: str = "",
) -> List[str]:
    """Generate three separate plots."""
    if category_order is None:
        category_order = BINARY_CATEGORY_ORDER
    if category_labels is None:
        category_labels = BINARY_CATEGORY_LABELS
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return []

    cats_present = [c for c in category_order if c in results]
    if not cats_present:
        return []

    cos_means, cos_stds, labels = [], [], []
    top1s, top5s, top10s, top50s = [], [], [], []
    has_matching = False
    for cat in cats_present:
        r = results[cat]
        cos = r["unmixing"]["component_cosine"]
        cos_means.append(cos["mean"])
        cos_stds.append(cos["std"])
        labels.append(category_labels[cat])
        if "matching" in r:
            has_matching = True
            top1s.append(r["matching"]["top1_accuracy"] * 100)
            top5s.append(r["matching"]["top5_accuracy"] * 100)
            top10s.append(r["matching"]["top10_accuracy"] * 100)
            top50s.append(r["matching"].get("top50_accuracy", 0) * 100)

    palette = ["#2c7bb6", "#abd9e9", "#fdae61", "#f46d43", "#d73027"]
    if len(cats_present) > len(palette):
        palette = palette + ["#888888"] * (len(cats_present) - len(palette))
    colors = palette[: len(cats_present)]

    saved: List[str] = []

    # --- Plot 1: Component cosine bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, cos_means, yerr=cos_stds, capsize=4, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Component Cosine Similarity", fontsize=13)
    ax.set_title("Unmixing Quality by Pair Category", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.08)
    for bar, m, n_samp in zip(bars, cos_means, [results[c]["n_samples"] for c in cats_present]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"{m:.3f}\nn={n_samp}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    name = f"generalization_bar{suffix}.png"
    fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)
    print(f"  Saved {name}")

    # --- Helper: build mixture-level 3x3 grid (symmetric) ---
    def _build_mixture_grid(metric_key: str) -> np.ndarray:
        """Build symmetric 3x3 grid of per-mixture metric by category."""
        tag_order_local = ["S12", "S1", "U"]
        g = np.full((3, 3), np.nan)
        for cat in cats_present:
            r = results[cat]
            m = r.get("matching", {})
            rm = r.get("refinement_matching", {})
            # Check matching dict first, then refinement (explicit None check to handle 0.0)
            val = m.get(metric_key)
            if val is None:
                val = rm.get(metric_key)
            if val is None:
                continue
            parts = cat.split("_")
            idx_a = tag_order_local.index(parts[0])
            idx_b = tag_order_local.index(parts[1])
            g[idx_a, idx_b] = val * 100
            g[idx_b, idx_a] = val * 100  # symmetric
        return g

    # --- Helper: render a single 3x3 heatmap on an axis ---
    def _render_heatmap(ax, g, title, xlabel, ylabel, cmap="YlGnBu"):
        tag_order_local = ["S12", "S1", "U"]
        im = ax.imshow(g, vmin=0, vmax=100, cmap=cmap, aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(tag_order_local, fontsize=13)
        ax.set_yticklabels(tag_order_local, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=15, fontweight="bold")
        for i in range(3):
            for j in range(3):
                val = g[i, j]
                if not np.isnan(val):
                    color = "white" if val > 50 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=14, fontweight="bold", color=color)
        return im

    # --- Plot 2: Per-component heatmaps ---
    tag_order = ["S12", "S1", "U"]
    acc_keys = [
        ("top1_accuracy", "Top-1"),
        ("top5_accuracy", "Top-5"),
        ("top10_accuracy", "Top-10"),
        ("top50_accuracy", "Top-50"),
    ]

    # Detect ternary by checking if any category has 3 parts
    is_ternary = any(len(c.split("_")) == 3 for c in cats_present)

    if is_ternary:
        # 3D grid: grid[row][co1][co2], visualized as 3x3 panel grid
        grid3d = _build_per_component_grid_3d(results)
        slice_labels = ["Best co-comp: S12", "Best co-comp: S1", "Best co-comp: U"]

        n_acc = len(acc_keys)
        fig, axes = plt.subplots(3, n_acc, figsize=(5*n_acc+1, 13), constrained_layout=True)
        im = None
        for si, co1_tag in enumerate(tag_order):
            for ki, (acc_key, topk_title) in enumerate(acc_keys):
                ax = axes[si, ki]
                g = np.full((3, 3), np.nan)
                for i, row_tag in enumerate(tag_order):
                    for j, co2_tag in enumerate(tag_order):
                        cell = grid3d[row_tag][co1_tag][co2_tag]
                        if cell is not None and acc_key in cell:
                            g[i, j] = cell[acc_key] * 100
                im = ax.imshow(g, vmin=0, vmax=100, cmap="YlGnBu", aspect="equal")
                ax.set_xticks(range(3))
                ax.set_yticks(range(3))
                ax.set_xticklabels(tag_order, fontsize=12)
                ax.set_yticklabels(tag_order, fontsize=12)
                if si == 2:
                    ax.set_xlabel("Worst co-comp", fontsize=12)
                if ki == 0:
                    ax.set_ylabel(f"{slice_labels[si]}\nComponent evaluated", fontsize=12)
                if si == 0:
                    ax.set_title(f"{topk_title} Accuracy", fontsize=14, fontweight="bold")
                for i in range(3):
                    for j in range(3):
                        val = g[i, j]
                        if not np.isnan(val):
                            color = "white" if val > 50 else "black"
                            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                                    fontsize=12, fontweight="bold", color=color)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, label="Accuracy (%)")
    else:
        # 2D grid: per-component panels + mixture R@1 panel
        grid = _build_per_component_grid(results)

        # Build per-component grids
        acc_grids = []
        for acc_key, _ in acc_keys:
            g = np.full((3, 3), np.nan)
            for i, row_tag in enumerate(tag_order):
                for j, col_tag in enumerate(tag_order):
                    cell = grid[row_tag][col_tag]
                    if cell is not None and acc_key in cell:
                        g[i, j] = cell[acc_key] * 100
            acc_grids.append(g)

        # Build mixture R@1 grid (symmetric)
        mix_grid = _build_mixture_grid("mixture_match_rate")
        has_mix_grid = not np.all(np.isnan(mix_grid))

        n_panels = len(acc_keys) + (1 if has_mix_grid else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels+1, 5.5), constrained_layout=True)
        im = None
        for idx, (g, (_, title)) in enumerate(zip(acc_grids, acc_keys)):
            im = _render_heatmap(axes[idx], g, f"{title} Accuracy",
                                 "Paired with", "Component evaluated")
        if has_mix_grid:
            im = _render_heatmap(axes[-1], mix_grid, "Mixture R@1",
                                 "Component B", "Component A")
        if im is not None:
            fig.colorbar(im, ax=list(axes), shrink=0.75, pad=0.02, label="Accuracy (%)")

    name = f"generalization_heatmap{suffix}.png"
    fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)
    print(f"  Saved {name}")

    # --- Plot 3: Top-k matching accuracy (cosine) ---
    if has_matching:
        fig, ax = plt.subplots(figsize=(10, 5))
        w = 0.18
        x2 = np.arange(len(labels))
        ax.bar(x2 - 1.5*w, top1s, w, label="Top-1", color="#2c7bb6", alpha=0.85)
        ax.bar(x2 - 0.5*w, top5s, w, label="Top-5", color="#abd9e9", alpha=0.85)
        ax.bar(x2 + 0.5*w, top10s, w, label="Top-10", color="#fdae61", alpha=0.85)
        ax.bar(x2 + 1.5*w, top50s, w, label="Top-50", color="#f46d43", alpha=0.85)
        ax.set_xticks(x2)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title("Cosine Matching: Molecule Identification by Category", fontsize=15, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=11, loc="upper right")
        for xi, (t1, t5, t10, t50) in enumerate(zip(top1s, top5s, top10s, top50s)):
            ax.text(xi - 1.5*w, t1 + 1.5, f"{t1:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi - 0.5*w, t5 + 1.5, f"{t5:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + 0.5*w, t10 + 1.5, f"{t10:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + 1.5*w, t50 + 1.5, f"{t50:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        name = f"generalization_matching{suffix}.png"
        fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)
        print(f"  Saved {name}")

    # --- Plot 4: Refinement matching per-component heatmap + pair R@1 (binary only) ---
    has_refinement = any("refinement_matching" in results[c] for c in cats_present)
    if has_refinement and not is_ternary:
        grid = _build_per_component_grid(results)
        refine_acc_keys = [
            ("refine_top1_accuracy", "cR@1 (per-component)"),
            ("refine_top5_accuracy", "cR@5 (per-component)"),
            ("refine_top10_accuracy", "cR@10 (per-component)"),
            ("refine_top50_accuracy", "cR@50 (per-component)"),
        ]

        # Build per-component grids
        refine_grids = []
        for acc_key, _ in refine_acc_keys:
            g = np.full((3, 3), np.nan)
            for i, row_tag in enumerate(tag_order):
                for j, col_tag in enumerate(tag_order):
                    cell = grid[row_tag][col_tag]
                    if cell is not None and acc_key in cell:
                        g[i, j] = cell[acc_key] * 100
            refine_grids.append(g)

        # Add pair R@1 grid (per-mixture)
        pair_r1_grid = _build_mixture_grid("pair_top1_accuracy")
        refine_grids.append(pair_r1_grid)
        refine_acc_keys.append((None, "Pair R@1 (per-mixture)"))

        n_racc = len(refine_acc_keys)
        fig, axes = plt.subplots(1, n_racc, figsize=(5*n_racc+1, 5), constrained_layout=True)
        im = None
        for idx, (ax, g, (_, title)) in enumerate(zip(axes, refine_grids, refine_acc_keys)):
            # Use symmetric labels for pair R@1, asymmetric for per-component
            if idx < len(refine_acc_keys) - 1:
                xlabel, ylabel = "Paired with", "Component evaluated"
            else:
                xlabel, ylabel = "Component 2", "Component 1"
            im = _render_heatmap(ax, g, title, xlabel, ylabel, cmap="YlGnBu")

        if im is not None:
            fig.colorbar(im, ax=axes.tolist(), shrink=0.75, pad=0.02, label="Accuracy (%)")

        name = f"generalization_refinement_heatmap{suffix}.png"
        fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)
        print(f"  Saved {name}")

    # --- Plot 5: Refinement matching bar chart (cosine vs refinement comparison) ---
    if has_refinement:
        cr1s, cr5s, pr1s, pr5s = [], [], [], []
        cos_t1s = []
        ref_labels = []
        for cat in cats_present:
            r = results[cat]
            rm = r.get("refinement_matching", {})
            m = r.get("matching", {})
            if rm:
                ref_labels.append(category_labels[cat])
                cos_t1s.append(m.get("top1_accuracy", 0) * 100)
                cr1s.append(rm.get("component_top1_accuracy", 0) * 100)
                cr5s.append(rm.get("component_top5_accuracy", 0) * 100)
                pr1s.append(rm.get("pair_top1_accuracy", 0) * 100)
                pr5s.append(rm.get("pair_top5_accuracy", 0) * 100)

        if ref_labels:
            fig, ax = plt.subplots(figsize=(10, 5))
            w = 0.15
            x2 = np.arange(len(ref_labels))
            ax.bar(x2 - 2*w, cos_t1s, w, label="Cosine R@1", color="#2c7bb6", alpha=0.85)
            ax.bar(x2 - 1*w, cr1s, w, label="Refine cR@1", color="#d73027", alpha=0.85)
            ax.bar(x2 + 0*w, cr5s, w, label="Refine cR@5", color="#fc8d59", alpha=0.85)
            ax.bar(x2 + 1*w, pr1s, w, label="Pair R@1", color="#91bfdb", alpha=0.85)
            ax.bar(x2 + 2*w, pr5s, w, label="Pair R@5", color="#e0f3f8", alpha=0.85, edgecolor="#999")
            ax.set_xticks(x2)
            ax.set_xticklabels(ref_labels, fontsize=13)
            ax.set_ylabel("Accuracy (%)", fontsize=13)
            ax.set_title("Cosine vs Refinement Matching by Category", fontsize=15, fontweight="bold")
            ax.set_ylim(0, 110)
            ax.legend(fontsize=11, loc="upper right", ncol=2)
            for xi, vals in enumerate(zip(cos_t1s, cr1s, cr5s, pr1s, pr5s)):
                for vi, (v, off) in enumerate(zip(vals, [-2*w, -1*w, 0, 1*w, 2*w])):
                    ax.text(xi + off, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            name = f"generalization_refinement_matching{suffix}.png"
            fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved.append(name)
            print(f"  Saved {name}")

    return saved


def write_analysis_md(
    results: Dict[str, Dict],
    categories: Dict[str, Set[int]],
    n_holdout: int,
    output_dir: Path,
    plot_names: Optional[List[str]] = None,
    has_matching: bool = True,
    category_order: List[str] = None,
    category_labels: Dict[str, str] = None,
) -> None:
    """Write comprehensive generalization analysis to markdown."""
    if category_order is None:
        category_order = BINARY_CATEGORY_ORDER
    if category_labels is None:
        category_labels = BINARY_CATEGORY_LABELS
    md_path = output_dir / "generalization_analysis.md"
    lines: List[str] = []

    def add(s: str = ""):
        lines.append(s)

    # Build ordered list of result keys (binary first, then drug:)
    keys_present = []
    for c in category_order:
        if c in results:
            keys_present.append(c)
    for c in category_order:
        dkey = f"drug:{c}"
        if dkey in results:
            keys_present.append(dkey)

    def _label(key: str) -> str:
        cat = key.removeprefix("drug:")
        base = category_labels[cat]
        if key.startswith("drug:"):
            return f"{base} [drug]"
        return base

    # --- Header ---
    add("# Generalization Analysis")
    add()
    add("## 1. Molecule Visibility Categories")
    add()
    add("| Category | Definition |")
    add("|----------|------------|")
    add("| **S12** | In Stage 1 DB AND in Stage 2 training |")
    add("| **S1** | In Stage 1 DB only (not in Stage 2 training) |")
    add("| **U** | Not in Stage 1 DB (drug-only molecules) |")
    add()
    add(f"n_holdout_molecules = {n_holdout}")
    add()
    add("| Category | Count |")
    add("|----------|------:|")
    add(f"| S12 | {len(categories['S12']):,} |")
    add(f"| S1 | {len(categories['S1']):,} |")
    add(f"| U | {len(categories['U']):,} |")
    add(f"| **Total** | **{sum(len(v) for v in categories.values()):,}** |")
    add()

    # --- Results Summary (combined unmixing + matching) ---
    has_refine = any("refinement_matching" in results[k] for k in keys_present)

    add("## 2. Results Summary")
    add()
    add("### Cosine Matching")
    add()
    header = "| Category | Source | N | Comp Cos | Comp MSE |"
    sep = "|----------|:------:|--:|:--------:|:--------:|"
    if has_matching and any("matching" in results[k] for k in keys_present):
        header += " Top-1 | Top-5 | Top-10 | Top-50 |"
        sep += "------:|------:|-------:|-------:|"
    add(header)
    add(sep)
    for key in keys_present:
        r = results[key]
        u = r["unmixing"]
        source = r.get("source", "")
        row = (
            f"| {_label(key)} | {source} | {r['n_samples']:,} "
            f"| {u['component_cosine']['mean']:.4f} "
            f"| {u['component_mse']['mean']:.6f} |"
        )
        if "matching" in r:
            m = r["matching"]
            row = row.rstrip("|") + (
                f" {m['top1_accuracy']:.4f} "
                f"| {m['top5_accuracy']:.4f} "
                f"| {m['top10_accuracy']:.4f} "
                f"| {m.get('top50_accuracy', 0):.4f} |"
            )
        add(row)
    add()

    if has_refine:
        add("### Refinement Matching (peak_BCR)")
        add()
        add("| Category | Source | N | cR@1 | cR@5 | cR@10 | cR@50 | pR@1 | pR@5 | Recon |")
        add("|----------|:------:|--:|-----:|-----:|------:|------:|-----:|-----:|------:|")
        for key in keys_present:
            r = results[key]
            source = r.get("source", "")
            rm = r.get("refinement_matching", {})
            if not rm:
                continue
            add(
                f"| {_label(key)} | {source} | {r['n_samples']:,} "
                f"| {rm.get('component_top1_accuracy', 0):.4f} "
                f"| {rm.get('component_top5_accuracy', 0):.4f} "
                f"| {rm.get('component_top10_accuracy', 0):.4f} "
                f"| {rm.get('component_top50_accuracy', 0):.4f} "
                f"| {rm.get('pair_top1_accuracy', 0):.4f} "
                f"| {rm.get('pair_top5_accuracy', 0):.4f} "
                f"| {rm.get('reconstruction_cosine_mean', 0):.4f} |"
            )
        add()

    # --- Per-component breakdown (by molecule visibility) ---
    has_per_mol = any(r.get("per_molecule_category") for r in results.values())
    if has_per_mol:
        add("## 3. Per-Component Breakdown (by molecule visibility)")
        add()
        add("### Cosine Matching")
        add()
        header = "| Category | Component | N | Cosine | MSE |"
        sep = "|----------|-----------|--:|:------:|:---:|"
        if has_matching:
            header += " Top-1 | Top-5 | Top-10 | Top-50 |"
            sep += "------:|------:|-------:|-------:|"
        add(header)
        add(sep)
        for key in keys_present:
            r = results[key]
            source = r.get("source", "")
            per_mol = r.get("per_molecule_category", {})
            if not per_mol:
                continue
            first_row = True
            for mol_cat in ["S12", "S1", "U"]:
                if mol_cat not in per_mol:
                    continue
                mc = per_mol[mol_cat]
                row_label = f"{_label(key)} [{source}]" if first_row else ""
                row = (
                    f"| {row_label} | {mol_cat} | {mc['n']:,} "
                    f"| {mc['cosine']['mean']:.4f} "
                    f"| {mc['mse']['mean']:.6f} |"
                )
                if "top1_accuracy" in mc:
                    row = row.rstrip("|") + (
                        f" {mc['top1_accuracy']:.4f} "
                        f"| {mc['top5_accuracy']:.4f} "
                        f"| {mc['top10_accuracy']:.4f} "
                        f"| {mc.get('top50_accuracy', 0):.4f} |"
                    )
                add(row)
                first_row = False
        add()

        # Refinement per-component breakdown
        has_refine_per_mol = any(
            "refine_top1_accuracy" in mc
            for r in results.values()
            for mc in r.get("per_molecule_category", {}).values()
        )
        if has_refine_per_mol:
            add("### Refinement Matching (peak_BCR, per-component)")
            add()
            add("| Category | Component | N | Cos Top-1 | cR@1 | cR@5 | cR@10 | cR@50 |")
            add("|----------|-----------|--:|----------:|-----:|-----:|------:|------:|")
            for key in keys_present:
                r = results[key]
                source = r.get("source", "")
                per_mol = r.get("per_molecule_category", {})
                if not per_mol:
                    continue
                first_row = True
                for mol_cat in ["S12", "S1", "U"]:
                    if mol_cat not in per_mol:
                        continue
                    mc = per_mol[mol_cat]
                    if "refine_top1_accuracy" not in mc:
                        continue
                    row_label = f"{_label(key)} [{source}]" if first_row else ""
                    add(
                        f"| {row_label} | {mol_cat} | {mc['n']:,} "
                        f"| {mc.get('top1_accuracy', 0):.4f} "
                        f"| {mc['refine_top1_accuracy']:.4f} "
                        f"| {mc['refine_top5_accuracy']:.4f} "
                        f"| {mc['refine_top10_accuracy']:.4f} "
                        f"| {mc['refine_top50_accuracy']:.4f} |"
                    )
                    first_row = False
            add()

    # --- Per-slot breakdown (slot bias) ---
    has_slot_data = any("per_slot" in results[k] and results[k]["per_slot"] for k in keys_present)
    if has_slot_data:
        slot_section = 4 if has_per_mol else 3
        add(f"## {slot_section}. Per-Slot Breakdown (slot bias)")
        add()
        header = "| Category | Slot | N | Cosine | MSE |"
        sep = "|----------|-----:|--:|:------:|:---:|"
        if has_matching:
            header += " Top-1 | Top-5 | Top-10 | Top-50 |"
            sep += "------:|------:|-------:|-------:|"
        add(header)
        add(sep)
        for key in keys_present:
            r = results[key]
            source = r.get("source", "")
            ps = r.get("per_slot", {})
            if not ps:
                continue
            first_row = True
            for slot_id in sorted(ps.keys(), key=int):
                s = ps[slot_id]
                row_label = f"{_label(key)} [{source}]" if first_row else ""
                row = (
                    f"| {row_label} | {slot_id} | {s['n']:,} "
                    f"| {s['cosine']['mean']:.4f} "
                    f"| {s['mse']['mean']:.6f} |"
                )
                if "top1_accuracy" in s:
                    row = row.rstrip("|") + (
                        f" {s['top1_accuracy']:.4f} "
                        f"| {s['top5_accuracy']:.4f} "
                        f"| {s['top10_accuracy']:.4f} "
                        f"| {s.get('top50_accuracy', 0):.4f} |"
                    )
                add(row)
                first_row = False
        add()

    # --- Plot references ---
    if plot_names:
        next_section = (5 if has_slot_data else 4) if has_per_mol else (4 if has_slot_data else 3)
        add(f"## {next_section}. Visualization")
        add()
        for pname in plot_names:
            add(f"![{pname}]({pname})")
            add()

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Analysis written to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generalization test for mixture unmixing")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to stage2 checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Data split seed")
    parser.set_defaults(data_dir="../data/processed_unmix", batch_size=32, no_matching=False, no_plots=False,
                        ternary=False, split_drug_data=False, categories=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select category order based on mode
    if args.ternary:
        cat_order = TERNARY_CATEGORY_ORDER
        cat_labels = TERNARY_CATEGORY_LABELS
    else:
        cat_order = BINARY_CATEGORY_ORDER
        cat_labels = BINARY_CATEGORY_LABELS

    target_categories = set(args.categories) if args.categories else set(cat_order)

    # Load config and model
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        config_dict = json.load(f)
    cfg = load_config(**config_dict)

    n_holdout = cfg.data.n_holdout_molecules

    print(f"Loading model from {args.checkpoint}")
    model = _load_model(Path(args.checkpoint), cfg, device)

    # Compute scale factor
    scale_factor = get_global_scale_factor(cfg.data)
    print(f"Scale factor: {scale_factor:.6f}")

    # Load combined molecule database (for spectra + SMILES)
    combined_db_path = data_dir / "combined_molecule_database.h5"
    with h5py.File(combined_db_path, "r") as f:
        liq_spectra = torch.from_numpy(f["liq_spectra"][:]).float()
        smiles_list = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["smiles"][:]]
    print(f"Combined database: {len(smiles_list)} molecules, spectra shape {liq_spectra.shape}")

    # Step 1: Molecule categorization (holdout-aware)
    print("\n" + "=" * 60)
    print("STEP 1: Molecule Categorization")
    print("=" * 60)
    categories = build_molecule_categories(
        data_dir, n_holdout=n_holdout, seed=args.seed,
        include_drug_data=cfg.data.include_drug_data,
    )

    # Prepare matching database and refinement matcher
    database = None
    peak_matcher = None
    if not args.no_matching:
        database = LiquidIRDatabase(
            cfg.data, apply_normalization=True, device=device,
            database_file="combined_molecule_database.h5",
        )
        # Refinement matching is disabled so that holdout results use the
        # same identification protocol as all other experiments (cosine retrieval).

    # Collect evaluation data by category, keyed as (source, category)
    # so the same category from different sources is kept separately.
    eval_data: Dict[str, Dict] = {}

    print("\n" + "=" * 60)
    print("STEP 2: Loading Evaluation Data")
    print("=" * 60)

    if args.ternary:
        ternary_by_cat = load_ternary_test_by_category(
            data_dir, categories, liq_spectra, scale_factor, seed=args.seed,
        )
        for cat, data in ternary_by_cat.items():
            if cat in target_categories:
                eval_data[cat] = data
    else:
        # Binary test split
        binary_by_cat = load_binary_test_by_category(
            data_dir, categories, liq_spectra, scale_factor, seed=args.seed,
        )
        for cat, data in binary_by_cat.items():
            if cat in target_categories:
                eval_data[cat] = data

        # Drug mixtures (use "drug:" prefix to avoid overwriting binary results)
        drug_split_seed = args.seed if args.split_drug_data else None
        drug_data = load_drug_mixtures_by_category(
            data_dir, categories, liq_spectra, scale_factor,
            split_seed=drug_split_seed,
        )
        for cat, data in drug_data.items():
            if cat in target_categories:
                eval_data[f"drug:{cat}"] = data

    # Report skipped categories
    for cat in cat_order:
        if cat in target_categories and cat not in eval_data and f"drug:{cat}" not in eval_data:
            print(f"  Skipping {cat_labels[cat]}: no data available")

    # Step 3: Evaluate each category
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)

    # Build ordered list of eval keys (binary categories first, then drug:)
    eval_keys = []
    for cat in cat_order:
        if cat in eval_data:
            eval_keys.append(cat)
    for cat in cat_order:
        dkey = f"drug:{cat}"
        if dkey in eval_data:
            eval_keys.append(dkey)

    all_results: Dict[str, Dict] = {}
    for key in eval_keys:
        data = eval_data[key]
        cat = key.removeprefix("drug:")
        source = data.get("source", "?")
        label = cat_labels.get(cat, cat)
        print(f"\nEvaluating {label} (source={source}, n={len(data['mixture_spectra'])})")
        result = evaluate_category(
            model, data, device,
            categories=categories,
            batch_size=args.batch_size,
            database=database,
            smiles_list=smiles_list if database else None,
            peak_matcher=peak_matcher,
        )
        result["source"] = source
        all_results[key] = result

    # Step 4: Output
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Check if any results have refinement matching
    has_refinement = any("refinement_matching" in r for r in all_results.values())

    # Print table
    col_w = 25 if args.ternary else 15
    header = f"{'Category':<{col_w}} {'Src':<6} {'N':>6} {'Cos':>6} {'Top1':>6}"
    if has_refinement:
        header += f" | {'cR@1':>6} {'cR@5':>6} {'pR@1':>6} {'pR@5':>6} {'Recon':>6}"
    else:
        header += f" {'Top5':>7} {'Top10':>7} {'Top50':>7}"
    print(header)
    print("-" * len(header))

    for key in eval_keys:
        r = all_results[key]
        cat = key.removeprefix("drug:")
        u = r["unmixing"]
        source = r.get("source", "?")
        label = cat_labels.get(cat, cat)
        m = r.get("matching", {})
        row = (
            f"{label:<{col_w}} {source:<6} {r['n_samples']:>6} "
            f"{u['component_cosine']['mean']:>6.3f} "
            f"{m.get('top1_accuracy', 0):>6.3f}"
        )
        if has_refinement:
            rm = r.get("refinement_matching", {})
            row += (
                f" | {rm.get('component_top1_accuracy', 0):>6.3f}"
                f" {rm.get('component_top5_accuracy', 0):>6.3f}"
                f" {rm.get('pair_top1_accuracy', 0):>6.3f}"
                f" {rm.get('pair_top5_accuracy', 0):>6.3f}"
                f" {rm.get('reconstruction_cosine_mean', 0):>6.3f}"
            )
        else:
            row += (
                f" {m.get('top5_accuracy', 0):>7.4f}"
                f" {m.get('top10_accuracy', 0):>7.4f}"
                f" {m.get('top50_accuracy', 0):>7.4f}"
            )
        print(row)

    print("=" * len(header))

    # Per-component breakdown by molecule category
    print("\n" + "=" * 115)
    print("PER-COMPONENT BREAKDOWN (by molecule visibility)")
    print("=" * 115)
    header2 = f"{'Category':<{col_w}} {'Comp':<5} {'N':>6} {'Cos':>6} {'Top1':>6}"
    if has_refinement:
        header2 += f" | {'cR@1':>6} {'cR@5':>6}"
    else:
        header2 += f" {'Top5':>7} {'Top10':>7} {'Top50':>7}"
    print(header2)
    print("-" * len(header2))

    for key in eval_keys:
        r = all_results[key]
        cat = key.removeprefix("drug:")
        label = cat_labels.get(cat, cat)
        source = r.get("source", "?")
        per_mol = r.get("per_molecule_category", {})
        first_row = True
        for mol_cat in ["S12", "S1", "U"]:
            if mol_cat not in per_mol:
                continue
            mc = per_mol[mol_cat]
            row_label = f"{label} [{source}]" if first_row else ""
            row = (
                f"{row_label:<{col_w}} {mol_cat:<5} {mc['n']:>6} "
                f"{mc['cosine']['mean']:>6.3f} "
                f"{mc.get('top1_accuracy', 0):>6.3f}"
            )
            if has_refinement:
                row += (
                    f" | {mc.get('refine_top1_accuracy', 0):>6.3f}"
                    f" {mc.get('refine_top5_accuracy', 0):>6.3f}"
                )
            else:
                if "top1_accuracy" in mc:
                    row += (
                        f" {mc['top5_accuracy']:>7.4f}"
                        f" {mc['top10_accuracy']:>7.4f}"
                        f" {mc.get('top50_accuracy', 0):>7.4f}"
                    )
            print(row)
            first_row = False

    print("=" * 115)

    # Per-slot breakdown (slot bias)
    print("\n" + "=" * 105)
    print("PER-SLOT BREAKDOWN (slot bias)")
    print("=" * 105)
    header3 = f"{'Category':<{col_w}} {'Slot':>4} {'N':>6} {'Cosine':>10} {'MSE':>12}"
    if not args.no_matching:
        header3 += f" {'Top1':>7} {'Top5':>7} {'Top10':>7} {'Top50':>7}"
    print(header3)
    print("-" * 115)

    for key in eval_keys:
        r = all_results[key]
        cat = key.removeprefix("drug:")
        label = cat_labels.get(cat, cat)
        source = r.get("source", "?")
        per_slot = r.get("per_slot", {})
        first_row = True
        for slot_id in sorted(per_slot.keys(), key=int):
            s = per_slot[slot_id]
            row_label = f"{label} [{source}]" if first_row else ""
            row = (
                f"{row_label:<{col_w}} {slot_id:>4} {s['n']:>6} "
                f"{s['cosine']['mean']:>10.4f} "
                f"{s['mse']['mean']:>12.6f}"
            )
            if "top1_accuracy" in s:
                row += (
                    f" {s['top1_accuracy']:>7.4f}"
                    f" {s['top5_accuracy']:>7.4f}"
                    f" {s['top10_accuracy']:>7.4f}"
                    f" {s.get('top50_accuracy', 0):>7.4f}"
                )
            print(row)
            first_row = False

    print("=" * 115)

    # Print metrics legend
    print("\nMetrics:")
    print("  Cos      = Component Cosine Similarity (unmixing quality)")
    print("  Top1     = Cosine per-component R@1")
    if has_refinement:
        print("  cR@1/5   = Refinement per-component R@1/5 (from joint pairs)")
        print("  pR@1/5   = Refinement per-mixture R@1/5 (both GT in pair)")
        print("  Recon    = Region-weighted reconstruction cosine")

    # Save JSON (strip per_sample for file size, keep per_molecule_detail)
    save_results = {}
    for key, r in all_results.items():
        save_r = {k: v for k, v in r.items() if k != "per_sample"}
        save_results[key] = save_r

    meta = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "seed": args.seed,
        "n_holdout_molecules": n_holdout,
        "ternary": args.ternary,
        "split_drug_data": args.split_drug_data,
        "categories": {
            "S12_count": len(categories["S12"]),
            "S1_count": len(categories["S1"]),
            "U_count": len(categories["U"]),
        },
    }
    output = {"metadata": meta, "results": save_results}
    json_path = output_dir / "generalization_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Save per-molecule detail summary (aggregated by co-component category)
    per_mol_summary = {}
    for key, r in all_results.items():
        pmd = r.get("per_molecule_detail", {})
        if not pmd:
            continue
        cat_summary = {}
        for smiles, entries in pmd.items():
            mol_cat = entries[0]["mol_cat"] if entries else "?"
            # Group by co-component category pattern
            by_co_cat = defaultdict(list)
            for e in entries:
                co_key = ",".join(sorted(e["co_cats"])) if e["co_cats"] else "none"
                by_co_cat[co_key].append(e)
            mol_info = {"mol_cat": mol_cat, "total_appearances": len(entries), "by_co_category": {}}
            for co_key, co_entries in by_co_cat.items():
                mol_info["by_co_category"][co_key] = {
                    "n": len(co_entries),
                    "top1_rate": float(np.mean([e["top1"] for e in co_entries])),
                    "top5_rate": float(np.mean([e["top5"] for e in co_entries])),
                    "cosine_mean": float(np.mean([e["cosine"] for e in co_entries])),
                    "cosine_std": float(np.std([e["cosine"] for e in co_entries])),
                }
            cat_summary[smiles] = mol_info
        per_mol_summary[key] = cat_summary

    if per_mol_summary:
        mol_json_path = output_dir / "per_molecule_by_co_category.json"
        with open(mol_json_path, "w") as f:
            json.dump(per_mol_summary, f, indent=2)
        print(f"Per-molecule detail saved to {mol_json_path}")


    # Step 4b: Plots
    plot_names = []
    if not args.no_plots:
        print("\nGenerating plots...")
        if args.ternary:
            plot_names = generate_plots(all_results, output_dir,
                                        category_order=cat_order, category_labels=cat_labels)
        else:
            # Split by source: binary keys have no prefix, drug keys have "drug:" prefix
            binary_results = {k: v for k, v in all_results.items() if not k.startswith("drug:")}
            drug_results = {k.removeprefix("drug:"): v for k, v in all_results.items() if k.startswith("drug:")}
            if binary_results:
                print("  Binary test plots:")
                plot_names.extend(generate_plots(
                    binary_results, output_dir,
                    category_order=cat_order, category_labels=cat_labels,
                    suffix="_binary",
                ))
            if drug_results:
                print("  Drug mixture plots:")
                plot_names.extend(generate_plots(
                    drug_results, output_dir,
                    category_order=cat_order, category_labels=cat_labels,
                    suffix="_drug",
                ))

    # Step 4c: Write analysis markdown
    print("\nWriting analysis markdown...")
    write_analysis_md(
        all_results, categories, n_holdout, output_dir,
        plot_names=plot_names,
        has_matching=not args.no_matching,
        category_order=cat_order,
        category_labels=cat_labels,
    )

    print("\nDone.")



# ---------------------------------------------------------------------------
# Subcommand: aggregate per-molecule results across seeds
# ---------------------------------------------------------------------------

def aggregate_main():
    """Aggregate per_molecule_by_co_category.json across multiple seed runs."""
    parser = argparse.ArgumentParser(
        description="Aggregate per-molecule generalization results across seeds"
    )
    parser.add_argument("--input_dirs", nargs="+", required=True, help="List of seed result directories")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    input_dirs = args.input_dirs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-molecule files from each seed
    all_seed_data = []
    for d in sorted(input_dirs):
        path = Path(d) / "per_molecule_by_co_category.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            all_seed_data.append(json.load(f))
        print(f"Loaded: {path}")

    if not all_seed_data:
        print("ERROR: No data loaded")
        return

    n_seeds = len(all_seed_data)
    print(f"\nAggregating across {n_seeds} seeds")

    # Structure: category_key -> smiles -> co_cat -> list of per-seed stats
    combined = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed_data in all_seed_data:
        for cat_key, molecules in seed_data.items():
            for smiles, mol_info in molecules.items():
                for co_cat, stats in mol_info.get("by_co_category", {}).items():
                    combined[cat_key][smiles][co_cat].append(stats)

    # Aggregate: mean/std across seeds
    aggregated = {}
    for cat_key, molecules in combined.items():
        cat_agg = {}
        for smiles, co_cats in molecules.items():
            mol_entry = {"n_seeds": 0, "by_co_category": {}}
            for co_cat, seed_stats_list in co_cats.items():
                mol_entry["n_seeds"] = max(mol_entry["n_seeds"], len(seed_stats_list))
                top1_rates = [s["top1_rate"] for s in seed_stats_list]
                top5_rates = [s["top5_rate"] for s in seed_stats_list]
                cosines = [s["cosine_mean"] for s in seed_stats_list]
                n_per_seed = [s["n"] for s in seed_stats_list]
                mol_entry["by_co_category"][co_cat] = {
                    "n_per_seed_mean": float(np.mean(n_per_seed)),
                    "n_total": int(np.sum(n_per_seed)),
                    "top1_rate_mean": float(np.mean(top1_rates)),
                    "top1_rate_std": float(np.std(top1_rates)),
                    "top5_rate_mean": float(np.mean(top5_rates)),
                    "top5_rate_std": float(np.std(top5_rates)),
                    "cosine_mean": float(np.mean(cosines)),
                    "cosine_std": float(np.std(cosines)),
                }
            cat_agg[smiles] = mol_entry
        aggregated[cat_key] = cat_agg

    # Save full aggregated results
    with open(output_dir / "per_molecule_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Generate summary markdown
    lines = ["# Per-Molecule Generalization Summary (Aggregated)\n"]
    lines.append(f"Seeds: {n_seeds}\n")

    for cat_key, molecules in aggregated.items():
        lines.append(f"\n## Category: {cat_key}\n")
        lines.append(f"Molecules: {len(molecules)}\n")

        co_cat_population = defaultdict(lambda: {"top1_rates": [], "cosines": [], "mols": []})
        for smiles, mol_info in molecules.items():
            for co_cat, stats in mol_info["by_co_category"].items():
                co_cat_population[co_cat]["top1_rates"].append(stats["top1_rate_mean"])
                co_cat_population[co_cat]["cosines"].append(stats["cosine_mean"])
                co_cat_population[co_cat]["mols"].append(smiles)

        lines.append("\n| Co-Component Cat | N Molecules | Mean R@1 | Std R@1 | Mean Cos | Std Cos |")
        lines.append("|-----------------|------------|---------|---------|---------|---------|")
        for co_cat in sorted(co_cat_population.keys()):
            pop = co_cat_population[co_cat]
            t1 = np.array(pop["top1_rates"])
            cos = np.array(pop["cosines"])
            lines.append(
                f"| {co_cat} | {len(t1)} | {t1.mean():.3f} | {t1.std():.3f} | "
                f"{cos.mean():.4f} | {cos.std():.4f} |"
            )

        # Paired comparison: molecules appearing in multiple co-categories
        multi_co = {s: info for s, info in molecules.items()
                    if len(info["by_co_category"]) > 1}
        if multi_co:
            lines.append(f"\n### Paired comparison (same molecule, different co-component): {len(multi_co)} molecules\n")
            lines.append("| Molecule | Co=S12 R@1 | Co=S1 R@1 | Delta R@1 | Co=S12 Cos | Co=S1 Cos |")
            lines.append("|----------|-----------|----------|-----------|-----------|----------|")

            deltas = []
            for smiles, info in sorted(multi_co.items()):
                by_co = info["by_co_category"]
                s12_r1 = by_co.get("S12", {}).get("top1_rate_mean")
                s1_r1 = by_co.get("S1", {}).get("top1_rate_mean")
                s12_cos = by_co.get("S12", {}).get("cosine_mean")
                s1_cos = by_co.get("S1", {}).get("cosine_mean")
                if s12_r1 is not None and s1_r1 is not None:
                    delta = s12_r1 - s1_r1
                    deltas.append(delta)
                    lines.append(
                        f"| {smiles[:35]} | {s12_r1:.3f} | {s1_r1:.3f} | "
                        f"{delta:+.3f} | {s12_cos:.4f} | {s1_cos:.4f} |"
                    )

            if deltas:
                deltas = np.array(deltas)
                lines.append(f"\n**Mean Delta R@1 (Co=S12 minus Co=S1): {deltas.mean():+.3f} +/- {deltas.std():.3f}**")
                lines.append(f"Positive = S12 co-component helps identification")
                lines.append(f"N molecules: {len(deltas)}, Fraction helped (Delta > 0): {(deltas > 0).mean():.1%}")

    with open(output_dir / "per_molecule_summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary: {output_dir / 'per_molecule_summary.md'}")
    print(f"Full data: {output_dir / 'per_molecule_aggregated.json'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        sys.argv.pop(1)  # Remove 'aggregate' so argparse sees the rest
        aggregate_main()
    else:
        main()