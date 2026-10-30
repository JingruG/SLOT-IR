#!/usr/bin/env python3
"""Evaluate the baselines on the binary, drug and ternary test sets, overall and per molecule category.

Usage (normally called by run_fair_baselines.py):
    python evaluate_baselines.py --methods nnls --datasets binary drug ternary --output_dir results/fair/test/classical
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Path setup: make slot_ir importable
# ---------------------------------------------------------------------------
_BASELINES_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINES_ROOT.parent
sys.path.insert(0, str(_BASELINES_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "slot_ir"))

from utils.data_loader import compute_global_scale_factor

# Imports from slot_ir (via sys.path)
from src.data import split_indices
from src.matching import LiquidIRDatabase

# Category ordering and labels
CATEGORY_ORDER = [
    "S12_S12", "S12_S1", "S1_S1",
    "S12_U", "S1_U", "U_U",
]
CATEGORY_LABELS = {
    "S12_S12": "(S12, S12)",
    "S12_S1":  "(S12, S1)",
    "S1_S1":   "(S1, S1)",
    "S12_U":   "(S12, U)",
    "S1_U":    "(S1, U)",
    "U_U":     "(U, U)",
}
# Ternary categories ordered by difficulty
TERNARY_CATEGORY_ORDER = [
    "S12_S12_S12", "S12_S12_S1", "S12_S1_S1", "S1_S1_S1",
]
TERNARY_CATEGORY_LABELS = {
    "S12_S12_S12": "(S12, S12, S12)",
    "S12_S12_S1":  "(S12, S12, S1)",
    "S12_S1_S1":   "(S12, S1, S1)",
    "S1_S1_S1":    "(S1, S1, S1)",
}

_CAT_RANK = {"S12": 0, "S1": 1, "U": 2}


# ---------------------------------------------------------------------------
# Molecule categorization (reuses slot_ir logic)
# ---------------------------------------------------------------------------

def build_molecule_categories(data_dir: Path, seed: int = 42) -> Dict[str, Set[int]]:
    """Categorize every molecule index into S12, S1, or U."""
    with h5py.File(data_dir / "molecule_spectra_database.h5", "r") as f:
        n_stage1 = f["liq_spectra"].shape[0]

    with h5py.File(data_dir / "mix_ml_instances.h5", "r") as f:
        component_indices = f["component_indices"][:]
        n_mix = component_indices.shape[0]

    train_idx, _, _ = split_indices(n_mix, seed=seed)
    s12: Set[int] = {int(m) for idx in train_idx for m in component_indices[idx]}

    all_stage1 = set(range(n_stage1))
    s1 = all_stage1 - s12

    combined_db_path = data_dir / "combined_molecule_database.h5"
    with h5py.File(combined_db_path, "r") as f:
        n_combined = f["liq_spectra"].shape[0]
    u = set(range(n_stage1, n_combined)) - s12

    print("Molecule categorization:")
    print(f"  S12 (seen in Stage 1+2 training):       {len(s12)}")
    print(f"  S1  (Stage 1 only / held out):          {len(s1)}")
    print(f"  U   (Unseen in Stage 1):                {len(u)}")

    return {"S12": s12, "S1": s1, "U": u}


def classify_molecule(mol_idx: int, categories: Dict[str, Set[int]]) -> str:
    if mol_idx in categories["S12"]:
        return "S12"
    elif mol_idx in categories["S1"]:
        return "S1"
    elif mol_idx in categories["U"]:
        return "U"
    raise ValueError(f"Molecule index {mol_idx} not found in any category")


def classify_pair(mol_a: int, mol_b: int, categories: Dict[str, Set[int]]) -> str:
    cat_a = classify_molecule(mol_a, categories)
    cat_b = classify_molecule(mol_b, categories)
    if _CAT_RANK[cat_a] <= _CAT_RANK[cat_b]:
        return f"{cat_a}_{cat_b}"
    return f"{cat_b}_{cat_a}"


def classify_triple(
    mol_a: int, mol_b: int, mol_c: int,
    categories: Dict[str, Set[int]],
) -> str:
    cats = sorted(
        [classify_molecule(m, categories) for m in (mol_a, mol_b, mol_c)],
        key=lambda c: _CAT_RANK[c],
    )
    return "_".join(cats)


# ---------------------------------------------------------------------------
# Numpy re-implementation of _per_sample_metrics
# ---------------------------------------------------------------------------

def per_sample_metrics_np(
    pred_components: np.ndarray,
    target_components: np.ndarray,
    mixture: np.ndarray,
) -> List[Dict]:
    """Compute per-sample unmixing metrics (numpy, mirrors evaluate.py)."""
    bsz = pred_components.shape[0]
    k_target = target_components.shape[1]
    results = []

    for b in range(bsz):
        # L2 cost matrix for Hungarian
        cost = np.zeros((pred_components.shape[1], k_target))
        for i in range(pred_components.shape[1]):
            for j in range(k_target):
                cost[i, j] = np.sqrt(np.sum((pred_components[b, i] - target_components[b, j]) ** 2))
        row_ind, col_ind = linear_sum_assignment(cost)

        comp_cosines = []
        comp_mses = []
        for ki in range(len(row_ind)):
            p = pred_components[b, row_ind[ki]]
            t = target_components[b, col_ind[ki]]
            pn = p / (np.linalg.norm(p) + 1e-8)
            tn = t / (np.linalg.norm(t) + 1e-8)
            comp_cosines.append(float(np.dot(pn, tn)))
            comp_mses.append(float(np.mean((p - t) ** 2)))

        # Mixture reconstruction
        pred_mix = pred_components[b, row_ind].sum(axis=0)
        true_mix = mixture[b]
        pm = pred_mix / (np.linalg.norm(pred_mix) + 1e-8)
        tm = true_mix / (np.linalg.norm(true_mix) + 1e-8)

        results.append({
            "component_cosine": float(np.mean(comp_cosines)),
            "component_mse": float(np.mean(comp_mses)),
            "per_component_cosines": comp_cosines,
            "per_component_mses": comp_mses,
            "matched_pred_order": row_ind.tolist(),
            "matched_target_order": col_ind.tolist(),
            "mixture_cosine": float(np.dot(pm, tm)),
            "mixture_mse": float(np.mean((pred_mix - true_mix) ** 2)),
        })

    return results


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_dataset_raw(
    data_dir: Path,
    dataset_file: str,
    liq_spectra: np.ndarray,
    scale_factor: float,
    indices: Optional[np.ndarray] = None,
) -> Dict:
    """Load a mixture dataset (or subset by indices), apply scaling."""
    mix_path = data_dir / dataset_file
    with h5py.File(mix_path, "r") as f:
        mixture_spectra = np.array(f["mixture_spectra"][:], dtype=np.float32)
        component_indices = f["component_indices"][:]

    if indices is not None:
        mixture_spectra = mixture_spectra[indices]
        component_indices = component_indices[indices]

    n_components = component_indices.shape[1]
    sel_mix = np.clip(mixture_spectra * scale_factor, 0, 1)

    comp_list = []
    for k in range(n_components):
        comp_list.append(np.clip(liq_spectra[component_indices[:, k]] * scale_factor, 0, 1))
    sel_components = np.stack(comp_list, axis=1)

    return {
        "mixture_spectra": sel_mix,
        "component_spectra": sel_components,
        "component_indices": component_indices,
    }


def load_binary_test(
    data_dir: Path, liq_spectra: np.ndarray, scale_factor: float, seed: int = 42,
    split: str = "test",
) -> Dict:
    """Load the binary test split (or the validation split, for model selection)."""
    mix_path = data_dir / "mix_ml_instances.h5"
    with h5py.File(mix_path, "r") as f:
        n_mix = f["mixture_spectra"].shape[0]
    _, val_idx, test_idx = split_indices(n_mix, seed=seed)
    if split == "val":
        test_idx = val_idx
    return _load_dataset_raw(data_dir, "mix_ml_instances.h5", liq_spectra, scale_factor, np.array(test_idx))


def load_drug_all(
    data_dir: Path, liq_spectra: np.ndarray, scale_factor: float,
) -> Dict:
    """Load entire drug dataset (OOD, no split)."""
    return _load_dataset_raw(data_dir, "mix_drugs_ml_instances.h5", liq_spectra, scale_factor)


def load_ternary_test(
    data_dir: Path, liq_spectra: np.ndarray, scale_factor: float, seed: int = 42,
    split: str = "test",
) -> Dict:
    """Load the ternary test split (or the validation split, for model selection)."""
    ternary_path = data_dir / "mix_ternary_ml_instances.h5"
    if not ternary_path.exists():
        return {}
    with h5py.File(ternary_path, "r") as f:
        n_mix = f["mixture_spectra"].shape[0]
    _, val_idx, test_idx = split_indices(n_mix, seed=seed)
    if split == "val":
        test_idx = val_idx
    return _load_dataset_raw(data_dir, "mix_ternary_ml_instances.h5", liq_spectra, scale_factor, np.array(test_idx))


# ---------------------------------------------------------------------------
# Data loading by category
# ---------------------------------------------------------------------------

def _split_by_category_binary(
    data: Dict,
    categories: Dict[str, Set[int]],
) -> Dict[str, Dict]:
    """Split loaded binary data into per-category subsets."""
    comp_indices = data["component_indices"]
    by_category: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(comp_indices)):
        mol_a, mol_b = int(comp_indices[i, 0]), int(comp_indices[i, 1])
        try:
            cat = classify_pair(mol_a, mol_b, categories)
            by_category[cat].append(i)
        except ValueError:
            continue

    results = {}
    for cat, idxs in by_category.items():
        sel = np.array(idxs)
        results[cat] = {
            "mixture_spectra": data["mixture_spectra"][sel],
            "component_spectra": data["component_spectra"][sel],
            "component_indices": data["component_indices"][sel],
        }
    return results


def _split_by_category_ternary(
    data: Dict,
    categories: Dict[str, Set[int]],
) -> Dict[str, Dict]:
    """Split loaded ternary data into per-category subsets."""
    comp_indices = data["component_indices"]
    by_category: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(comp_indices)):
        mols = [int(comp_indices[i, k]) for k in range(3)]
        try:
            cat = classify_triple(*mols, categories=categories)
            by_category[cat].append(i)
        except ValueError:
            continue

    results = {}
    for cat, idxs in by_category.items():
        sel = np.array(idxs)
        results[cat] = {
            "mixture_spectra": data["mixture_spectra"][sel],
            "component_spectra": data["component_spectra"][sel],
            "component_indices": data["component_indices"][sel],
        }
    return results


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_on_data(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    data: Dict,
    categories: Dict[str, Set[int]],
    batch_size: int = 64,
    database: Optional[LiquidIRDatabase] = None,
    smiles_list: Optional[List[str]] = None,
) -> Dict:
    """Evaluate a baseline on a dataset."""
    mixture = data["mixture_spectra"]
    components = data["component_spectra"]
    comp_indices = data["component_indices"]
    n_samples = len(mixture)
    n_components = components.shape[1]

    all_metrics: List[Dict] = []

    # Per-component matching hits (per sample, per component)
    # sample_comp_hits[sample_idx][ki] = {"top1": 0/1, "top5": 0/1, "top10": 0/1}
    sample_comp_hits: List[List[Dict]] = []

    per_mol_cat_cosines: Dict[str, List[float]] = defaultdict(list)
    per_mol_cat_mses: Dict[str, List[float]] = defaultdict(list)
    per_mol_cat_top1: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_top5: Dict[str, List[int]] = defaultdict(list)
    per_mol_cat_top10: Dict[str, List[int]] = defaultdict(list)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_mix = mixture[start:end]
        batch_comp = components[start:end]

        pred_components = predict_fn(batch_mix)
        batch_metrics = per_sample_metrics_np(pred_components, batch_comp, batch_mix)
        all_metrics.extend(batch_metrics)

        for bi, m in enumerate(batch_metrics):
            sample_idx = start + bi
            target_order = m["matched_target_order"]
            pred_order = m["matched_pred_order"]

            comp_hits = []
            for ki, tgt_ki in enumerate(target_order):
                mol_idx = int(comp_indices[sample_idx, tgt_ki])
                try:
                    mol_cat = classify_molecule(mol_idx, categories)
                except ValueError:
                    mol_cat = "?"

                per_mol_cat_cosines[mol_cat].append(m["per_component_cosines"][ki])
                per_mol_cat_mses[mol_cat].append(m["per_component_mses"][ki])

                hit = {"top1": 0, "top5": 0, "top10": 0}
                if database is not None and smiles_list is not None:
                    gt_smiles = smiles_list[mol_idx]
                    pred_slot_idx = pred_order[ki]
                    query = torch.from_numpy(pred_components[bi, pred_slot_idx]).float()
                    matches = database.find_best_match(query, top_k=10)
                    hit["top1"] = 1 if matches and matches[0][0] == gt_smiles else 0
                    hit["top5"] = 1 if gt_smiles in {s for s, _, _ in matches[:5]} else 0
                    hit["top10"] = 1 if gt_smiles in {s for s, _, _ in matches} else 0

                    per_mol_cat_top1[mol_cat].append(hit["top1"])
                    per_mol_cat_top5[mol_cat].append(hit["top5"])
                    per_mol_cat_top10[mol_cat].append(hit["top10"])

                comp_hits.append(hit)
            sample_comp_hits.append(comp_hits)

    # ── Aggregate unmixing metrics ──
    metric_keys = ["component_cosine", "component_mse", "mixture_cosine", "mixture_mse"]
    agg = {}
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # ── Aggregate matching metrics ──
    matching = {}
    if database is not None and smiles_list is not None:
        comp_top1_list = []
        comp_top5_list = []
        comp_top10_list = []
        mix_top1_list = []
        mix_top5_list = []
        mix_top10_list = []

        for hits in sample_comp_hits:
            for h in hits:
                comp_top1_list.append(h["top1"])
                comp_top5_list.append(h["top5"])
                comp_top10_list.append(h["top10"])
            # Per-mixture: ALL components must be correct
            mix_top1_list.append(1 if all(h["top1"] for h in hits) else 0)
            mix_top5_list.append(1 if all(h["top5"] for h in hits) else 0)
            mix_top10_list.append(1 if all(h["top10"] for h in hits) else 0)

        matching = {
            "component_top1_accuracy": float(np.mean(comp_top1_list)),
            "component_top5_accuracy": float(np.mean(comp_top5_list)),
            "component_top10_accuracy": float(np.mean(comp_top10_list)),
            "mixture_top1_accuracy": float(np.mean(mix_top1_list)),
            "mixture_top5_accuracy": float(np.mean(mix_top5_list)),
            "mixture_top10_accuracy": float(np.mean(mix_top10_list)),
        }

    # ── Per-molecule-category breakdown ──
    per_mol_cat = {}
    for cat_label in ["S12", "S1", "U"]:
        if cat_label in per_mol_cat_cosines:
            cos_vals = per_mol_cat_cosines[cat_label]
            mse_vals = per_mol_cat_mses[cat_label]
            entry: Dict = {
                "n": len(cos_vals),
                "cosine": {"mean": float(np.mean(cos_vals)), "std": float(np.std(cos_vals))},
                "mse": {"mean": float(np.mean(mse_vals)), "std": float(np.std(mse_vals))},
            }
            if cat_label in per_mol_cat_top1:
                entry["top1_accuracy"] = float(np.mean(per_mol_cat_top1[cat_label]))
                entry["top5_accuracy"] = float(np.mean(per_mol_cat_top5[cat_label]))
                entry["top10_accuracy"] = float(np.mean(per_mol_cat_top10[cat_label]))
            per_mol_cat[cat_label] = entry

    result: Dict = {
        "n_samples": n_samples,
        "unmixing": agg,
        "per_molecule_category": per_mol_cat,
    }
    if matching:
        result["matching"] = matching

    return result


# ---------------------------------------------------------------------------
# Baseline model loading
# ---------------------------------------------------------------------------

def _load_classical_model(
    method: str,
    data_dir: str,
    n_components: int,
    scale_factor: float,
    seed: int = 42,
    dataset_type: str = "binary",
    models_dir: Optional[str] = None,
    sunsal_lambda: float = 1e-3,
    device: Optional[str] = None,
):
    """Fit a classical baseline on the correct training split and return predict_fn."""
    if dataset_type == "ternary":
        mix_file = "mix_ternary_ml_instances.h5"
    else:
        mix_file = "mix_ml_instances.h5"

    mix_path = Path(data_dir) / mix_file
    with h5py.File(mix_path, "r") as f:
        all_mixtures = np.array(f["mixture_spectra"][:], dtype=np.float32)
        all_comp_idx = np.array(f["component_indices"][:])
        n_mix = all_mixtures.shape[0]

    train_idx, val_idx, _ = split_indices(n_mix, seed=seed)
    train_val_idx = train_idx + val_idx
    train_val_mixtures = np.clip(all_mixtures[train_val_idx] * scale_factor, 0, 1)

    if method in DICTIONARY_METHODS:
        # One learned spectrum per molecule; no reference library. Only the
        # number of distinct molecules in the training mixtures is used.
        from classical.dictionary_unmixing import MCRALSDictionary, VCAFCLSDictionary
        # Learned from the training split only, like the deep learning models
        # (the validation split is kept for model selection).
        train_mixtures = np.clip(all_mixtures[train_idx] * scale_factor, 0, 1)
        n_spectra = len(np.unique(all_comp_idx[train_idx]))
        save_path = None
        if models_dir is not None:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(models_dir) / f"{method}_{dataset_type}_spectra.npz"
        cls = MCRALSDictionary if method == "mcr_als_dict" else VCAFCLSDictionary
        model = cls(n_components=n_components, n_spectra=n_spectra, seed=seed)
        print(f"  {method}: {n_spectra} spectra from {len(train_mixtures)} training mixtures")
        model.fit(train_mixtures, save_path=save_path)
        return lambda x: model.predict_batch(x)["components"]

    # The 8,550-spectrum identification library (includes the drug molecules)
    db_file = "combined_molecule_database.h5"
    db_path = Path(data_dir) / db_file
    print(f"  Library for {method}: {db_file}")
    with h5py.File(db_path, "r") as f:
        single_phase = np.clip(
            np.array(f["liq_spectra"][:], dtype=np.float32) * scale_factor, 0, 1
        )

    if method == "sunsal_admm":
        from classical.sunsal_admm import SUnSALADMM
        model = SUnSALADMM(n_components=n_components, lambda_reg=sunsal_lambda, device=device)
        model.fit(reference_spectra=single_phase)
    elif method == "nnls":
        from classical.nnls import NNLS
        model = NNLS(n_components=n_components)
        model.fit(train_val_mixtures, reference_spectra=single_phase)
    else:
        raise ValueError(f"Unknown classical method: {method}")

    return lambda x: model.predict_batch(x)["components"]


def _load_dl_model(
    method: str,
    dataset_type: str,
    n_components: int,
    models_dir: str,
    device: str = "cpu",
):
    """Load a saved DL baseline model and return predict_fn."""
    models_path = Path(models_dir)

    if dataset_type == "ternary":
        model_file = f"{method}_{dataset_type}_model.pt"
    else:
        model_file = f"{method}_{dataset_type}_mix_ml_model.pt"
    model_path = models_path / model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if method == "cnnaeu":
        from deep_learning.cnnaeu import CNNAEU
        model = CNNAEU(n_components=n_components, input_dim=1250)
    elif method == "deeptrans_hsu":
        from deep_learning.deeptrans_hsu import DeepTransHSU
        model = DeepTransHSU(n_components=n_components, input_dim=1250)
    elif method == "cycunet":
        from deep_learning.cycunet import CyCUNet
        model = CyCUNet(n_components=n_components, input_dim=1250)
    else:
        raise ValueError(f"Unknown DL method: {method}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    model = model.to(device)
    print(f"  Loaded {method} from {model_path}")

    def predict_fn(mixtures_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(mixtures_np).float().to(device)
            if method == "cycunet":
                out = model(t)
                return out["components"].cpu().numpy()
            else:
                _, components, _ = model(t)
                return components.cpu().numpy()

    return predict_fn


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

DICTIONARY_METHODS = ["mcr_als_dict", "vca_fcls_dict"]
CLASSICAL_METHODS = ["nnls", "sunsal_admm"] + DICTIONARY_METHODS
DL_METHODS = ["cnnaeu", "deeptrans_hsu", "cycunet"]
ALL_METHODS = CLASSICAL_METHODS + DL_METHODS

METHOD_DISPLAY_NAMES = {
    "mcr_als_dict": "MCR-ALS",
    "vca_fcls_dict": "VCA+FCLS",
    "sunsal_admm": "SUnSAL",
    "nnls": "NNLS",
    "cnnaeu": "CNNAEU",
    "deeptrans_hsu": "DeepTrans-HSU",
    "cycunet": "CyCU-Net",
}


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_overall_table(results: Dict[str, Dict], has_matching: bool) -> None:
    """Print overall results for each dataset (binary_test, drug, ternary_test)."""
    header = f"  {'Dataset':<18} {'N':>6} {'Comp Cos':>10} {'Comp MSE':>12} {'Mix Cos':>10}"
    if has_matching:
        header += f" {'cR@1':>7} {'cR@5':>7} {'cR@10':>7} {'mR@1':>7} {'mR@5':>7} {'mR@10':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ds_name in ["binary_test", "drug", "ternary_test"]:
        if ds_name not in results:
            continue
        r = results[ds_name]
        u = r["unmixing"]
        row = (
            f"  {ds_name:<18} {r['n_samples']:>6} "
            f"{u['component_cosine']['mean']:>10.4f} "
            f"{u['component_mse']['mean']:>12.6f} "
            f"{u['mixture_cosine']['mean']:>10.4f}"
        )
        if "matching" in r:
            m = r["matching"]
            row += (
                f" {m['component_top1_accuracy']:>7.4f}"
                f" {m['component_top5_accuracy']:>7.4f}"
                f" {m['component_top10_accuracy']:>7.4f}"
                f" {m['mixture_top1_accuracy']:>7.4f}"
                f" {m['mixture_top5_accuracy']:>7.4f}"
                f" {m['mixture_top10_accuracy']:>7.4f}"
            )
        print(row)


def print_category_table(
    all_results: Dict[str, Dict],
    category_order: List[str],
    category_labels: Dict[str, str],
    has_matching: bool,
) -> None:
    header = f"  {'Category':<18} {'N':>6} {'Comp Cos':>10} {'Comp MSE':>12} {'Mix Cos':>10}"
    if has_matching:
        header += f" {'cR@1':>7} {'cR@5':>7} {'cR@10':>7} {'mR@1':>7} {'mR@5':>7} {'mR@10':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cat in category_order:
        if cat not in all_results:
            continue
        r = all_results[cat]
        u = r["unmixing"]
        row = (
            f"  {category_labels.get(cat, cat):<18} {r['n_samples']:>6} "
            f"{u['component_cosine']['mean']:>10.4f} "
            f"{u['component_mse']['mean']:>12.6f} "
            f"{u['mixture_cosine']['mean']:>10.4f}"
        )
        if "matching" in r:
            m = r["matching"]
            row += (
                f" {m['component_top1_accuracy']:>7.4f}"
                f" {m['component_top5_accuracy']:>7.4f}"
                f" {m['component_top10_accuracy']:>7.4f}"
                f" {m['mixture_top1_accuracy']:>7.4f}"
                f" {m['mixture_top5_accuracy']:>7.4f}"
                f" {m['mixture_top10_accuracy']:>7.4f}"
            )
        print(row)


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------

def save_method_results(
    output_dir: Path,
    method: str,
    method_results: Dict,
    metadata: Dict,
) -> Path:
    """Save one method's results to disk immediately."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {"metadata": metadata, "results": method_results}
    json_path = output_dir / f"{method}_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    return json_path


def save_combined_results(
    output_dir: Path,
    all_method_results: Dict[str, Dict],
    metadata: Dict,
) -> Path:
    """Save combined results for all methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {"metadata": metadata, "results": all_method_results}
    json_path = output_dir / "baseline_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified baseline evaluation with overall + per-category metrics",
    )
    parser.add_argument("--data_dir", type=str, default="../data/processed_unmix")
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        help="Methods to evaluate (e.g. cnnaeu nnls) or 'all'",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["binary", "drug"],
        choices=["binary", "drug", "ternary"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--models_dir", type=str, default="results/models")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="val = evaluate on the validation split, for model selection")
    parser.add_argument("--sunsal_lambda", type=float, default=1e-3)
    args = parser.parse_args()
    if args.split == "val" and "drug" in args.datasets:
        args.datasets = [d for d in args.datasets if d != "drug"]
        print("--split val: drug set skipped (it has no validation split)")

    args.seed = 42  # data split seed
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ALL_METHODS if "all" in args.methods else args.methods
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    has_matching = True

    # Compute scale factor
    scale_factor = compute_global_scale_factor(str(data_dir))
    print(f"Global scale factor: {scale_factor:.6f}")

    # Load combined molecule database
    combined_db_path = data_dir / "combined_molecule_database.h5"
    with h5py.File(combined_db_path, "r") as f:
        liq_spectra = np.array(f["liq_spectra"][:], dtype=np.float32)
        smiles_list = [s.decode("utf-8") if isinstance(s, bytes) else str(s)
                       for s in f["smiles"][:]]
    print(f"Combined database: {len(smiles_list)} molecules, spectra shape {liq_spectra.shape}")

    # Build molecule categories
    print("\n" + "=" * 60)
    print("STEP 1: Molecule Categorization")
    print("=" * 60)
    categories = build_molecule_categories(data_dir, seed=args.seed)

    # Build matching database
    database = None
    if has_matching:
        from src.config import DataConfig
        tmp_cfg = DataConfig(processed_unmix_dir=data_dir)
        database = LiquidIRDatabase(
            tmp_cfg, apply_normalization=True, device=device,
            database_file="combined_molecule_database.h5",
        )

    # Load evaluation data
    print("\n" + "=" * 60)
    print("STEP 2: Loading Evaluation Data")
    print("=" * 60)

    # Overall datasets (entire binary_test / drug / ternary_test)
    overall_data: Dict[str, Dict] = {}
    if "binary" in args.datasets:
        overall_data["binary_test"] = load_binary_test(
            data_dir, liq_spectra, scale_factor, seed=args.seed, split=args.split,
        )
        print(f"  binary_test: {overall_data['binary_test']['mixture_spectra'].shape[0]} samples")

    if "drug" in args.datasets:
        drug_path = data_dir / "mix_drugs_ml_instances.h5"
        if drug_path.exists():
            overall_data["drug"] = load_drug_all(data_dir, liq_spectra, scale_factor)
            print(f"  drug (all):  {overall_data['drug']['mixture_spectra'].shape[0]} samples")

    if "ternary" in args.datasets:
        ternary_data = load_ternary_test(data_dir, liq_spectra, scale_factor, seed=args.seed,
                                         split=args.split)
        if ternary_data:
            overall_data["ternary_test"] = ternary_data
            print(f"  ternary_test: {overall_data['ternary_test']['mixture_spectra'].shape[0]} samples")

    # Per-category splits (from the same data)
    binary_by_cat: Dict[str, Dict] = {}
    drug_by_cat: Dict[str, Dict] = {}
    ternary_by_cat: Dict[str, Dict] = {}

    if "binary_test" in overall_data:
        binary_by_cat = _split_by_category_binary(overall_data["binary_test"], categories)
        print(f"\n  Binary test per-category:")
        for cat in CATEGORY_ORDER:
            if cat in binary_by_cat:
                print(f"    {CATEGORY_LABELS[cat]}: {len(binary_by_cat[cat]['mixture_spectra'])}")

    if "drug" in overall_data:
        drug_by_cat = _split_by_category_binary(overall_data["drug"], categories)
        print(f"\n  Drug per-category:")
        for cat in CATEGORY_ORDER:
            if cat in drug_by_cat:
                print(f"    {CATEGORY_LABELS[cat]}: {len(drug_by_cat[cat]['mixture_spectra'])}")

    if "ternary_test" in overall_data:
        ternary_by_cat = _split_by_category_ternary(overall_data["ternary_test"], categories)
        print(f"\n  Ternary test per-category:")
        for cat in TERNARY_CATEGORY_ORDER:
            if cat in ternary_by_cat:
                print(f"    {TERNARY_CATEGORY_LABELS[cat]}: {len(ternary_by_cat[cat]['mixture_spectra'])}")

    # Metadata (shared across saves)
    metadata = {
        "seed": args.seed,
        "split": args.split,
        "library": "combined",
        "sunsal_lambda": args.sunsal_lambda,
        "datasets": args.datasets,
        "methods": methods,
        "scale_factor": scale_factor,
        "categories": {
            "S12_count": len(categories["S12"]),
            "S1_count": len(categories["S1"]),
            "U_count": len(categories["U"]),
        },
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Evaluate each method
    # ═══════════════════════════════════════════════════════════════════════
    all_method_results: Dict[str, Dict] = {}

    for method in methods:
        print("\n" + "=" * 80)
        print(f"EVALUATING: {METHOD_DISPLAY_NAMES.get(method, method)}")
        print("=" * 80)

        method_results: Dict[str, Dict] = {}
        t_start = time.time()

        try:
            is_classical = method in CLASSICAL_METHODS

            # ── Binary / Drug evaluation (n_components=2) ──
            has_binary_or_drug = "binary_test" in overall_data or "drug" in overall_data
            if has_binary_or_drug:
                n_comp = 2
                if is_classical:
                    print(f"  Fitting {method} (binary, K={n_comp})...")
                    predict_fn = _load_classical_model(
                        method, str(data_dir), n_comp, scale_factor, seed=args.seed,
                        models_dir=args.models_dir,
                        sunsal_lambda=args.sunsal_lambda, device=device,
                    )
                else:
                    predict_fn = _load_dl_model(
                        method, "binary", n_comp, args.models_dir, device=device,
                    )

                # Overall binary_test
                if "binary_test" in overall_data:
                    print(f"\n  Evaluating on binary_test (n={overall_data['binary_test']['mixture_spectra'].shape[0]})...")
                    r = evaluate_on_data(
                        predict_fn, overall_data["binary_test"], categories, args.batch_size,
                        database=database, smiles_list=smiles_list if database else None,
                    )
                    method_results["binary_test"] = r
                    _print_quick_summary("binary_test", r, has_matching)

                    # Per-category
                    for cat in CATEGORY_ORDER:
                        if cat in binary_by_cat:
                            key = f"binary:{cat}"
                            r = evaluate_on_data(
                                predict_fn, binary_by_cat[cat], categories, args.batch_size,
                                database=database, smiles_list=smiles_list if database else None,
                            )
                            method_results[key] = r

                # Overall drug
                if "drug" in overall_data:
                    print(f"\n  Evaluating on drug (n={overall_data['drug']['mixture_spectra'].shape[0]})...")
                    r = evaluate_on_data(
                        predict_fn, overall_data["drug"], categories, args.batch_size,
                        database=database, smiles_list=smiles_list if database else None,
                    )
                    method_results["drug"] = r
                    _print_quick_summary("drug", r, has_matching)

                    # Per-category
                    for cat in CATEGORY_ORDER:
                        if cat in drug_by_cat:
                            key = f"drug:{cat}"
                            r = evaluate_on_data(
                                predict_fn, drug_by_cat[cat], categories, args.batch_size,
                                database=database, smiles_list=smiles_list if database else None,
                            )
                            method_results[key] = r

            # ── Ternary evaluation (n_components=3) ──
            if "ternary_test" in overall_data:
                n_comp = 3
                if is_classical:
                    print(f"\n  Fitting {method} (ternary, K={n_comp})...")
                    predict_fn_ternary = _load_classical_model(
                        method, str(data_dir), n_comp, scale_factor,
                        seed=args.seed, dataset_type="ternary",
                        models_dir=args.models_dir,
                        sunsal_lambda=args.sunsal_lambda, device=device,
                    )
                else:
                    predict_fn_ternary = _load_dl_model(
                        method, "ternary", n_comp, args.models_dir, device=device,
                    )

                print(f"\n  Evaluating on ternary_test (n={overall_data['ternary_test']['mixture_spectra'].shape[0]})...")
                r = evaluate_on_data(
                    predict_fn_ternary, overall_data["ternary_test"], categories, args.batch_size,
                    database=database, smiles_list=smiles_list if database else None,
                )
                method_results["ternary_test"] = r
                _print_quick_summary("ternary_test", r, has_matching)

                # Per-category
                for cat in TERNARY_CATEGORY_ORDER:
                    if cat in ternary_by_cat:
                        key = f"ternary:{cat}"
                        r = evaluate_on_data(
                            predict_fn_ternary, ternary_by_cat[cat], categories, args.batch_size,
                            database=database, smiles_list=smiles_list if database else None,
                        )
                        method_results[key] = r

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            method_results["error"] = str(e)

        elapsed = time.time() - t_start
        method_results["_elapsed_seconds"] = round(elapsed, 1)
        all_method_results[method] = method_results

        # ── Save this method's results to disk immediately ──
        json_path = save_method_results(output_dir, method, method_results, metadata)
        print(f"\n  ✓ Saved {method} results to {json_path} ({elapsed:.1f}s)")

        # ── Print full summary for this method ──
        _print_method_summary(method, method_results, has_matching)

    # ── Save combined results ──
    combined_path = save_combined_results(output_dir, all_method_results, metadata)
    print(f"\n{'='*80}")
    print(f"All results saved to {combined_path}")
    print("Done.")


def _print_quick_summary(ds_name: str, r: Dict, has_matching: bool) -> None:
    """Print a one-line summary during evaluation."""
    u = r["unmixing"]
    line = f"    comp_cos={u['component_cosine']['mean']:.4f}  mix_cos={u['mixture_cosine']['mean']:.4f}"
    if "matching" in r:
        m = r["matching"]
        line += f"  cR@1={m['component_top1_accuracy']:.4f}  mR@1={m['mixture_top1_accuracy']:.4f}"
    print(line)


def _print_method_summary(method: str, method_results: Dict, has_matching: bool) -> None:
    """Print formatted summary tables for one method."""
    if "error" in method_results and isinstance(method_results["error"], str):
        return

    display_name = METHOD_DISPLAY_NAMES.get(method, method)
    print(f"\n{'─'*80}")
    print(f"SUMMARY: {display_name}")
    print(f"{'─'*80}")

    # Overall table
    overall = {k: v for k, v in method_results.items()
               if k in ("binary_test", "drug", "ternary_test")}
    if overall:
        print("\n  Overall:")
        print_overall_table(overall, has_matching)

    # Binary per-category
    binary_cats = {k.split(":", 1)[1]: v for k, v in method_results.items()
                   if k.startswith("binary:") and k.split(":", 1)[1] in CATEGORY_ORDER}
    if binary_cats:
        print(f"\n  Binary test per-category:")
        print_category_table(binary_cats, CATEGORY_ORDER, CATEGORY_LABELS, has_matching)

    # Drug per-category
    drug_cats = {k.split(":", 1)[1]: v for k, v in method_results.items()
                 if k.startswith("drug:") and k.split(":", 1)[1] in CATEGORY_ORDER}
    if drug_cats:
        print(f"\n  Drug per-category:")
        print_category_table(drug_cats, CATEGORY_ORDER, CATEGORY_LABELS, has_matching)

    # Ternary per-category
    ternary_cats = {k.split(":", 1)[1]: v for k, v in method_results.items()
                    if k.startswith("ternary:") and k.split(":", 1)[1] in TERNARY_CATEGORY_ORDER}
    if ternary_cats:
        print(f"\n  Ternary test per-category:")
        print_category_table(ternary_cats, TERNARY_CATEGORY_ORDER, TERNARY_CATEGORY_LABELS, has_matching)

    print(f"\n  Metrics: cR@k = per-component Recall@k, mR@k = per-mixture Recall@k (all components correct)")


if __name__ == "__main__":
    main()
