#!/usr/bin/env python
"""Ablation experiment runner for slot_ir Stage 2 (multi-seed).

Usage (from slot_ir/):
    python run_ablations.py --experiments plus_drug plus_synthetic
    python run_ablations.py --config configs/ablation/no_s1_tuned.json --experiments no_s1 --results-dir results/no_s1_tuned
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from slot_ir package
# ---------------------------------------------------------------------------
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from src.config import load_config, SLOT_UNMIXER_ROOT
from src.data import MixtureDataset, split_indices
from src.training import train_stage
from src.evaluate import evaluate_checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_SPLIT_SEED = 42
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_RESULTS_DIR = SLOT_UNMIXER_ROOT / "results/ablation"

# Experiment definitions: each entry maps a name to config overrides and
# evaluation protocol.  Overrides are applied on top of the base config
# loaded from --config.
#
# drug_eval controls how the drug dataset is evaluated:
#   "all"  -- full drug dataset (OOD, drug data NOT in training)
#   "test" -- drug test split only (drug train/val were used in training)
EXPERIMENTS: Dict[str, Dict] = {
    "no_s1": {
        "label": "No S1",
        "description": "Train: binary_train | Eval: binary_test, drug_all (OOD)",
        "drug_eval": "all",
        "overrides": {
            "stage": {"checkpoint_path": None},
        },
    },
    "plus_drug": {
        "label": "+Drug",
        "description": "Train: binary_train+drug_train | Eval: binary_test, drug_test",
        "drug_eval": "test",
        "overrides": {
            "data": {"include_drug_data": True},
        },
    },
    "plus_synthetic": {
        "label": "+Synthetic",
        "description": "Train: binary_train + 6555 synthetic drug+drug pairs | Eval: binary_test, drug_all (OOD)",
        "drug_eval": "all",
        "overrides": {
            "augmentation": {
                "enable": True,
                "source": "stratified",
                "num_drug_train_samples": 0,
                "include_s1_drugs": True,
            },
        },
    },
}

ALL_EXPERIMENT_NAMES = list(EXPERIMENTS.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation experiments for Stage 2 unmixing (multi-seed)")
    p.add_argument("--config", type=str, default="configs/stage2_binary.json",
                   help="Base Stage 2 config JSON (default: configs/stage2_binary.json)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Model initialisation seeds (default: 0 1 2 3 4)")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Base results directory (default: results/ablation)")
    p.add_argument("--experiments", type=str, nargs="+",
                   default=ALL_EXPERIMENT_NAMES,
                   choices=ALL_EXPERIMENT_NAMES,
                   help="Which experiments to run (default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------

def _apply_overrides(base: Dict, overrides: Dict) -> Dict:
    """Deep-merge *overrides* into a deep copy of *base*."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            for sub_key, sub_value in value.items():
                merged[key][sub_key] = sub_value
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_experiment(
    config_data: Dict,
    checkpoint_dir: Path,
    test_idx: List[int],
    drug_eval: str,
    no_matching: bool,
) -> Dict:
    """Evaluate on binary test set and drug set."""
    results: Dict = {}

    # Find best or last checkpoint
    ckpt = checkpoint_dir / "checkpoints" / "stage2_best.pt"
    if not ckpt.exists():
        ckpt = checkpoint_dir / "checkpoints" / "stage2_last.pt"
    if not ckpt.exists():
        print(f"[Warning] No checkpoint found in {checkpoint_dir}, skipping eval")
        return results

    # Binary test set (always the same held-out split)
    print(f"[Eval] Binary test set ({len(test_idx)} samples) ...")
    test_results = evaluate_checkpoint(
        deepcopy(config_data),
        str(ckpt),
        test_idx,
        no_matching=no_matching,
        no_refinement=True,
    )
    results["test"] = test_results

    # Drug set: split depends on whether drug data was used in training
    cfg_tmp = load_config(**deepcopy(config_data))
    drug_file = cfg_tmp.data.drug_mixture_dataset
    drug_dataset = MixtureDataset(cfg_tmp.data, dataset_file=drug_file)
    n_drug = len(drug_dataset)
    del drug_dataset

    if drug_eval == "test":
        # Drug data was in training; evaluate on the held-out drug test split
        _, _, drug_idx = split_indices(n_drug, seed=cfg_tmp.seed)
        split_label = "drug_test"
    else:
        # Drug data is fully OOD; evaluate on the entire drug dataset
        drug_idx = list(range(n_drug))
        split_label = "drug_all"
    del cfg_tmp

    print(f"[Eval] Drug {split_label} ({len(drug_idx)} samples) ...")
    drug_results = evaluate_checkpoint(
        deepcopy(config_data),
        str(ckpt),
        drug_idx,
        dataset_file=drug_file,
        no_matching=no_matching,
        no_refinement=True,
    )
    drug_results["split"] = split_label
    results["drug"] = drug_results

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _extract_scalar(results: Dict, dataset_key: str, section: str, metric: str) -> Optional[float]:
    """Extract a scalar metric from nested results."""
    ds = results.get(dataset_key, {})
    sec = ds.get(section, {})
    val = sec.get(metric)
    if isinstance(val, dict) and "mean" in val:
        return val["mean"]
    if isinstance(val, (int, float)):
        return float(val)
    return None


# Metrics we track for aggregation and display
_TRACKED_METRICS = [
    ("test", "unmixing", "component_cosine", "Bin Comp Cos"),
    ("test", "unmixing", "component_mse", "Bin Comp MSE"),
    ("test", "matching", "top1_accuracy", "Bin Match R@1"),
    ("test", "matching", "top5_accuracy", "Bin Match R@5"),
    ("test", "matching", "mixture_match_rate", "Bin Mix Match"),
    ("drug", "unmixing", "component_cosine", "Drug Comp Cos"),
    ("drug", "unmixing", "component_mse", "Drug Comp MSE"),
    ("drug", "matching", "top1_accuracy", "Drug Match R@1"),
    ("drug", "matching", "top5_accuracy", "Drug Match R@5"),
    ("drug", "matching", "mixture_match_rate", "Drug Mix Match"),
]


def aggregate_seeds(
    per_seed_results: Dict[int, Dict],
    seeds: List[int],
) -> Dict:
    """Aggregate per-seed results into mean/std for each tracked metric."""
    agg: Dict = {}
    for ds_key, section, metric, label in _TRACKED_METRICS:
        vals = []
        per_seed = {}
        for s in seeds:
            v = _extract_scalar(per_seed_results.get(s, {}), ds_key, section, metric)
            if v is not None:
                vals.append(v)
                per_seed[str(s)] = v
        if vals:
            agg[label] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
                "per_seed": per_seed,
            }
    return agg


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_experiment_summary(
    exp_name: str,
    exp_label: str,
    aggregated: Dict,
    seeds: List[int],
) -> None:
    """Print a formatted table for one experiment across seeds."""
    print(f"\n  {exp_name} ({exp_label})")

    seed_cols = "".join(f"{'s=' + str(s):>10}" for s in seeds)
    print(f"  {'Metric':<20s} {seed_cols} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-' * (20 + len(seeds) * 10 + 20)}")

    for _, _, _, label in _TRACKED_METRICS:
        if label not in aggregated:
            continue
        a = aggregated[label]
        val_strs = []
        for s in seeds:
            v = a["per_seed"].get(str(s))
            val_strs.append(f"{v:>10.4f}" if v is not None else f"{'--':>10s}")
        print(f"  {label:<20s} {''.join(val_strs)} {a['mean']:>10.4f} {a['std']:>10.4f}")


def print_comparison_table(
    all_aggregated: Dict[str, Dict],
    experiment_names: List[str],
) -> None:
    """Print a compact comparison table: mean +/- std for each experiment."""
    labels = {name: EXPERIMENTS[name]["label"] for name in experiment_names}

    col_width = 20
    name_width = 20
    exp_cols = "".join(f"{labels[n]:>{col_width}}" for n in experiment_names)
    total_width = name_width + len(experiment_names) * col_width

    print(f"\n{'=' * total_width}")
    print(f"ABLATION COMPARISON (mean +/- std over seeds)")
    print(f"{'=' * total_width}")
    print(f"{'Metric':<{name_width}}{exp_cols}")
    print(f"{'-' * total_width}")

    for _, _, _, label in _TRACKED_METRICS:
        any_present = any(label in all_aggregated.get(n, {}) for n in experiment_names)
        if not any_present:
            continue
        val_strs = []
        for n in experiment_names:
            a = all_aggregated.get(n, {}).get(label)
            if a is None:
                val_strs.append(f"{'--':>{col_width}}")
            else:
                s = f"{a['mean']:.4f}+/-{a['std']:.4f}"
                val_strs.append(f"{s:>{col_width}}")
        print(f"{label:<{name_width}}{''.join(val_strs)}")

    print(f"{'=' * total_width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results_base = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    results_base.mkdir(parents=True, exist_ok=True)

    # Load base config
    config_path = Path(args.config)
    with open(config_path) as f:
        base_config = json.load(f)

    # Compute the fixed data split (seed=42)
    tmp_cfg = load_config(**deepcopy(base_config))
    base_dataset = MixtureDataset(tmp_cfg.data,
                                  dataset_file=tmp_cfg.data.mixture_dataset)
    num_samples = len(base_dataset)
    train_idx, val_idx, test_idx = split_indices(num_samples,
                                                 seed=DATA_SPLIT_SEED)
    del base_dataset, tmp_cfg

    print(f"\n{'=' * 70}")
    print("ABLATION EXPERIMENTS (MULTI-SEED)")
    print(f"{'=' * 70}")
    print(f"  Config:          {config_path}")
    print(f"  Data split seed: {DATA_SPLIT_SEED}")
    print(f"  Split sizes:     train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_idx)}")
    print(f"  Model seeds:     {args.seeds}")
    print(f"  Experiments:     {args.experiments}")
    print(f"  Results dir:     {results_base}")
    print(f"{'=' * 70}\n")

    # Per-experiment, per-seed training + evaluation
    all_aggregated: Dict[str, Dict] = {}
    all_per_seed: Dict[str, Dict[int, Dict]] = {}

    for exp_name in args.experiments:
        exp_def = EXPERIMENTS[exp_name]
        exp_dir = results_base / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT: {exp_name} ({exp_def['label']})")
        print(f"  {exp_def['description']}")
        print(f"{'=' * 70}")

        # Build experiment config (apply experiment-specific overrides)
        exp_base_config = _apply_overrides(base_config, exp_def["overrides"])
        exp_base_config.setdefault("stage", {})

        per_seed_results: Dict[int, Dict] = {}

        for seed in args.seeds:
            seed_dir = exp_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- {exp_name} / seed={seed} ---")

            # Build per-seed config
            seed_config = deepcopy(exp_base_config)
            seed_config["seed"] = seed
            seed_config["results_dir"] = str(seed_dir)
            seed_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

            # Train
            print(f"[Train] {exp_name} seed={seed} ...")
            train_stage(deepcopy(seed_config),
                        split_override=(train_idx, val_idx, test_idx))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Evaluate
            print(f"[Eval] {exp_name} seed={seed} ...")
            seed_results = _evaluate_experiment(
                seed_config,
                seed_dir,
                test_idx,
                drug_eval=exp_def["drug_eval"],
                no_matching=False,
            )

            if seed_results:
                seed_results["seed"] = seed
                per_seed_results[seed] = seed_results

                # Save per-seed results
                if "test" in seed_results:
                    with open(seed_dir / "test_results.json", "w") as f:
                        json.dump(seed_results["test"], f, indent=2)
                if "drug" in seed_results:
                    with open(seed_dir / "drug_results.json", "w") as f:
                        json.dump(seed_results["drug"], f, indent=2)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate this experiment across seeds
        exp_aggregated = aggregate_seeds(per_seed_results, args.seeds)
        all_aggregated[exp_name] = exp_aggregated
        all_per_seed[exp_name] = per_seed_results

        # Save per-experiment aggregation
        exp_summary = {
            "experiment": exp_name,
            "label": exp_def["label"],
            "description": exp_def["description"],
            "drug_eval": exp_def["drug_eval"],
            "data_split_seed": DATA_SPLIT_SEED,
            "model_seeds": args.seeds,
            "aggregated": exp_aggregated,
            "per_seed_results": {str(k): v for k, v in per_seed_results.items()},
        }
        with open(exp_dir / "aggregated.json", "w") as f:
            json.dump(exp_summary, f, indent=2)

        # Print per-experiment table
        if exp_aggregated:
            print_experiment_summary(exp_name, exp_def["label"],
                                    exp_aggregated, args.seeds)

    # Save global aggregation
    summary = {
        "protocol": "ablation_multi_seed",
        "data_split_seed": DATA_SPLIT_SEED,
        "model_seeds": args.seeds,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "experiments": args.experiments,
        "per_experiment_aggregated": all_aggregated,
        "per_experiment_per_seed": {
            exp: {str(s): r for s, r in seeds.items()}
            for exp, seeds in all_per_seed.items()
        },
    }
    out_path = results_base / "aggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    completed = [n for n in args.experiments if n in all_aggregated and all_aggregated[n]]
    if completed:
        print_comparison_table(all_aggregated, completed)

    print(f"\n[Done] Aggregated results saved to {out_path}")


if __name__ == "__main__":
    main()
