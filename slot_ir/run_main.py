#!/usr/bin/env python
"""Main experiment pipeline: Stage 1 (once) + Stage 2 binary/ternary (multi-seed).

Usage (from slot_ir/):
    python run_main.py
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
DEFAULT_RESULTS_DIR = SLOT_UNMIXER_ROOT / "results/main"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full pipeline: Stage 1 + Stage 2 binary/ternary (multi-seed)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Model initialisation seeds (default: 0 1 2 3 4)")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Base results directory (default: results/main)")
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

def _evaluate_binary(
    config_data: Dict,
    checkpoint_dir: Path,
    test_idx: List[int],
    no_matching: bool,
) -> Dict:
    """Evaluate binary model on binary_test + drug_all (OOD)."""
    results: Dict = {}

    ckpt = checkpoint_dir / "checkpoints" / "stage2_best.pt"
    if not ckpt.exists():
        print(f"[Warning] No checkpoint found in {checkpoint_dir}, skipping eval")
        return results

    # Binary test set
    print(f"[Eval] Binary test set ({len(test_idx)} samples) ...")
    test_results = evaluate_checkpoint(
        deepcopy(config_data),
        str(ckpt),
        test_idx,
        no_matching=no_matching,
        no_refinement=True,
    )
    results["test"] = test_results

    # Drug OOD (all 115 drug molecules, binary pairs)
    cfg_tmp = load_config(**deepcopy(config_data))
    drug_file = cfg_tmp.data.drug_mixture_dataset
    drug_dataset = MixtureDataset(cfg_tmp.data, dataset_file=drug_file)
    n_drug = len(drug_dataset)
    del drug_dataset, cfg_tmp

    drug_idx = list(range(n_drug))
    print(f"[Eval] Drug OOD ({len(drug_idx)} samples) ...")
    drug_results = evaluate_checkpoint(
        deepcopy(config_data),
        str(ckpt),
        drug_idx,
        dataset_file=drug_file,
        no_matching=no_matching,
        no_refinement=True,
    )
    drug_results["split"] = "drug_all"
    results["drug"] = drug_results

    return results


def _evaluate_ternary(
    config_data: Dict,
    checkpoint_dir: Path,
    test_idx: List[int],
    no_matching: bool,
) -> Dict:
    """Evaluate ternary model on ternary_test only."""
    results: Dict = {}

    ckpt = checkpoint_dir / "checkpoints" / "stage2_best.pt"
    if not ckpt.exists():
        print(f"[Warning] No checkpoint found in {checkpoint_dir}, skipping eval")
        return results

    # Ternary test set
    print(f"[Eval] Ternary test set ({len(test_idx)} samples) ...")
    test_results = evaluate_checkpoint(
        deepcopy(config_data),
        str(ckpt),
        test_idx,
        no_matching=no_matching,
        no_refinement=True,
    )
    results["test"] = test_results

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


# Metrics tracked for aggregation and display
_TRACKED_METRICS = [
    ("test", "unmixing", "component_cosine", "Test Comp Cos"),
    ("test", "unmixing", "component_mse", "Test Comp MSE"),
    ("test", "matching", "top1_accuracy", "Test Match R@1"),
    ("test", "matching", "top5_accuracy", "Test Match R@5"),
    ("test", "matching", "mixture_match_rate", "Test Mix Match"),
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

def print_stage_summary(
    stage_name: str,
    aggregated: Dict,
    seeds: List[int],
) -> None:
    """Print a formatted table for one stage across seeds."""
    print(f"\n  {stage_name}")

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
    binary_agg: Dict,
    ternary_agg: Dict,
) -> None:
    """Print a compact comparison of binary vs ternary."""
    col_width = 24
    name_width = 20
    total_width = name_width + 2 * col_width

    print(f"\n{'=' * total_width}")
    print("PIPELINE COMPARISON: Binary vs Ternary (mean +/- std)")
    print(f"{'=' * total_width}")
    print(f"{'Metric':<{name_width}}{'Binary':>{col_width}}{'Ternary':>{col_width}}")
    print(f"{'-' * total_width}")

    for _, _, _, label in _TRACKED_METRICS:
        b = binary_agg.get(label)
        t = ternary_agg.get(label)
        if b is None and t is None:
            continue
        b_str = f"{b['mean']:.4f}+/-{b['std']:.4f}" if b else "--"
        t_str = f"{t['mean']:.4f}+/-{t['std']:.4f}" if t else "--"
        print(f"{label:<{name_width}}{b_str:>{col_width}}{t_str:>{col_width}}")

    print(f"{'=' * total_width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results_base = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    results_base.mkdir(parents=True, exist_ok=True)

    # Load configs
    s1_config_path = Path("configs/stage1.json")
    s2_binary_config_path = Path("configs/stage2_binary.json")
    s2_ternary_config_path = Path("configs/stage2_ternary.json")

    with open(s1_config_path) as f:
        s1_config = json.load(f)
    with open(s2_binary_config_path) as f:
        s2_binary_config = json.load(f)
    with open(s2_ternary_config_path) as f:
        s2_ternary_config = json.load(f)

    # Stage 1 output directory and checkpoint path
    s1_dir = results_base / "stage1"
    s1_ckpt_path = s1_dir / "checkpoints" / "stage1_best.pt"

    # Compute fixed data splits for binary and ternary datasets
    tmp_cfg_bin = load_config(**deepcopy(s2_binary_config))
    binary_dataset = MixtureDataset(tmp_cfg_bin.data,
                                    dataset_file=tmp_cfg_bin.data.mixture_dataset)
    n_binary = len(binary_dataset)
    bin_train_idx, bin_val_idx, bin_test_idx = split_indices(n_binary, seed=DATA_SPLIT_SEED)
    del binary_dataset, tmp_cfg_bin

    tmp_cfg_ter = load_config(**deepcopy(s2_ternary_config))
    ternary_dataset = MixtureDataset(tmp_cfg_ter.data,
                                     dataset_file=tmp_cfg_ter.data.mixture_dataset)
    n_ternary = len(ternary_dataset)
    ter_train_idx, ter_val_idx, ter_test_idx = split_indices(n_ternary, seed=DATA_SPLIT_SEED)
    del ternary_dataset, tmp_cfg_ter

    print(f"\n{'=' * 70}")
    print("MAIN EXPERIMENT PIPELINE")
    print(f"{'=' * 70}")
    print(f"  Stage 1 config:       {s1_config_path}")
    print(f"  Stage 2 binary:       {s2_binary_config_path}")
    print(f"  Stage 2 ternary:      {s2_ternary_config_path}")
    print(f"  Data split seed:      {DATA_SPLIT_SEED}")
    print(f"  Binary split:         train={len(bin_train_idx)}  val={len(bin_val_idx)}  "
          f"test={len(bin_test_idx)}")
    print(f"  Ternary split:        train={len(ter_train_idx)}  val={len(ter_val_idx)}  "
          f"test={len(ter_test_idx)}")
    print(f"  Model seeds:          {args.seeds}")
    print(f"  Results dir:          {results_base}")
    print(f"{'=' * 70}\n")

    # =========================================================================
    # Stage 1: Train ONCE (shared encoder)
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  STAGE 1: Spectral Translation (trained once)")
    print(f"{'=' * 70}")

    s1_overrides = deepcopy(s1_config)
    s1_overrides["seed"] = DATA_SPLIT_SEED
    s1_overrides["results_dir"] = str(s1_dir)
    s1_overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    train_stage(s1_overrides)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Stage 1] Checkpoint saved to {s1_ckpt_path}")

    # Resolve Stage 1 checkpoint: use our trained one, or fall back to config default
    if s1_ckpt_path.exists():
        stage1_checkpoint = str(s1_ckpt_path)
    else:
        stage1_checkpoint = s2_binary_config.get("stage", {}).get("checkpoint_path")
        if stage1_checkpoint:
            print(f"[Stage 1] Using existing checkpoint: {stage1_checkpoint}")
        else:
            print("[Warning] No Stage 1 checkpoint found. Stage 2 will train from scratch.")

    # =========================================================================
    # Stage 2 Binary: Multi-seed training + evaluation
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  STAGE 2 BINARY: Multi-seed")
    print(f"{'=' * 70}")

    binary_per_seed: Dict[int, Dict] = {}

    for seed in args.seeds:
        seed_dir = results_base / "binary" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- binary / seed={seed} ---")

        # Build per-seed config
        seed_config = deepcopy(s2_binary_config)
        seed_config["seed"] = seed
        seed_config["results_dir"] = str(seed_dir)
        seed_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if stage1_checkpoint:
            seed_config.setdefault("stage", {})["checkpoint_path"] = stage1_checkpoint

        # Train
        print(f"[Train] binary seed={seed} ...")
        train_stage(deepcopy(seed_config),
                    split_override=(bin_train_idx, bin_val_idx, bin_test_idx))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate: binary_test + drug_all (OOD)
        print(f"[Eval] binary seed={seed} ...")
        seed_results = _evaluate_binary(
            seed_config,
            seed_dir,
            bin_test_idx,
            no_matching=False,
        )

        if seed_results:
            seed_results["seed"] = seed
            binary_per_seed[seed] = seed_results

            if "test" in seed_results:
                with open(seed_dir / "test_results.json", "w") as f:
                    json.dump(seed_results["test"], f, indent=2)
            if "drug" in seed_results:
                with open(seed_dir / "drug_results.json", "w") as f:
                    json.dump(seed_results["drug"], f, indent=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate binary
    binary_aggregated = aggregate_seeds(binary_per_seed, args.seeds)
    binary_summary = {
        "stage": "binary",
        "data_split_seed": DATA_SPLIT_SEED,
        "model_seeds": args.seeds,
        "n_train": len(bin_train_idx),
        "n_val": len(bin_val_idx),
        "n_test": len(bin_test_idx),
        "aggregated": binary_aggregated,
        "per_seed_results": {str(k): v for k, v in binary_per_seed.items()},
    }
    binary_dir = results_base / "binary"
    binary_dir.mkdir(parents=True, exist_ok=True)
    with open(binary_dir / "aggregated.json", "w") as f:
        json.dump(binary_summary, f, indent=2)

    if binary_aggregated:
        print_stage_summary("Stage 2 Binary", binary_aggregated, args.seeds)

    # =========================================================================
    # Stage 2 Ternary: Multi-seed training + evaluation
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  STAGE 2 TERNARY: Multi-seed")
    print(f"{'=' * 70}")

    ternary_per_seed: Dict[int, Dict] = {}

    for seed in args.seeds:
        seed_dir = results_base / "ternary" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- ternary / seed={seed} ---")

        # Build per-seed config
        seed_config = deepcopy(s2_ternary_config)
        seed_config["seed"] = seed
        seed_config["results_dir"] = str(seed_dir)
        seed_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if stage1_checkpoint:
            seed_config.setdefault("stage", {})["checkpoint_path"] = stage1_checkpoint

        # Train
        print(f"[Train] ternary seed={seed} ...")
        train_stage(deepcopy(seed_config),
                    split_override=(ter_train_idx, ter_val_idx, ter_test_idx))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate: ternary_test only
        print(f"[Eval] ternary seed={seed} ...")
        seed_results = _evaluate_ternary(
            seed_config,
            seed_dir,
            ter_test_idx,
            no_matching=False,
        )

        if seed_results:
            seed_results["seed"] = seed
            ternary_per_seed[seed] = seed_results

            if "test" in seed_results:
                with open(seed_dir / "test_results.json", "w") as f:
                    json.dump(seed_results["test"], f, indent=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate ternary
    ternary_aggregated = aggregate_seeds(ternary_per_seed, args.seeds)
    ternary_summary = {
        "stage": "ternary",
        "data_split_seed": DATA_SPLIT_SEED,
        "model_seeds": args.seeds,
        "n_train": len(ter_train_idx),
        "n_val": len(ter_val_idx),
        "n_test": len(ter_test_idx),
        "aggregated": ternary_aggregated,
        "per_seed_results": {str(k): v for k, v in ternary_per_seed.items()},
    }
    ternary_dir = results_base / "ternary"
    ternary_dir.mkdir(parents=True, exist_ok=True)
    with open(ternary_dir / "aggregated.json", "w") as f:
        json.dump(ternary_summary, f, indent=2)

    if ternary_aggregated:
        print_stage_summary("Stage 2 Ternary", ternary_aggregated, args.seeds)

    # =========================================================================
    # Global summary
    # =========================================================================
    summary = {
        "protocol": "main_pipeline",
        "data_split_seed": DATA_SPLIT_SEED,
        "model_seeds": args.seeds,
        "stage1_checkpoint": stage1_checkpoint,
        "binary": {
            "n_train": len(bin_train_idx),
            "n_val": len(bin_val_idx),
            "n_test": len(bin_test_idx),
            "aggregated": binary_aggregated,
        },
        "ternary": {
            "n_train": len(ter_train_idx),
            "n_val": len(ter_val_idx),
            "n_test": len(ter_test_idx),
            "aggregated": ternary_aggregated,
        },
    }
    out_path = results_base / "aggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    if binary_aggregated or ternary_aggregated:
        print_comparison_table(binary_aggregated, ternary_aggregated)

    print(f"\n[Done] Aggregated results saved to {out_path}")


if __name__ == "__main__":
    main()
