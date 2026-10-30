#!/usr/bin/env python
"""Stage 1 contribution controls: autoencoder and shuffled-pair baselines (multi-seed).

Usage (from slot_ir/):
    python run_stage1_controls.py
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
DEFAULT_RESULTS_DIR = SLOT_UNMIXER_ROOT / "results/stage1_controls"

# Variant definitions
VARIANTS: Dict[str, Dict] = {
    "baseline": {
        "label": "Baseline (gas->liq)",
        "stage1_mode": "translation",
    },
    "autoencoder": {
        "label": "Autoencoder (liq->liq)",
        "stage1_mode": "autoencoder",
    },
    "gas_autoencoder": {
        "label": "Gas Autoencoder (gas->gas)",
        "stage1_mode": "gas_autoencoder",
    },
    "shuffled": {
        "label": "Shuffled pairs",
        "stage1_mode": "shuffled",
    },
}

ALL_VARIANT_NAMES = list(VARIANTS.keys())
DEFAULT_VARIANT_NAMES = ["baseline", "autoencoder", "gas_autoencoder", "shuffled"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 contribution controls: autoencoder and shuffled-pair baselines (multi-seed)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Model initialisation seeds (default: 0 1 2 3 4)")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Base results directory (default: results/stage1_controls)")
    p.add_argument("--variants", type=str, nargs="+",
                   default=DEFAULT_VARIANT_NAMES,
                   choices=ALL_VARIANT_NAMES,
                   help="Which variants to run (default: all four Stage 1 objectives)")
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

def _evaluate_seed(
    config_data: Dict,
    checkpoint_dir: Path,
    test_idx: List[int],
    no_matching: bool,
) -> Dict:
    """Evaluate on binary test set and drug set (OOD)."""
    results: Dict = {}

    ckpt = checkpoint_dir / "checkpoints" / "stage2_best.pt"
    if not ckpt.exists():
        ckpt = checkpoint_dir / "checkpoints" / "stage2_last.pt"
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

    # Drug OOD (all drug molecules)
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

def print_variant_summary(
    variant_name: str,
    variant_label: str,
    aggregated: Dict,
    seeds: List[int],
) -> None:
    """Print a formatted table for one variant across seeds."""
    print(f"\n  {variant_name} ({variant_label})")

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
    variant_names: List[str],
) -> None:
    """Print a compact comparison table: mean +/- std for each variant."""
    labels = {name: VARIANTS[name]["label"] for name in variant_names}

    col_width = 24
    name_width = 20
    exp_cols = "".join(f"{labels[n]:>{col_width}}" for n in variant_names)
    total_width = name_width + len(variant_names) * col_width

    print(f"\n{'=' * total_width}")
    print(f"EXP D: STAGE 1 CONTRIBUTION (mean +/- std over seeds)")
    print(f"{'=' * total_width}")
    print(f"{'Metric':<{name_width}}{exp_cols}")
    print(f"{'-' * total_width}")

    for _, _, _, label in _TRACKED_METRICS:
        any_present = any(label in all_aggregated.get(n, {}) for n in variant_names)
        if not any_present:
            continue
        val_strs = []
        for n in variant_names:
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

    # Load configs
    s1_config_path = Path("configs/stage1.json")
    s2_config_path = Path("configs/stage2_binary.json")

    with open(s1_config_path) as f:
        s1_base_config = json.load(f)
    with open(s2_config_path) as f:
        s2_base_config = json.load(f)

    # Compute fixed data split for Stage 2 binary
    tmp_cfg = load_config(**deepcopy(s2_base_config))
    base_dataset = MixtureDataset(tmp_cfg.data, dataset_file=tmp_cfg.data.mixture_dataset)
    num_samples = len(base_dataset)
    train_idx, val_idx, test_idx = split_indices(num_samples, seed=DATA_SPLIT_SEED)
    del base_dataset, tmp_cfg

    print(f"\n{'=' * 70}")
    print("EXPERIMENT D: STAGE 1 CONTRIBUTION MECHANISM")
    print(f"{'=' * 70}")
    print(f"  Stage 1 config:  {s1_config_path}")
    print(f"  Stage 2 config:  {s2_config_path}")
    print(f"  Data split seed: {DATA_SPLIT_SEED}")
    print(f"  Split sizes:     train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_idx)}")
    print(f"  Model seeds:     {args.seeds}")
    print(f"  Variants:        {args.variants}")
    print(f"  Results dir:     {results_base}")
    print(f"{'=' * 70}\n")

    # Per-variant, per-seed training + evaluation
    all_aggregated: Dict[str, Dict] = {}
    all_per_seed: Dict[str, Dict[int, Dict]] = {}

    for variant_name in args.variants:
        variant_def = VARIANTS[variant_name]
        variant_dir = results_base / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  VARIANT: {variant_name} ({variant_def['label']})")
        print(f"  stage1_mode: {variant_def['stage1_mode']}")
        print(f"{'=' * 70}")

        per_seed_results: Dict[int, Dict] = {}

        for seed in args.seeds:
            seed_dir = variant_dir / f"seed_{seed}"
            s1_dir = seed_dir / "stage1"
            s2_dir = seed_dir / "stage2"
            s1_dir.mkdir(parents=True, exist_ok=True)
            s2_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- {variant_name} / seed={seed} ---")

            s1_ckpt_path = s1_dir / "checkpoints" / "stage1_best.pt"

            # ==== Stage 1 ====
            print(f"[Train] Stage 1 ({variant_def['stage1_mode']}) seed={seed} ...")
            s1_config = deepcopy(s1_base_config)
            s1_config["seed"] = seed
            s1_config["results_dir"] = str(s1_dir)
            s1_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            s1_config.setdefault("stage", {})["stage1_mode"] = variant_def["stage1_mode"]

            train_stage(s1_config)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Resolve Stage 1 checkpoint
            if s1_ckpt_path.exists():
                stage1_checkpoint = str(s1_ckpt_path)
            else:
                # Fall back to last checkpoint
                s1_last = s1_dir / "checkpoints" / "stage1_last.pt"
                if s1_last.exists():
                    stage1_checkpoint = str(s1_last)
                else:
                    stage1_checkpoint = s2_base_config.get("stage", {}).get("checkpoint_path")
                    if stage1_checkpoint:
                        print(f"[Warning] Using fallback checkpoint: {stage1_checkpoint}")
                    else:
                        print(f"[Warning] No S1 checkpoint for {variant_name}/seed_{seed}")
                        stage1_checkpoint = None

            # ==== Stage 2 ====
            print(f"[Train] Stage 2 (binary) seed={seed} ...")
            s2_config = deepcopy(s2_base_config)
            s2_config["seed"] = seed
            s2_config["results_dir"] = str(s2_dir)
            s2_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            if stage1_checkpoint:
                s2_config.setdefault("stage", {})["checkpoint_path"] = stage1_checkpoint

            train_stage(deepcopy(s2_config),
                        split_override=(train_idx, val_idx, test_idx))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ==== Evaluate ====
            print(f"[Eval] {variant_name} seed={seed} ...")
            eval_config = deepcopy(s2_base_config)
            eval_config["seed"] = seed
            eval_config["results_dir"] = str(s2_dir)
            eval_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            if stage1_checkpoint:
                eval_config.setdefault("stage", {})["checkpoint_path"] = stage1_checkpoint

            seed_results = _evaluate_seed(
                eval_config,
                s2_dir,
                test_idx,
                no_matching=False,
            )

            if seed_results:
                seed_results["seed"] = seed
                seed_results["variant"] = variant_name
                seed_results["stage1_mode"] = variant_def["stage1_mode"]
                per_seed_results[seed] = seed_results

                if "test" in seed_results:
                    with open(seed_dir / "test_results.json", "w") as f:
                        json.dump(seed_results["test"], f, indent=2)
                if "drug" in seed_results:
                    with open(seed_dir / "drug_results.json", "w") as f:
                        json.dump(seed_results["drug"], f, indent=2)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate this variant across seeds
        variant_aggregated = aggregate_seeds(per_seed_results, args.seeds)
        all_aggregated[variant_name] = variant_aggregated
        all_per_seed[variant_name] = per_seed_results

        # Save per-variant aggregation
        variant_summary = {
            "variant": variant_name,
            "label": variant_def["label"],
            "stage1_mode": variant_def["stage1_mode"],
            "data_split_seed": DATA_SPLIT_SEED,
            "model_seeds": args.seeds,
            "aggregated": variant_aggregated,
            "per_seed_results": {str(k): v for k, v in per_seed_results.items()},
        }
        with open(variant_dir / "aggregated.json", "w") as f:
            json.dump(variant_summary, f, indent=2)

        if variant_aggregated:
            print_variant_summary(variant_name, variant_def["label"],
                                  variant_aggregated, args.seeds)

    # Save global aggregation
    summary = {
        "protocol": "exp_d_stage1_contribution",
        "data_split_seed": DATA_SPLIT_SEED,
        "model_seeds": args.seeds,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "variants": args.variants,
        "per_variant_aggregated": all_aggregated,
        "per_variant_per_seed": {
            variant: {str(s): r for s, r in seeds.items()}
            for variant, seeds in all_per_seed.items()
        },
    }
    out_path = results_base / "aggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    completed = [n for n in args.variants if n in all_aggregated and all_aggregated[n]]
    if completed:
        print_comparison_table(all_aggregated, completed)

    print(f"\n[Done] Aggregated results saved to {out_path}")


if __name__ == "__main__":
    main()
