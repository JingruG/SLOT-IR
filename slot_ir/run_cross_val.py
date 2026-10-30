#!/usr/bin/env python
"""K-Fold Cross-Validation experiment for slot_ir Stage 2.

Usage (from slot_ir/):
    python run_cross_val.py --config configs/stage2_binary.json --results-dir results/cross_val_binary
    python run_cross_val.py --config configs/stage2_ternary.json --results-dir results/cross_val_ternary
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from slot_ir package
# ---------------------------------------------------------------------------
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from src.config import load_config, SLOT_UNMIXER_ROOT
from src.data import MixtureDataset, kfold_split_indices
from src.training import train_stage
from src.evaluate import evaluate_checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_K = 5
DEFAULT_SEED = 42
DEFAULT_RESULTS_DIR = SLOT_UNMIXER_ROOT / "results/cross_val"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K-Fold cross-validation experiment (Stage 2)")
    p.add_argument("--config", type=str, required=True,
                   help="Stage 2 config JSON (e.g. configs/stage2_binary.json)")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Base results directory (default: results/cross_val)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fold-level split construction
# ---------------------------------------------------------------------------

def build_fold_split(
    folds: List[List[int]],
    test_fold_idx: int,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Given k folds, build train/val/test for a specific fold."""
    test_idx = list(folds[test_fold_idx])

    remaining: List[int] = []
    for i, fold in enumerate(folds):
        if i != test_fold_idx:
            remaining.extend(fold)

    # Shuffle the remaining indices deterministically and split
    rng = random.Random(seed)
    rng.shuffle(remaining)
    n_val = max(1, int(len(remaining) * val_ratio))
    val_idx = remaining[:n_val]
    train_idx = remaining[n_val:]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _collect_scalar(results: Dict, prefix: str, section: str
                    ) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    sec = results.get(section, {})
    for k, v in sec.items():
        if isinstance(v, dict) and "mean" in v:
            val = v["mean"]
        elif isinstance(v, (int, float)):
            val = float(v)
        else:
            continue
        key = f"{prefix}{k}"
        out.setdefault(key, []).append(val)
    return out


def aggregate_results(
    all_results: Dict[int, Dict],
    k: int,
) -> Dict:
    """Aggregate per-fold results into mean/std summaries."""
    collected: Dict[str, List[float]] = {}
    for fold_idx in range(k):
        r = all_results.get(fold_idx, {})
        for prefix, section in [
            ("unmixing/", "unmixing"),
            ("matching/", "matching"),
            ("refinement/", "refinement_matching"),
        ]:
            for key, vals in _collect_scalar(r, prefix, section).items():
                collected.setdefault(key, []).extend(vals)

    agg: Dict = {}
    for key, vals in collected.items():
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_fold": {str(i): v for i, v in enumerate(vals)},
        }
    return agg


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_summary(
    all_results: Dict[int, Dict],
    aggregated: Dict,
    k: int,
) -> None:
    print("\n" + "=" * 110)
    print(f"{k}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 110)

    display = [
        ("unmixing/component_cosine", "Comp Cosine"),
        ("unmixing/component_mse", "Comp MSE"),
        ("matching/top1_accuracy", "Match R@1"),
        ("matching/top5_accuracy", "Match R@5"),
        ("matching/mixture_match_rate", "Mix Match"),
        ("refinement/component_top1_accuracy", "Ref cR@1"),
        ("refinement/pair_top1_accuracy", "Ref pR@1"),
    ]

    fold_cols = "".join(f"{'f' + str(i):>10}" for i in range(k))
    print(f"{'Metric':<25} {fold_cols} {'Mean':>10} {'Std':>10}")
    print("-" * 110)

    for key, label in display:
        if key not in aggregated:
            continue
        a = aggregated[key]
        vals = [a["per_fold"].get(str(i), float("nan")) for i in range(k)]
        val_str = "".join(f"{v:>10.4f}" for v in vals)
        print(f"{label:<25} {val_str} {a['mean']:>10.4f} {a['std']:>10.4f}")

    print("=" * 110)

    # Per-fold test set sizes
    sizes = [all_results.get(i, {}).get("n_test", "?") for i in range(k)]
    print(f"  Test set sizes: {sizes}")
    total = sum(s for s in sizes if isinstance(s, int))
    print(f"  Total samples evaluated: {total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    k = DEFAULT_K
    seed = DEFAULT_SEED
    results_base = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    results_base.mkdir(parents=True, exist_ok=True)

    # Load base config --------------------------------------------------------
    config_path = Path(args.config)
    with open(config_path) as f:
        base_config = json.load(f)

    # Compute k-fold partition ------------------------------------------------
    tmp_cfg = load_config(**deepcopy(base_config))
    base_dataset = MixtureDataset(tmp_cfg.data,
                                  dataset_file=tmp_cfg.data.mixture_dataset)
    num_samples = len(base_dataset)
    folds = kfold_split_indices(num_samples, k=k, seed=seed)
    del base_dataset, tmp_cfg

    print(f"\n{'=' * 70}")
    print(f"{k}-FOLD CROSS-VALIDATION EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"  Config:      {config_path}")
    print(f"  k:           {k}")
    print(f"  Seed:        {seed}")
    print(f"  Total data:  {num_samples}")
    print(f"  Fold sizes:  {[len(f) for f in folds]}")
    print(f"  Results dir: {results_base}")
    print(f"{'=' * 70}\n")

    # Resolve Stage 1 checkpoint path
    s1_ckpt = base_config.get("stage", {}).get("checkpoint_path")

    # Per-fold training + evaluation ------------------------------------------
    all_results: Dict[int, Dict] = {}

    for fold_idx in range(k):
        fold_dir = results_base / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_idx, val_idx, test_idx = build_fold_split(
            folds, fold_idx, val_ratio=0.1, seed=seed)

        print(f"\n{'=' * 70}")
        print(f"  FOLD {fold_idx}/{k}   "
              f"train={len(train_idx)}  val={len(val_idx)}  "
              f"test={len(test_idx)}")
        print(f"{'=' * 70}")

        # Build per-fold config overrides
        overrides = deepcopy(base_config)
        overrides["seed"] = seed
        overrides["results_dir"] = str(fold_dir)
        overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if s1_ckpt:
            overrides.setdefault("stage", {})["checkpoint_path"] = s1_ckpt

        # Train ---------------------------------------------------------------
        train_stage(overrides,
                    split_override=(train_idx, val_idx, test_idx))

        # Free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate on this fold's test set ------------------------------------
        ckpt = fold_dir / "checkpoints" / "stage2_best.pt"
        if not ckpt.exists():
            ckpt = fold_dir / "checkpoints" / "stage2_last.pt"

        if ckpt.exists():
            print(f"\n[Eval] Evaluating fold {fold_idx} "
                  f"({len(test_idx)} test samples) ...")
            results = evaluate_checkpoint(
                deepcopy(base_config),
                str(ckpt),
                test_idx,
                no_matching=False,
                no_refinement=True,
            )
            results["fold"] = fold_idx
            results["n_test"] = len(test_idx)
            results["n_train"] = len(train_idx)
            results["n_val"] = len(val_idx)
            all_results[fold_idx] = results

            with open(fold_dir / "test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"[Eval] Saved test_results.json for fold {fold_idx}")
        else:
            print(f"[Warning] No checkpoint for fold {fold_idx}, skipping eval")

        # Free GPU memory again after evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate ---------------------------------------------------------------
    aggregated = aggregate_results(all_results, k)

    summary = {
        "protocol": "k_fold_cross_validation",
        "k": k,
        "seed": seed,
        "n_total": num_samples,
        "fold_sizes": [len(f) for f in folds],
        "per_fold_results": {str(i): v for i, v in all_results.items()},
        "aggregated": aggregated,
    }
    out_path = results_base / "aggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(all_results, aggregated, k)
    print(f"\n[Done] Aggregated results saved to {out_path}")


if __name__ == "__main__":
    main()
