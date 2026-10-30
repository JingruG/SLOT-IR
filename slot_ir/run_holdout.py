#!/usr/bin/env python
"""Train Stage 2 with molecule-level holdout (multi-seed).

Usage (from slot_ir/):
    python run_holdout.py
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


_SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SELF_DIR))

from src.config import load_config, SLOT_UNMIXER_ROOT
from src.data import MixtureDataset, compute_molecule_holdout, split_indices
from src.training import train_stage

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
# The data split and the held-out molecules are fixed (seed 42), as in the evaluation
# (analysis.generalization_test --seed 42). Only the model seed changes per run.
DATA_SPLIT_SEED = 42
DEFAULT_CONFIG = "configs/generalization/stage2_binary_holdout.json"
DEFAULT_RESULTS_DIR = SLOT_UNMIXER_ROOT / "results" / "holdout"


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 2 with molecule holdout (multi-seed)")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                   help=f"Config JSON (default: {DEFAULT_CONFIG})")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--results-dir", type=str, default=None,
                   help=f"Base results dir (default: {DEFAULT_RESULTS_DIR})")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        base_config = json.load(f)

    print(f"Config: {args.config}")
    print(f"Results dir: {results_dir}")
    print(f"Seeds: {args.seeds}")
    print(f"n_holdout_molecules: {base_config['data'].get('n_holdout_molecules', 0)}")
    print()

    # ---- Fixed split and holdout (seed 42) ----
    cfg0 = load_config(**deepcopy(base_config))
    n_mix = len(MixtureDataset(cfg0.data, dataset_file=cfg0.data.mixture_dataset))
    train_idx, val_idx, test_idx = split_indices(n_mix, seed=DATA_SPLIT_SEED)
    _, holdout = compute_molecule_holdout(cfg0.data, train_idx, seed=DATA_SPLIT_SEED)
    with open(results_dir / "holdout_molecules.json", "w") as f:
        json.dump({"split_seed": DATA_SPLIT_SEED, "holdout_seed": DATA_SPLIT_SEED,
                   "holdout_molecules": sorted(int(m) for m in holdout)}, f, indent=2)
    print(f"Fixed split (seed {DATA_SPLIT_SEED}): {len(train_idx)}/{len(val_idx)}/{len(test_idx)}; "
          f"{len(holdout)} held-out molecules")

    # ---- Train each seed ----
    for seed in args.seeds:
        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  TRAINING SEED {seed}")
        print(f"{'='*60}\n")

        cfg_overrides = deepcopy(base_config)
        cfg_overrides["results_dir"] = str(seed_dir)
        cfg_overrides["seed"] = seed
        cfg_overrides.setdefault("data", {})["holdout_seed"] = DATA_SPLIT_SEED

        try:
            summary = train_stage(cfg_overrides, split_override=(train_idx, val_idx, test_idx))
            # Save summary
            with open(seed_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Seed {seed} done: val_metric = {summary.get('best_val_metric', 'N/A')}")
        except Exception as e:
            print(f"ERROR: Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- Evaluate each seed ----
    print(f"\n{'='*60}")
    print("  EVALUATION: Generalization Test")
    print(f"{'='*60}\n")

    for seed in args.seeds:
        seed_dir = results_dir / f"seed_{seed}"
        ckpt = seed_dir / "checkpoints" / "stage2_best.pt"
        if not ckpt.exists():
            print(f"Seed {seed}: checkpoint not found at {ckpt}, skipping")
            continue

        output_dir = results_dir / f"eval_seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating seed {seed}...")
        cmd = [
            sys.executable, "-m", "analysis.generalization_test",
            "--checkpoint", str(ckpt),
            "--config", args.config,
            "--output_dir", str(output_dir),
            "--seed", "42",
        ]

        import subprocess
        result = subprocess.run(cmd, cwd=str(_SELF_DIR))
        if result.returncode != 0:
            print(f"WARNING: Evaluation failed for seed {seed}")

    # ---- Aggregate ----
    eval_dirs = sorted(results_dir.glob("eval_seed_*"))
    if len(eval_dirs) > 1:
        print(f"\n{'='*60}")
        print("  AGGREGATING ACROSS SEEDS")
        print(f"{'='*60}\n")

        agg_dir = results_dir / "aggregated"
        cmd = [
            sys.executable, "-m", "analysis.generalization_test", "aggregate",
            "--input_dirs", *[str(d) for d in eval_dirs],
            "--output_dir", str(agg_dir),
        ]
        import subprocess
        subprocess.run(cmd, cwd=str(_SELF_DIR))

    print("\nDone.")


if __name__ == "__main__":
    main()
