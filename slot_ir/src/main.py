"""Train one stage from a JSON config, optionally followed by evaluation (used by hyperparam_search.py)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import run_full_evaluation  # noqa: E402
from src.training import train_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SLOT-IR stage from a config file")
    parser.add_argument("--config", type=str, required=True, help="JSON config")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the best Stage 2 checkpoint after training")
    args = parser.parse_args()

    overrides = json.loads(Path(args.config).read_text())
    overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(overrides.get("results_dir", "./results"))

    train_stage(overrides)

    if args.evaluate:
        checkpoint_path = results_dir / "checkpoints" / "stage2_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = results_dir / "checkpoints" / "stage2_last.pt"
        if not checkpoint_path.exists():
            print(f"[Error] No checkpoint found at {checkpoint_path}")
            return
        mixture_dataset = overrides.get("data", {}).get("mixture_dataset", "mix_ml_instances.h5")
        if "ternary" in mixture_dataset:
            eval_datasets = ["mix_ternary_ml_instances.h5"]
        else:
            eval_datasets = ["mix_ml_instances.h5", "mix_drugs_ml_instances.h5"]
        # Test split for reporting, validation split for model selection
        for split_name in ("test", "val"):
            run_full_evaluation(
                config_path=args.config,
                checkpoint_path=str(checkpoint_path),
                eval_datasets=eval_datasets,
                output_dir=str(results_dir / "evaluation"),
                split_name=split_name,
            )


if __name__ == "__main__":
    main()
