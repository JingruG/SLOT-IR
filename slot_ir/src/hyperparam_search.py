#!/usr/bin/env python3
"""Grid search for Stage 1, Stage 2 and the no-Stage-1 ablation, ranked by validation scores.

Usage (from slot_ir/):
    python -m src.hyperparam_search --mode stage1 --gpus 0 1 2 3
    python -m src.hyperparam_search --mode binary --gpus 0 1 2 3
    python -m src.hyperparam_search --mode ternary --gpus 0 1 2 3
    python -m src.hyperparam_search --mode no_s1 --gpus 0 1 2 3
"""

import argparse
import concurrent.futures
import itertools
import json
import os
import queue
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

# Stage 1: ranked by validation translation cosine
SEARCH_SPACE_STAGE1 = {
    "lr": [5e-5, 1e-4],
    "dropout": [0.1, 0.5],
}

# Stage 2: ranked by validation component cosine. Token windows are fixed by the
# Stage 1 checkpoint, so they are not searched here.
SEARCH_SPACE_STAGE2_BINARY = {
    "lr": [5e-5, 1e-4],
    "dropout": [0.3, 0.5],
}
SEARCH_SPACE_STAGE2_TERNARY = {
    "lr": [5e-5, 1e-4, 5e-4],
    "dropout": [0.3, 0.5],
}

# Stage 2 from random initialization (no Stage 1): 36 trials
SEARCH_SPACE_NO_S1 = {
    "lr": [1e-4, 5e-4, 1e-3],
    "dropout": [0.1, 0.3, 0.5],
    "epochs": [100, 200],
    "token_window_sizes": [[10, 25, 50], [5, 10, 25]],
}

# Map mode -> search space and base config
SEARCH_SPACES = {
    "stage1": SEARCH_SPACE_STAGE1,
    "binary": SEARCH_SPACE_STAGE2_BINARY,
    "ternary": SEARCH_SPACE_STAGE2_TERNARY,
    "no_s1": SEARCH_SPACE_NO_S1,
}

BASE_CONFIGS = {
    "stage1": "stage1.json",
    "binary": "stage2_binary.json",
    "ternary": "stage2_ternary.json",
    "no_s1": "ablation/no_s1.json",
}

# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_grid_configs(search_space: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations from search space (grid search)."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    return configs


def create_config_file(
    base_config: Dict,
    hyperparam_config: Dict,
    trial_id: int,
    output_dir: Path,
) -> Path:
    """Create a config file for a specific hyperparameter configuration."""
    config = json.loads(json.dumps(base_config))  # deep copy

    # Learning rate
    if "lr" in hyperparam_config:
        config.setdefault("optim", {})["lr"] = hyperparam_config["lr"]

    # Weight decay
    if "weight_decay" in hyperparam_config:
        config.setdefault("optim", {})["weight_decay"] = hyperparam_config["weight_decay"]

    # Batch size
    if "batch_size" in hyperparam_config:
        config.setdefault("training", {})["batch_size"] = hyperparam_config["batch_size"]

    # Model parameters
    model_params = [
        "d_model", "n_heads", "n_encoder_layers", "n_cross_layers",
        "dropout", "token_window_sizes",
    ]
    for param in model_params:
        if param in hyperparam_config:
            config.setdefault("model", {})[param] = hyperparam_config[param]

    # Epochs (training.epochs -> stage.max_epochs via load_config)
    if "epochs" in hyperparam_config:
        config.setdefault("training", {})["epochs"] = hyperparam_config["epochs"]

    # Loss weights
    if "component_weight" in hyperparam_config:
        config.setdefault("optim", {}).setdefault(
            "stage_loss_weights", {},
        )["component"] = hyperparam_config["component_weight"]
    if "similarity_weight" in hyperparam_config:
        config.setdefault("optim", {}).setdefault(
            "stage_loss_weights", {},
        )["similarity"] = hyperparam_config["similarity_weight"]

    # Set results directory for this trial
    config["results_dir"] = str(output_dir / f"trial_{trial_id:03d}")

    config_path = output_dir / f"config_trial_{trial_id:03d}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(config_path: Path, trial_id: int, mode: str, gpu_id: Optional[int] = None) -> Dict:
    """Run a single training trial."""
    cmd = [sys.executable, "-m", "src.main", "--config", str(config_path)]
    # Stage 2 runs full mixture evaluation; Stage 1 relies on training_summary.json
    if mode != "stage1":
        cmd.append("--evaluate")

    gpu_tag = f" [GPU {gpu_id}]" if gpu_id is not None else ""
    print(f"\n{'='*60}")
    print(f"Trial {trial_id}{gpu_tag}: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(cmd, env=env, timeout=3600 * 12)  # 12 hour timeout
        if result.returncode != 0:
            print(f"[ERROR] Trial {trial_id} failed with return code {result.returncode}")
            return {
                "trial_id": trial_id,
                "status": "failed",
                "error": f"Return code: {result.returncode}",
            }
        print(f"[SUCCESS] Trial {trial_id} completed")
        return {"trial_id": trial_id, "status": "success"}

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Trial {trial_id} timed out")
        return {"trial_id": trial_id, "status": "timeout"}
    except Exception as e:
        print(f"[ERROR] Trial {trial_id} exception: {e}")
        return {"trial_id": trial_id, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_results(output_dir: Path, n_trials: int, mode: str) -> List[Dict]:
    """Collect results from all completed trials."""
    results = []

    for trial_id in range(n_trials):
        trial_dir = output_dir / f"trial_{trial_id:03d}"
        config_file = output_dir / f"config_trial_{trial_id:03d}.json"
        result = {"trial_id": trial_id}

        # Load config
        if config_file.exists():
            with open(config_file) as f:
                result["config"] = json.load(f)

        # Training summary (saved by train_stage for all stages)
        summary_file = trial_dir / "training_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            result["training_summary"] = summary

            if mode == "stage1":
                metrics = summary.get("final_metrics", {})
                result["cosine_sim"] = metrics.get("cosine_sim", 0)
                result["mse"] = metrics.get("mse", 0)
                result["pearson"] = metrics.get("pearson", 0)
                result["best_val_loss"] = summary.get("best_val_loss", float("inf"))

        # Stage 2: read full evaluation results. Test metrics are
        # stored under "<dataset>_*", validation metrics under "val_<dataset>_*".
        # Trials are RANKED by validation metrics (see _sort_key); test is shown only.
        if mode not in ("stage1",):
            for fname, key_prefix in (("evaluation_results.json", ""), ("evaluation_results_val.json", "val_")):
                eval_file = trial_dir / "evaluation" / fname
                if not eval_file.exists():
                    continue
                with open(eval_file) as f:
                    eval_data = json.load(f)
                result["evaluation" if not key_prefix else "evaluation_val"] = eval_data

                for dataset_name, dataset_results in eval_data.items():
                    if not isinstance(dataset_results, dict):
                        continue
                    if "unmixing" not in dataset_results:
                        continue
                    prefix = key_prefix + dataset_name.replace("_instances.h5", "").replace("mix_", "")
                    u = dataset_results["unmixing"]
                    result[f"{prefix}_comp_cos"] = u.get("component_cosine", {}).get("mean", 0)
                    if "matching" in dataset_results:
                        m = dataset_results["matching"]
                        result[f"{prefix}_top1"] = m.get("top1_accuracy", 0)
                        result[f"{prefix}_top5"] = m.get("top5_accuracy", 0)

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_results_summary(results: List[Dict], mode: str):
    """Print ranked summary of hyperparameter search results."""
    print("\n" + "=" * 110)
    print(f"HYPERPARAMETER SEARCH RESULTS ({mode.upper()})")
    print("=" * 110)

    if mode == "stage1":
        _print_stage1_summary(results)
    else:
        _print_stage2_summary(results, mode)


def _print_stage1_summary(results: List[Dict]):
    """Print Stage 1 results sorted by cosine similarity."""
    valid = [r for r in results if r.get("cosine_sim", 0) > 0]
    if not valid:
        print("No valid results found.")
        return

    sorted_results = sorted(valid, key=lambda x: x["cosine_sim"], reverse=True)

    print(f"\n{'Rank':<6} {'Trial':<7} {'LR':<10} {'BS':<5} {'Drop':<6} "
          f"{'WD':<10} {'Cos Sim':<10} {'MSE':<12} {'Pearson':<10} {'Val Loss':<10}")
    print("-" * 96)

    for rank, r in enumerate(sorted_results, 1):
        cfg = r.get("config", {})
        lr = cfg.get("optim", {}).get("lr", "?")
        bs = cfg.get("training", {}).get("batch_size", "?")
        drop = cfg.get("model", {}).get("dropout", "?")
        wd = cfg.get("optim", {}).get("weight_decay", "?")
        print(
            f"{rank:<6} {r['trial_id']:<7} {lr:<10} {bs:<5} {drop:<6} {wd:<10} "
            f"{r['cosine_sim']:<10.4f} {r.get('mse', 0):<12.6f} "
            f"{r.get('pearson', 0):<10.4f} {r.get('best_val_loss', 0):<10.4f}"
        )

    _print_best_config(sorted_results[0])


def _print_stage2_summary(results: List[Dict], mode: str):
    """Print Stage 2 results sorted by VALIDATION component cosine (test shown only)."""
    if mode in ("binary", "no_s1"):
        sort_key = "val_ml_comp_cos"
        prefix = "ml"
        drug_prefix = "drugs_ml"
    else:
        sort_key = "val_ternary_ml_comp_cos"
        prefix = "ternary_ml"
        drug_prefix = None

    valid = [r for r in results if sort_key in r]
    if not valid:
        print("No valid results found.")
        return

    sorted_results = sorted(valid, key=lambda x: x.get(sort_key, 0), reverse=True)

    header = (
        f"{'Rank':<6} {'Trial':<7} {'LR':<10} {'Drop':<6} "
        f"{'WD':<10} {'BS':<5} {'Ep':<5} {'WinSizes':<14} "
        f"{'ValCCos':<9} {'CompCos':<9} {'Top1':<8} {'Top5':<8}"
    )
    if drug_prefix:
        header += f" {'DrugCCos':<9} {'DrugTop1':<9}"
    print(f"\n{header}")
    print("-" * len(header))

    for rank, r in enumerate(sorted_results, 1):
        cfg = r.get("config", {})
        lr = cfg.get("optim", {}).get("lr", "?")
        drop = cfg.get("model", {}).get("dropout", "?")
        wd = cfg.get("optim", {}).get("weight_decay", "?")
        bs = cfg.get("training", {}).get("batch_size", "?")
        ep = cfg.get("training", {}).get("epochs", "?")
        tws = cfg.get("model", {}).get("token_window_sizes", "?")
        tws_str = ",".join(str(s) for s in tws) if isinstance(tws, list) else str(tws)

        row = (
            f"{rank:<6} {r['trial_id']:<7} {lr:<10} {drop:<6} {wd:<10} "
            f"{bs:<5} {ep:<5} {tws_str:<14} "
            f"{r.get(sort_key, 0):<9.4f} "
            f"{r.get(f'{prefix}_comp_cos', 0):<9.4f} "
            f"{r.get(f'{prefix}_top1', 0):<8.4f} "
            f"{r.get(f'{prefix}_top5', 0):<8.4f}"
        )
        if drug_prefix:
            row += (
                f" {r.get(f'{drug_prefix}_comp_cos', 0):<9.4f} "
                f"{r.get(f'{drug_prefix}_top1', 0):<9.4f}"
            )
        print(row)

    _print_best_config(sorted_results[0])


def _print_best_config(best: Dict):
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(json.dumps(best.get("config", {}), indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grid search for Stage 1 and Stage 2")
    parser.add_argument("--mode", type=str, choices=list(SEARCH_SPACES), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to run trials on in parallel (e.g. --gpus 0 1 2 3)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "results" / "hyperparam_search" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    search_space = SEARCH_SPACES[args.mode]
    base_config_path = script_dir / "configs" / BASE_CONFIGS[args.mode]
    with open(base_config_path) as f:
        base_config = json.load(f)
    configs = generate_grid_configs(search_space)
    print(f"Mode: {args.mode}, base config: {base_config_path}, {len(configs)} trials")

    trial_args = [(hp, create_config_file(base_config, hp, i, output_dir)) for i, hp in enumerate(configs)]
    if args.gpus:
        gpu_pool: queue.Queue = queue.Queue()
        for g in args.gpus:
            gpu_pool.put(g)

        def _run_on_gpu(trial_id: int, hp_config: Dict, config_path: Path) -> Dict:
            gpu_id = gpu_pool.get()
            try:
                result = run_trial(config_path, trial_id, args.mode, gpu_id=gpu_id)
                result["hyperparams"] = hp_config
                return result
            finally:
                gpu_pool.put(gpu_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
            futures = [ex.submit(_run_on_gpu, i, hp, cp) for i, (hp, cp) in enumerate(trial_args)]
            trial_results = [f.result() for f in futures]
    else:
        trial_results = []
        for trial_id, (hp_config, config_path) in enumerate(trial_args):
            result = run_trial(config_path, trial_id, args.mode)
            result["hyperparams"] = hp_config
            trial_results.append(result)

    metadata = {"mode": args.mode, "n_trials": len(configs), "search_space": search_space,
                "base_config": str(base_config_path), "trials": trial_results}
    with open(output_dir / "search_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    results = collect_results(output_dir, len(configs), args.mode)
    print_results_summary(results, args.mode)
    summary_path = output_dir / f"search_{args.mode}_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
