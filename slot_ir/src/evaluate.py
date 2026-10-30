"""Evaluation of Stage 2 unmixing checkpoints (unmixing metrics and library matching)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .config import load_config, TrainingConfig
from .data import MixtureDataset, build_dataloader, split_indices
from .models import SlotUnmixer
from .matching import LiquidIRDatabase, match_components_to_smiles, RefinementMatcher


# ---------------------------------------------------------------------------
# Model loading (robust, with architecture detection)
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Path, cfg: TrainingConfig, device: torch.device) -> SlotUnmixer:
    """Load a Stage 2 checkpoint, auto-detecting slot count mismatches."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except RuntimeError:
            # Fallback for zip files with central directory issues
            import io
            with open(checkpoint_path, "rb") as f:
                checkpoint = torch.load(io.BytesIO(f.read()), map_location=device, weights_only=False)
    checkpoint_state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    # Detect max_components from checkpoint via routing projection
    for key in checkpoint_state:
        if "routing.proj.3.weight" in key:
            # Shape: [max_components * d_model, d_model * 2]
            out_features = checkpoint_state[key].shape[0]
            d_model = checkpoint_state[key].shape[1] // 2
            if d_model > 0:
                ckpt_slots = out_features // d_model
                if ckpt_slots != cfg.model.max_components:
                    print(f"[Load] Adjusting max_components {cfg.model.max_components} -> {ckpt_slots} to match checkpoint")
                    cfg.model.max_components = ckpt_slots
            break

    model = SlotUnmixer(cfg.model).to(device)

    missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
    if missing:
        print(f"[Load] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load] Unexpected keys (ignored): {len(unexpected)}")

    model.eval()
    print(f"[Load] Model loaded from {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Per-sample metric computation (not batch-averaged)
# ---------------------------------------------------------------------------

def _per_sample_metrics(
    pred_components: torch.Tensor,
    target_components: torch.Tensor,
) -> List[Dict[str, float]]:
    """Compute unmixing metrics for each sample in the batch individually."""
    bsz, k_pred, length = pred_components.shape
    k_target = target_components.shape[1]
    results = []

    for b in range(bsz):
        # Hungarian matching
        cost = torch.cdist(pred_components[b], target_components[b], p=2)  # [K_pred, K_target]
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        matched_pred = pred_components[b, row_ind]    # [K_matched, L]
        matched_tgt = target_components[b, col_ind]   # [K_matched, L]
        k_matched = len(row_ind)  # min(k_pred, k_target)

        # Per-component cosine + MSE, then average over matched components
        comp_cosines = []
        comp_mses = []
        for ki in range(k_matched):
            p = matched_pred[ki]
            t = matched_tgt[ki]
            pn = p / (p.norm() + 1e-8)
            tn = t / (t.norm() + 1e-8)
            comp_cosines.append((pn * tn).sum().item())
            comp_mses.append(F.mse_loss(p, t).item())

        results.append({
            "component_cosine": float(np.mean(comp_cosines)),
            "component_mse": float(np.mean(comp_mses)),
            "per_component_cosines": comp_cosines,
            "per_component_mses": comp_mses,
            "matched_pred_order": row_ind.tolist(),
            "matched_target_order": col_ind.tolist(),
        })

    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    model: SlotUnmixer,
    loader,
    dataset: MixtureDataset,
    device: torch.device,
    database: Optional[LiquidIRDatabase] = None,
    max_samples: Optional[int] = None,
    peak_matcher: Optional[RefinementMatcher] = None,
) -> Dict:
    """Run evaluation and return aggregated results dict."""
    model.eval()

    all_sample_metrics: List[Dict[str, float]] = []
    all_similarities: List[float] = []

    # Aggregated matching counts (order-invariant)
    total_top1_correct = 0
    total_top5_correct = 0
    total_top10_correct = 0
    total_top50_correct = 0
    total_components = 0
    mixture_top1_matches = []

    # Refinement matching accumulators (peak_BCR joint + residual)
    all_pair_correct: List[List[bool]] = []       # per-sample list of bool at each top-k
    corrected_hits: List[List[int]] = []           # per-sample corrected component hits at each top-k
    all_recon_cosines: List[float] = []
    refine_total_components = 0

    # Stage tracking
    stage_counts = {
        "bypass_both_fixed": 0,
        "refinement_both": 0,
        "refinement_slot_0": 0,
        "refinement_slot_1": 0,
    }

    # Molecule consistency tracking
    molecule_results = {}  # {smiles: {"correct": int, "total": int}}

    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if max_samples is not None and sample_count >= max_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(mixture=batch["mixture"], stage=2)
            pred_components = outputs["components"]        # [B, K, L]
            target_components = batch["components"]        # [B, K, L]
            mixture = batch["mixture"]                     # [B, L]

            # Per-sample unmixing metrics
            batch_metrics = _per_sample_metrics(pred_components, target_components)

            # Trim if max_samples would be exceeded
            remaining = (max_samples - sample_count) if max_samples is not None else len(batch_metrics)
            batch_metrics = batch_metrics[:remaining]
            all_sample_metrics.extend(batch_metrics)

            # Molecule matching (order-invariant)
            if database is not None:
                batch_size = len(batch_metrics)
                gt_smiles = [dataset.get_smiles(sample_count + i) for i in range(batch_size)]
                # Strict scoring: R@K counted on the Hungarian L2 pairs of spectra,
                # the same rule as the baselines (evaluate_baselines.py).
                matching = match_components_to_smiles(
                    pred_components[:batch_size],
                    database,
                    ground_truth_smiles=gt_smiles,
                    top_k=50,
                    pairs=[(m["matched_pred_order"], m["matched_target_order"]) for m in batch_metrics],
                )
                # Collect similarities
                for b in range(batch_size):
                    all_similarities.extend(matching["similarities"][b])

                # Aggregate order-invariant accuracy
                # Use K_true (from data) not K_pred (from model) so K-mismatch
                # experiments (e.g. K=3 model on K=2 data) are evaluated fairly.
                if "per_sample_top1_correct" in matching:
                    n_comp = target_components.shape[1]
                    total_top1_correct += sum(matching["per_sample_top1_correct"])
                    total_top5_correct += sum(matching["per_sample_top5_correct"])
                    total_top10_correct += sum(matching["per_sample_top10_correct"])
                    total_top50_correct += sum(matching["per_sample_top50_correct"])
                    total_components += batch_size * n_comp
                    for i in range(batch_size):
                        mixture_top1_matches.append(matching["per_sample_top1_correct"][i] == n_comp)

                # Refinement matching (peak_BCR: joint + residual), binary only
                if peak_matcher is not None and n_comp == 2:
                    for i in range(batch_size):
                        gt_smi = gt_smiles[i]
                        gt_set = set(gt_smi)
                        try:
                            ref_result = peak_matcher.match_components_joint(
                                pred_components[i],  # [K, L]
                                mixture[i],           # [L]
                                ground_truth_smiles=gt_smi,
                                top_k=50,
                            )
                        except Exception:
                            continue

                        joint_top_k = ref_result["joint_top_k"]
                        all_recon_cosines.append(ref_result["reconstruction_cosine"])
                        refine_total_components += n_comp

                        # Track stage
                        stage = ref_result.get("stage", "unknown")
                        if stage in stage_counts:
                            stage_counts[stage] += 1

                        # Track molecule consistency (top-1 only)
                        top1_pair = joint_top_k[0][0] if joint_top_k else ()
                        for gt_mol in gt_smi:
                            if gt_mol not in molecule_results:
                                molecule_results[gt_mol] = {"correct": 0, "total": 0}
                            molecule_results[gt_mol]["total"] += 1
                            if gt_mol in top1_pair:
                                molecule_results[gt_mol]["correct"] += 1

                        # Compute corrected hits and pair correctness at top-k cutoffs
                        sample_pair_correct = []
                        sample_corrected = []
                        for cutoff in [1, 5, 10, 50]:
                            pairs_at_k = joint_top_k[:cutoff]
                            # Pair correct: both GT molecules appear in any single pair
                            pair_ok = any(
                                set(pair_smi) == gt_set
                                for pair_smi, _ in pairs_at_k
                            )
                            sample_pair_correct.append(pair_ok)
                            # Corrected per-component: each GT mol in any pair
                            all_paired_smiles = set()
                            for pair_smi, _ in pairs_at_k:
                                all_paired_smiles.update(pair_smi)
                            n_hit = sum(1 for g in gt_smi if g in all_paired_smiles)
                            sample_corrected.append(n_hit)

                        all_pair_correct.append(sample_pair_correct)
                        corrected_hits.append(sample_corrected)

            sample_count += len(batch_metrics)

    # Aggregate unmixing metrics
    metric_keys = ["component_cosine", "component_mse"]
    unmixing = {}
    for key in metric_keys:
        vals = [m[key] for m in all_sample_metrics]
        unmixing[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    result: Dict = {
        "n_samples": sample_count,
        "unmixing": unmixing,
    }

    # Aggregate matching metrics (order-invariant)
    if database is not None and total_components > 0:
        result["matching"] = {
            "top1_accuracy": float(total_top1_correct) / total_components,
            "top5_accuracy": float(total_top5_correct) / total_components,
            "top10_accuracy": float(total_top10_correct) / total_components,
            "top50_accuracy": float(total_top50_correct) / total_components,
            "mixture_match_rate": float(np.mean(mixture_top1_matches)),
            "mean_similarity": float(np.mean(all_similarities)),
        }

    # Aggregate refinement matching metrics (peak_BCR)
    if corrected_hits and refine_total_components > 0:
        n_samples_ref = len(corrected_hits)
        n_comp = 2  # binary only
        # corrected_hits[i] = [hits@1, hits@5, hits@10, hits@50]
        # all_pair_correct[i] = [bool@1, bool@5, bool@10, bool@50]
        corrected_arr = np.array(corrected_hits)   # [N, 4]
        pair_arr = np.array(all_pair_correct)       # [N, 4]
        recon_arr = np.array(all_recon_cosines)

        result["refinement_matching"] = {
            "component_top1_accuracy": float(corrected_arr[:, 0].sum()) / refine_total_components,
            "component_top5_accuracy": float(corrected_arr[:, 1].sum()) / refine_total_components,
            "component_top10_accuracy": float(corrected_arr[:, 2].sum()) / refine_total_components,
            "component_top50_accuracy": float(corrected_arr[:, 3].sum()) / refine_total_components,
            "pair_top1_accuracy": float(pair_arr[:, 0].mean()),
            "pair_top5_accuracy": float(pair_arr[:, 1].mean()),
            "pair_top10_accuracy": float(pair_arr[:, 2].mean()),
            "pair_top50_accuracy": float(pair_arr[:, 3].mean()),
            "reconstruction_cosine_mean": float(recon_arr.mean()),
            "reconstruction_cosine_std": float(recon_arr.std()),
            "stage_counts": stage_counts,
            "n_samples_refined": n_samples_ref,
        }

        # Store molecule consistency for later analysis
        result["molecule_consistency"] = molecule_results

    return result


# ---------------------------------------------------------------------------
# Molecule Consistency Analysis
# ---------------------------------------------------------------------------

def run_full_evaluation(
    config_path: Optional[str],
    checkpoint_path: str,
    eval_datasets: List[str],
    output_dir: str,
    split_name: str = "test",
) -> Dict:
    """Run evaluation on multiple datasets and generate summary report."""
    import torch

    # Load config
    if config_path:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        cfg = load_config(**config_data)
    else:
        cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Checkpoint: {checkpoint_path}")

    # Load model
    model = _load_model(Path(checkpoint_path), cfg, device)

    # Create output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    database = None  # Track database for cleanup

    for dataset_file in eval_datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {dataset_file}")
        print("=" * 60)

        # Determine split: use 'test' for training datasets, 'all' for OOD datasets
        # When drug data is included in training, drug datasets also need test-only split
        is_ood = "drugs" in dataset_file.lower()
        if is_ood and cfg.data.include_drug_data:
            split = "test"
        elif is_ood:
            split = "all"
        else:
            split = "test"
        if split_name == "val":
            if split == "all":
                print(f"[Skip] {dataset_file} has no validation split")
                continue
            split = "val"

        # Always use combined database for matching (contains all molecules)
        # This ensures consistent matching across all evaluation datasets
        print(f"[Database] Using combined database for matching (all molecules)")
        database = LiquidIRDatabase(
            cfg.data, apply_normalization=True, device=device,
            database_file="combined_molecule_database.h5"
        )

        try:
            # Load dataset
            base_dataset = MixtureDataset(cfg.data, dataset_file=dataset_file)

            if split == "all":
                dataset = base_dataset
            else:
                train_idx, val_idx, test_idx = split_indices(len(base_dataset), seed=cfg.seed)
                dataset = MixtureDataset(
                    cfg.data, dataset_file=dataset_file,
                    split_indices=val_idx if split == "val" else test_idx,
                )

            print(f"[Data] {len(dataset)} samples ({split} split)")
            loader = build_dataloader(dataset, cfg, shuffle=False)

            # Run evaluation (no refinement matching)
            results = run_evaluation(
                model, loader, dataset, device, database=database, max_samples=None,
                peak_matcher=None,
            )
            results["dataset"] = dataset_file
            results["split"] = split

            all_results[dataset_file] = results

            # Print per-dataset summary
            _print_dataset_summary(dataset_file, results)

        except Exception as e:
            print(f"[Error] Failed to evaluate {dataset_file}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_file] = {"error": str(e)}

    # Save all results
    results_path = out_dir / ("evaluation_results_val.json" if split_name == "val" else "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Save] Results saved to {results_path}")

    # Print final summary table
    _print_summary_table(all_results)

    return all_results


def _print_dataset_summary(dataset_name: str, results: Dict) -> None:
    """Print summary for a single dataset."""
    if "error" in results:
        print(f"  Error: {results['error']}")
        return

    u = results.get("unmixing", {})
    print(f"\n  Samples: {results.get('n_samples', 0)}")
    print(f"\n  Component Metrics:")
    print(f"    Cosine Similarity: {u.get('component_cosine', {}).get('mean', 0):.4f} +/- {u.get('component_cosine', {}).get('std', 0):.4f}")
    print(f"    MSE:               {u.get('component_mse', {}).get('mean', 0):.6f} +/- {u.get('component_mse', {}).get('std', 0):.6f}")

    if "matching" in results:
        m = results["matching"]
        print(f"\n  Cosine Matching (per-component):")
        print(f"    R@1:               {m.get('top1_accuracy', 0):.4f}")
        print(f"    R@5:               {m.get('top5_accuracy', 0):.4f}")
        print(f"    R@10:              {m.get('top10_accuracy', 0):.4f}")
        print(f"    R@50:              {m.get('top50_accuracy', 0):.4f}")
        print(f"\n  Cosine Matching (per-mixture):")
        print(f"    R@1:               {m.get('mixture_match_rate', 0):.4f}")

    if "refinement_matching" in results:
        rm = results["refinement_matching"]
        print(f"\n  Refinement Matching (peak_BCR):")
        print(f"    Component R@1:     {rm.get('component_top1_accuracy', 0):.4f}")
        print(f"    Component R@5:     {rm.get('component_top5_accuracy', 0):.4f}")
        print(f"    Pair R@1:          {rm.get('pair_top1_accuracy', 0):.4f}")
        print(f"    Pair R@5:          {rm.get('pair_top5_accuracy', 0):.4f}")
        print(f"    Recon Cosine:      {rm.get('reconstruction_cosine_mean', 0):.4f}")

        # Stage statistics
        if "stage_counts" in rm:
            n_total = rm.get("n_samples_refined", sum(rm["stage_counts"].values()))
            print(f"\n  Refinement Stage Stats:")
            stage_labels = {
                "bypass_both_fixed": "Bypass (≥0.98)",
                "refinement_both": "Refine (both)",
                "refinement_slot_0": "Refine (slot 0)",
                "refinement_slot_1": "Refine (slot 1)",
            }
            for stage_key, label in stage_labels.items():
                count = rm["stage_counts"].get(stage_key, 0)
                pct = 100.0 * count / n_total if n_total > 0 else 0.0
                print(f"    {label:18s}  {count:5d} ({pct:5.1f}%)")


def _print_summary_table(all_results: Dict) -> None:
    """Print formatted summary table of all results."""
    print("\n" + "=" * 130)
    print("EVALUATION SUMMARY")
    print("=" * 130)

    # Check if any dataset has refinement matching results
    has_refinement = any(
        "refinement_matching" in r for r in all_results.values() if isinstance(r, dict) and "error" not in r
    )

    # Header
    if has_refinement:
        header = (
            f"{'Dataset':<25} {'N':>6} {'Cos':>6} {'Top1':>6} {'Mix':>6}"
            f" | {'cTop1':>6} {'cTop5':>6} {'pTop1':>6} {'pTop5':>6} {'Recon':>6}"
        )
        sep_len = 100
    else:
        header = f"{'Dataset':<25} {'Samples':>7} {'Comp Cos':>9} {'Comp MSE':>11} {'Top1':>7} {'Top5':>7} {'Top10':>7} {'Top50':>7} {'Mix Acc':>8}"
        sep_len = 110
    print(header)
    print("-" * sep_len)

    for dataset_name, results in all_results.items():
        if "error" in results:
            print(f"{dataset_name:<25} ERROR: {results['error']}")
            continue

        u = results.get("unmixing", {})
        m = results.get("matching", {})
        n = results.get("n_samples", 0)

        comp_cos = u.get("component_cosine", {}).get("mean", 0)
        comp_mse = u.get("component_mse", {}).get("mean", 0)
        top1 = m.get("top1_accuracy", 0)
        top5 = m.get("top5_accuracy", 0)
        top10 = m.get("top10_accuracy", 0)
        top50 = m.get("top50_accuracy", 0)
        mix_acc = m.get("mixture_match_rate", 0)

        short_name = dataset_name.replace("_instances.h5", "").replace("mix_", "")

        if has_refinement:
            rm = results.get("refinement_matching", {})
            ct1 = rm.get("component_top1_accuracy", 0)
            ct5 = rm.get("component_top5_accuracy", 0)
            pt1 = rm.get("pair_top1_accuracy", 0)
            pt5 = rm.get("pair_top5_accuracy", 0)
            recon = rm.get("reconstruction_cosine_mean", 0)
            print(
                f"{short_name:<25} {n:>6} {comp_cos:>6.3f} {top1:>6.3f} {mix_acc:>6.3f}"
                f" | {ct1:>6.3f} {ct5:>6.3f} {pt1:>6.3f} {pt5:>6.3f} {recon:>6.3f}"
            )
        else:
            print(f"{short_name:<25} {n:>7} {comp_cos:>9.4f} {comp_mse:>11.6f} {top1:>7.4f} {top5:>7.4f} {top10:>7.4f} {top50:>7.4f} {mix_acc:>8.4f}")

    print("=" * sep_len)

    # Print metrics explanation
    print("\nMetrics:")
    print("  Cos       = Component Cosine Similarity (higher is better, 1.0 = perfect)")
    print("  Top1      = Cosine per-component R@1")
    print("  Mix       = Cosine per-mixture R@1 (all components correct)")
    if has_refinement:
        print("  cTop1/5   = Refinement per-component R@1/5 (from joint pairs)")
        print("  pTop1/5   = Refinement per-mixture R@1/5 (both GT in pair)")
        print("  Recon     = Region-weighted reconstruction cosine")


def evaluate_checkpoint(
    config_data: Dict,
    checkpoint_path: str,
    test_indices: List[int],
    dataset_file: Optional[str] = None,
    no_matching: bool = False,
    no_refinement: bool = False,
) -> Dict:
    """Load a checkpoint and evaluate it on the given test indices."""
    cfg = load_config(**config_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(checkpoint_path), cfg, device)

    ds_file = dataset_file or cfg.data.mixture_dataset
    dataset = MixtureDataset(cfg.data, dataset_file=ds_file,
                             split_indices=test_indices)
    loader = build_dataloader(dataset, cfg, shuffle=False)

    database = None
    peak_matcher = None
    if not no_matching:
        database = LiquidIRDatabase(
            cfg.data, apply_normalization=True, device=device,
            database_file="combined_molecule_database.h5",
        )
        if not no_refinement:
            peak_matcher = RefinementMatcher(database, top_n_pairs=5)

    results = run_evaluation(
        model, loader, dataset, device,
        database=database, peak_matcher=peak_matcher,
    )
    return results
