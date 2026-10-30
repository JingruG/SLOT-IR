from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import TrainingConfig, load_config, SLOT_UNMIXER_ROOT
from torch.utils.data import ConcatDataset

from .data import (
    PureSpectrumDataset,
    AutoencoderDataset,
    GasAutoencoderDataset,
    ShuffledPairsDataset,
    MixtureDataset,
    SyntheticMixtureDataset,
    build_dataloader,
    split_indices,
    compute_molecule_holdout,
    filter_indices_by_molecules,
)
from .losses import compute_total_loss, hungarian_component_loss
from scipy.optimize import linear_sum_assignment


def match_components_hungarian(
    pred_components: torch.Tensor,
    target_components: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Match predicted components to target components using Hungarian algorithm."""
    bsz, k_pred, length = pred_components.shape
    k_target = target_components.shape[1]
    matched_pred_list = []
    matched_target_list = []
    assignments = []
    
    for b in range(bsz):
        # Compute cost matrix (MSE distance between all pairs)
        # Use all predicted components to find the best matches
        cost = torch.cdist(pred_components[b], target_components[b], p=2)  # [K_pred, K_target]
        cost_np = cost.detach().cpu().numpy()
        
        # Find optimal assignment using Hungarian algorithm
        # This returns min(K_pred, K_target) = K_target assignments
        row_indices, col_indices = linear_sum_assignment(cost_np)
        
        # Reorder components according to optimal assignment
        # row_indices are the indices of the best-matching predicted components
        matched_pred = pred_components[b, row_indices]  # [K_target, L]
        matched_target = target_components[b, col_indices]  # [K_target, L]
        
        matched_pred_list.append(matched_pred)
        matched_target_list.append(matched_target)
        assignments.append((row_indices, col_indices))
    
    matched_pred_tensor = torch.stack(matched_pred_list)  # [B, K_target, L]
    matched_target_tensor = torch.stack(matched_target_list)  # [B, K_target, L]
    
    return matched_pred_tensor, matched_target_tensor, assignments
from .models import SlotUnmixer
from .visualization import (
    plot_mixture_breakdown,
    plot_loss_curves,
    plot_evaluation_metrics,
    plot_translation_examples,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _build_stage_dataset(cfg: TrainingConfig, split_override=None):
    """Build train/val/test datasets for the given stage."""
    stage = cfg.stage.stage
    if stage in (0, 1):
        dataset = PureSpectrumDataset(cfg.data, include_fg=False)
        if split_override is not None:
            train_idx, val_idx, test_idx = split_override
        else:
            train_idx, val_idx, test_idx = split_indices(len(dataset), seed=cfg.seed)
        split = {"train": train_idx, "val": val_idx, "test": test_idx}

        # Select dataset class based on stage1_mode
        stage1_mode = cfg.stage.stage1_mode
        if stage1_mode == "autoencoder":
            DatasetClass = AutoencoderDataset
            print(f"[Data] Stage 1 mode: autoencoder (liq->liq)")
        elif stage1_mode == "gas_autoencoder":
            DatasetClass = GasAutoencoderDataset
            print(f"[Data] Stage 1 mode: gas autoencoder (gas->gas)")
        elif stage1_mode == "shuffled":
            DatasetClass = ShuffledPairsDataset
            print(f"[Data] Stage 1 mode: shuffled pairs (gas_i->liq_j)")
        else:
            DatasetClass = PureSpectrumDataset
            if stage1_mode != "translation":
                print(f"[Warning] Unknown stage1_mode '{stage1_mode}', defaulting to translation")

        if DatasetClass == ShuffledPairsDataset:
            datasets = {
                name: ShuffledPairsDataset(
                    cfg.data,
                    idx,
                    include_fg=False,
                    shuffle_seed=cfg.seed + 1000,
                )
                for name, idx in split.items()
            }
        else:
            datasets = {
                name: DatasetClass(
                    cfg.data,
                    idx,
                    include_fg=False,
                )
                for name, idx in split.items()
            }
        return datasets
    if stage == 2:
        base_dataset = MixtureDataset(cfg.data, dataset_file=cfg.data.mixture_dataset)
        if split_override is not None:
            train_idx, val_idx, test_idx = split_override
        else:
            train_idx, val_idx, test_idx = split_indices(len(base_dataset), seed=cfg.seed)

        # Molecule-level holdout: remove training instances containing held-out molecules
        holdout_molecules = None
        if cfg.data.n_holdout_molecules > 0:
            h_seed = cfg.data.holdout_seed if cfg.data.holdout_seed is not None else cfg.seed
            kept, holdout_molecules = compute_molecule_holdout(cfg.data, train_idx, seed=h_seed)
            train_idx = filter_indices_by_molecules(cfg.data, train_idx, kept)
            val_idx = filter_indices_by_molecules(cfg.data, val_idx, kept)

        split = {"train": train_idx, "val": val_idx, "test": test_idx}
        datasets = {
            name: MixtureDataset(
                cfg.data,
                dataset_file=cfg.data.mixture_dataset,
                split_indices=indices,
                augment=(name == "train"),
                aug_cfg=cfg.augmentation,
            )
            for name, indices in split.items()
        }

        if cfg.data.include_drug_data:
            drug_base = MixtureDataset(cfg.data, dataset_file=cfg.data.drug_mixture_dataset)
            drug_train_idx, drug_val_idx, _ = split_indices(len(drug_base), seed=cfg.seed)
            drug_train = MixtureDataset(
                cfg.data, dataset_file=cfg.data.drug_mixture_dataset, split_indices=drug_train_idx,
                augment=True, aug_cfg=cfg.augmentation,
            )
            drug_val = MixtureDataset(
                cfg.data, dataset_file=cfg.data.drug_mixture_dataset, split_indices=drug_val_idx,
            )
            print(f"[Data] Including drug data: {len(drug_train)} train + {len(drug_val)} val instances")
            datasets["train"] = ConcatDataset([datasets["train"], drug_train])
            datasets["val"] = ConcatDataset([datasets["val"], drug_val])

        # Compositional augmentation: add synthetic mixtures to training set
        if cfg.augmentation.enable and (
            cfg.augmentation.num_synthetic_samples > 0
            or cfg.augmentation.generate_all_combinations
            or cfg.augmentation.source == "stratified"
        ):
            synthetic_train = SyntheticMixtureDataset(
                cfg.data,
                num_components=cfg.model.max_components,
                num_samples=cfg.augmentation.num_synthetic_samples,
                seed=cfg.seed,
                source=cfg.augmentation.source,
                num_drug_train_samples=cfg.augmentation.num_drug_train_samples,
                include_s1_drugs=cfg.augmentation.include_s1_drugs,
                exclude_molecule_indices=holdout_molecules,
                generate_all_combinations=cfg.augmentation.generate_all_combinations,
                sampling_strategy=cfg.augmentation.sampling_strategy,
                hard_negative_temperature=cfg.augmentation.hard_negative_temperature,
                aug_cfg=cfg.augmentation,
            )
            print(f"[Augmentation] Adding {len(synthetic_train)} synthetic "
                  f"{cfg.model.max_components}-component mixtures "
                  f"(source: {cfg.augmentation.source}, shift: {cfg.augmentation.shift_percent:.1%})")
            datasets["train"] = ConcatDataset([datasets["train"], synthetic_train])

        # Print dataset statistics
        print(f"\n[Data] Final dataset sizes:")
        print(f"  Train: {len(datasets['train'])} samples")
        print(f"  Val:   {len(datasets['val'])} samples")
        print(f"  Test:  {len(datasets['test'])} samples")

        return datasets
    dataset = {
        "train": MixtureDataset(cfg.data, dataset_file=cfg.data.drug_mixture_dataset),
        "val": MixtureDataset(cfg.data, dataset_file=cfg.data.drug_mixture_dataset),
        "test": MixtureDataset(cfg.data, dataset_file=cfg.data.drug_mixture_dataset),
    }
    return dataset


def load_model(cfg: TrainingConfig) -> SlotUnmixer:
    """Load model, handling architecture mismatches gracefully."""
    model = SlotUnmixer(cfg.model)

    if cfg.stage.checkpoint_path is not None:
        checkpoint_path = Path(cfg.stage.checkpoint_path)
        # If path doesn't exist, try to find best or last checkpoint
        if not checkpoint_path.exists():
            checkpoint_dir = checkpoint_path.parent
            if checkpoint_dir.exists():
                # Try best first, then last
                for tag in ["best", "last"]:
                    alt_path = checkpoint_dir / f"stage{cfg.stage.stage - 1}_{tag}.pt"
                    if alt_path.exists():
                        checkpoint_path = alt_path
                        print(f"[Load] Using {alt_path.name} instead of {checkpoint_path.name}")
                        break
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_state = checkpoint.get("model", checkpoint)
            
            # Check for architecture mismatch (different max_components)
            model_state = model.state_dict()
            checkpoint_max_components = None
            model_max_components = cfg.model.max_components
            
            # Try to infer max_components from checkpoint via routing projection
            for key in checkpoint_state.keys():
                if "routing.proj.3.weight" in key:
                    # Shape: [max_components * d_model, d_model * 2]
                    out_features = checkpoint_state[key].shape[0]
                    d_model = checkpoint_state[key].shape[1] // 2
                    if d_model > 0:
                        checkpoint_max_components = out_features // d_model
                    break
            
            # For Stage 2, filter out routing and prediction components if they don't match
            # This allows loading Stage 1 components even if checkpoint has wrong slot count
            if cfg.stage.stage == 2 and checkpoint_max_components is not None:
                if checkpoint_max_components != model_max_components:
                    print(f"[Load] Architecture mismatch detected:")
                    print(f"  Checkpoint has {checkpoint_max_components} slots (from Stage 2 checkpoint)")
                    print(f"  Model config has {model_max_components} slots")
                    print(f"  Filtering out routing and prediction components...")
                    
                    # Filter out routing and prediction head components
                    filtered_state = {}
                    for key, value in checkpoint_state.items():
                        if not any(x in key for x in ["routing.", "prediction"]):
                            filtered_state[key] = value
                    checkpoint_state = filtered_state
                    print(f"  Loading {len(checkpoint_state)} compatible components (encoder, tokenizer, etc.)")
            
            if cfg.stage.stage == 2 and getattr(cfg.stage, "reinit_heads", False):
                n_before = len(checkpoint_state)
                checkpoint_state = {k: v for k, v in checkpoint_state.items()
                                    if not any(x in k for x in ["routing.", "prediction"])}
                print(f"[Load] reinit_heads: dropped {n_before - len(checkpoint_state)} routing/prediction "
                      f"tensors; heads start from fresh random weights")

            # Load state dict with strict=False to handle mismatches
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
                if missing_keys:
                    # Filter out expected missing keys (routing/prediction if filtered)
                    routing_pred_keys = [k for k in missing_keys if any(x in k for x in ["routing", "prediction"])]
                    other_missing = [k for k in missing_keys if k not in routing_pred_keys]
                    if routing_pred_keys:
                        print(f"[Load] Routing/prediction components will be randomly initialized ({len(routing_pred_keys)} keys)")
                    if other_missing:
                        print(f"[Load] Other missing keys: {len(other_missing)} keys")
                if unexpected_keys:
                    print(f"[Load] Unexpected keys (ignored): {len(unexpected_keys)} keys")
                print(f"[Load] ✓ Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"[Warning] Error loading checkpoint: {e}")
                print(f"  Model will be randomly initialized")

    return model


def _git_commit() -> str:
    """Return the current git commit hash (with '-dirty' if there are local edits)."""
    import subprocess
    try:
        root = Path(__file__).resolve().parent
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def save_run_config(cfg: TrainingConfig, cfg_overrides: Dict | None = None) -> None:
    """Save the resolved config, the raw overrides and the git commit to results_dir/run_config.json."""
    import dataclasses
    out = Path(cfg.results_dir) / "run_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "git_commit": _git_commit(),
        "overrides": cfg_overrides or {},
        "resolved_config": dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg),
    }
    with open(out, "w") as f:
        json.dump(record, f, indent=2, default=str)


def train_stage(cfg_overrides: Dict | None = None, split_override=None) -> Dict:
    """Train a single stage and return the training summary."""
    import copy
    raw_overrides = copy.deepcopy(cfg_overrides or {})
    cfg = load_config(**(cfg_overrides or {}))
    save_run_config(cfg, raw_overrides)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Print key config values for verification
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Stage: {cfg.stage.stage}")
    print(f"  max_components (slots): {cfg.model.max_components}")
    print(f"  results_dir: {cfg.results_dir}")
    print(f"  checkpoint_path: {cfg.stage.checkpoint_path}")
    if cfg.stage.stage == 2:
        print(f"  mixture_dataset: {cfg.data.mixture_dataset}")
        if cfg.data.include_drug_data:
            print(f"  include_drug_data: True ({cfg.data.drug_mixture_dataset})")
        if cfg.augmentation.enable:
            aug = cfg.augmentation
            if aug.generate_all_combinations:
                print(f"  augmentation: ON (compositional: ALL {cfg.model.max_components}-combinations)")
            elif aug.num_synthetic_samples > 0:
                print(f"  augmentation: ON (compositional: +{aug.num_synthetic_samples} synthetic mixtures)")
    print("=" * 60 + "\n")
    
    datasets = _build_stage_dataset(cfg, split_override=split_override)
    train_loader = build_dataloader(datasets["train"], cfg, shuffle=True)
    val_loader = build_dataloader(datasets["val"], cfg, shuffle=False)

    model = load_model(cfg).to(device)
    
    # Verify model architecture
    print(f"[Model] Created ({type(model).__name__}, K={cfg.model.max_components})")

    optimizer = AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(len(train_loader), 1))

    # Loss tracking (step-level for plotting)
    train_loss_history = {}
    val_loss_history = {}
    val_metrics_history = {}

    # Epoch-level tracking (for convergence curves)
    epoch_history = {
        'epoch': [],
        'train_loss': [],
        'train_metrics': [],
        'val_loss': [],
        'val_metrics': [],
        'learning_rate': [],
    }

    # Track best model
    best_val_loss = float('inf')
    best_val_metric = -float('inf')  # For metrics like cosine (higher is better)
    best_checkpoint_path = None
    best_epoch = 0

    # Early stopping tracking
    early_stopping_patience = cfg.stage.early_stopping_patience
    min_delta_loss = cfg.stage.early_stopping_min_delta_loss
    min_delta_metric = cfg.stage.early_stopping_min_delta_metric
    patience_counter = 0
    should_stop_early = False

    global_step = 0
    model.train()
    max_epochs = cfg.stage.max_epochs or 1000
    for epoch in range(max_epochs):
        progress = tqdm(train_loader, desc=f"Stage {cfg.stage.stage} Epoch {epoch}", leave=False)
        for batch in progress:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(
                gas=batch.get("gas"),
                mixture=batch.get("mixture"),
                fg=batch.get("fg"),
                stage=cfg.stage.stage,
            )
            loss_dict = compute_total_loss(
                outputs,
                batch,
                {**cfg.optim.stage_loss_weights, "confidence_temperature": cfg.model.confidence_temperature},
                stage=cfg.stage.stage,
            )

            loss_dict["total"].backward()
            max_norm = (
                cfg.optim.max_grad_norm_stage_pretrain
                if cfg.stage.stage <= 2
                else cfg.optim.max_grad_norm_self_train
            )
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()

            # Track training losses
            for key, value in loss_dict.items():
                if key not in train_loss_history:
                    train_loss_history[key] = []
                train_loss_history[key].append(value.item())

            global_step += 1
            # Format losses with appropriate precision (4 digits for losses, 3 for grad_norm)
            postfix = {k: f"{v.item():.4f}" for k, v in loss_dict.items() if k != "total"}
            postfix["grad_norm"] = f"{total_grad_norm:.3f}"
            
            # Add diagnostic info every 100 steps
            if global_step % 100 == 0:
                with torch.no_grad():
                    if cfg.stage.stage in (0, 1):
                        pred_mean = outputs["liq_pred"].mean().item()
                        target_mean = batch["liq"].mean().item()
                        postfix["pred_m"] = f"{pred_mean:.3f}"
                        postfix["tgt_m"] = f"{target_mean:.3f}"
                    else:
                        comp_mean = outputs["components"].mean().item()
                        mix_mean = batch["mixture"].mean().item()
                        postfix["comp_m"] = f"{comp_mean:.3f}"
                        postfix["mix_m"] = f"{mix_mean:.3f}"
            
            progress.set_postfix(postfix)

            if global_step % cfg.val_every == 0:
                val_summary = evaluate(model, val_loader, device, cfg, val_loss_history)
                if val_summary:
                    val_metrics_history[global_step] = val_summary
                    # Save best checkpoint if validation loss improved
                    current_val_loss = val_summary.get("total", float('inf'))
                    
                    # Get key metric for early stopping (cosine similarity for Stage 2, loss for Stage 1)
                    if cfg.stage.stage >= 2:
                        current_metric = val_summary.get("component_cosine", -float('inf'))
                        metric_name = "component_cosine"
                    else:
                        current_metric = val_summary.get("cosine_sim", -float('inf'))
                        metric_name = "cosine_sim"
                    
                    # Check for improvement
                    loss_improved = (best_val_loss - current_val_loss) >= min_delta_loss
                    metric_improved = (current_metric - best_val_metric) >= min_delta_metric
                    improved = loss_improved or metric_improved
                    
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        # Remove the previous best checkpoint
                        if best_checkpoint_path and best_checkpoint_path.exists():
                            best_checkpoint_path.unlink()
                        # Save new best checkpoint
                        best_checkpoint_path = save_checkpoint(model, optimizer, cfg, "best")
                        print(f"[Checkpoint] Saved best model (val_loss={best_val_loss:.4f})")
                    
                    if current_metric > best_val_metric:
                        best_val_metric = current_metric
                    
                    # Early stopping logic
                    if early_stopping_patience is not None:
                        if improved:
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                            if patience_counter >= early_stopping_patience:
                                print(f"\n[Early Stopping] No improvement for {patience_counter} evaluations "
                                      f"(loss: {current_val_loss:.4f}, {metric_name}: {current_metric:.4f})")
                                print(f"Best model: loss={best_val_loss:.4f}, {metric_name}={best_val_metric:.4f}")
                                should_stop_early = True
                                break
                
                # Save loss curves periodically
                if train_loss_history and val_loss_history:
                    plot_loss_curves(train_loss_history, val_loss_history, cfg.viz, cfg.stage.stage)
        
        # Break from outer loop if early stopping triggered
        if should_stop_early:
            break
    
    # Final evaluation and visualization
    final_metrics = evaluate(model, val_loader, device, cfg, val_loss_history)
    if train_loss_history and val_loss_history:
        plot_loss_curves(train_loss_history, val_loss_history, cfg.viz, cfg.stage.stage)
    if val_metrics_history:
        plot_evaluation_metrics(val_metrics_history, cfg.viz, cfg.stage.stage)
    
    # Save training summary (used by hyperparameter search to collect Stage 1 metrics)
    summary_path = cfg.results_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    training_summary = {
        "stage": cfg.stage.stage,
        "best_val_loss": best_val_loss,
        "best_val_metric": best_val_metric,
        "final_metrics": final_metrics if final_metrics else {},
    }
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)

    # Save final checkpoint
    final_checkpoint_path = save_checkpoint(model, optimizer, cfg, "last")
    print(f"[Checkpoint] Saved final model")
    
    # Generate interpretability plots (pass validation dataset for SMILES lookup)
    generate_interpretability_plots(model, val_loader, device, cfg, dataset=datasets.get("val"))

    return training_summary


def _print_eval_summary(stage: int, loss_summary: Dict[str, float], metrics_summary: Dict[str, float]) -> None:
    """Print formatted evaluation summary."""
    if stage in (0, 1):
        # Stage 1: Translation metrics (single line)
        print(f"[Eval][Stage {stage}] "
              f"loss={loss_summary.get('total', 0):.4f} | "
              f"trans={loss_summary.get('translation', 0):.4f}, "
              f"sim={loss_summary.get('similarity', 0):.4f} | "
              f"mse={metrics_summary.get('mse', 0):.4f}, "
              f"cos={metrics_summary.get('cosine_sim', 0):.4f}, "
              f"r={metrics_summary.get('pearson', 0):.4f}")
    else:
        # Stage 2: Unmixing metrics (single line, include only available losses)
        extras = []
        for key in ("component",):
            if key in loss_summary:
                extras.append(f"{key[:4]}={loss_summary[key]:.4f}")
        extras_str = ", ".join(extras)
        metrics_parts = [
            f"c_mse={metrics_summary.get('component_mse', 0):.4f}",
            f"c_cos={metrics_summary.get('component_cosine', 0):.4f}",
        ]
        metrics_str = ", ".join(metrics_parts)
        line = f"[Eval][Stage {stage}] loss={loss_summary.get('total', 0):.4f}"
        if extras_str:
            line += f" | {extras_str}"
        line += f" | {metrics_str}"
        print(line)


def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    stage: int,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    with torch.no_grad():
        if stage in (0, 1):
            pred = outputs["liq_pred"]
            target = batch["liq"]
            # MSE
            metrics["mse"] = torch.nn.functional.mse_loss(pred, target).item()
            # Cosine similarity
            pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
            target_norm = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
            metrics["cosine_sim"] = (pred_norm * target_norm).sum(dim=-1).mean().item()
            # Pearson correlation
            pred_centered = pred - pred.mean(dim=-1, keepdim=True)
            target_centered = target - target.mean(dim=-1, keepdim=True)
            numerator = (pred_centered * target_centered).sum(dim=-1)
            pred_std = pred_centered.norm(dim=-1) + 1e-8
            target_std = target_centered.norm(dim=-1) + 1e-8
            metrics["pearson"] = (numerator / (pred_std * target_std)).mean().item()
        else:
            pred_components = outputs["components"]  # [B, K_pred, L]
            target_components = batch["components"]  # [B, K_target, L]
            
            # Apply Hungarian matching to find optimal component assignment
            # This ensures we compare each predicted component to its best-matching GT component
            # Note: match_components_hungarian now returns [B, K_target, L] tensors
            matched_pred_tensor, matched_target_tensor, _ = match_components_hungarian(
                pred_components, target_components
            )
            
            # Compute per-component metrics
            bsz, k_target, length = matched_pred_tensor.shape
            
            component_mses = []
            component_cosines = []
            
            for b in range(bsz):
                for k_idx in range(k_target):
                    comp_mse = torch.nn.functional.mse_loss(
                        matched_pred_tensor[b, k_idx], matched_target_tensor[b, k_idx]
                    ).item()
                    component_mses.append(comp_mse)
                    
                    # Component cosine similarity
                    pred_norm = matched_pred_tensor[b, k_idx] / (matched_pred_tensor[b, k_idx].norm() + 1e-8)
                    target_norm = matched_target_tensor[b, k_idx] / (matched_target_tensor[b, k_idx].norm() + 1e-8)
                    comp_cosine = (pred_norm * target_norm).sum().item()
                    component_cosines.append(comp_cosine)
            
            # Aggregate component metrics
            metrics["component_mse"] = sum(component_mses) / len(component_mses) if component_mses else 0.0
            metrics["component_cosine"] = sum(component_cosines) / len(component_cosines) if component_cosines else 0.0
    return metrics


def evaluate(
    model: SlotUnmixer,
    loader,
    device: torch.device,
    cfg: TrainingConfig,
    loss_history: Optional[Dict[str, list]] = None,
) -> Dict[str, float]:
    model.eval()
    all_losses = []
    all_metrics = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(
                gas=batch.get("gas"),
                mixture=batch.get("mixture"),
                fg=batch.get("fg"),
                stage=cfg.stage.stage,
            )
            loss_dict = compute_total_loss(
                outputs,
                batch,
                {**cfg.optim.stage_loss_weights, "confidence_temperature": cfg.model.confidence_temperature},
                stage=cfg.stage.stage,
            )
            all_losses.append({k: v.item() for k, v in loss_dict.items()})
            metrics = compute_metrics(outputs, batch, cfg.stage.stage)
            all_metrics.append(metrics)
    
    if all_losses:
        loss_summary = {k: sum(m[k] for m in all_losses) / len(all_losses) for k in all_losses[0]}
        metrics_summary = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        summary = {**loss_summary, **metrics_summary}
        
        # Format evaluation output cleanly
        _print_eval_summary(cfg.stage.stage, loss_summary, metrics_summary)
        
        if loss_history is not None:
            for key in loss_summary:
                if key not in loss_history:
                    loss_history[key] = []
                loss_history[key].append(loss_summary[key])
        
        return summary
    model.train()
    return {}


def save_checkpoint(
    model: SlotUnmixer,
    optimizer: AdamW,
    cfg: TrainingConfig,
    tag: str,
    epoch: Optional[int] = None,
    training_history: Optional[Dict] = None,
) -> Path:
    """Save checkpoint with given tag (e.g., 'best' or 'last')."""
    # Save checkpoints under the run's results directory:
    #   - results/binary/checkpoints/ (binary runs)
    #   - results/ternary/checkpoints/ (ternary runs)
    results_dir = cfg.results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"stage{cfg.stage.stage}_{tag}.pt"

    checkpoint_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # Add metadata
    if epoch is not None:
        checkpoint_data["epoch"] = epoch

    if training_history is not None:
        checkpoint_data["training_history"] = training_history

    torch.save(checkpoint_data, path)
    return path


def generate_interpretability_plots(
    model: SlotUnmixer,
    loader,
    device: torch.device,
    cfg: TrainingConfig,
    dataset=None,
) -> None:
    """Generate interpretability plots after training."""
    model.eval()
    with torch.no_grad():
        # Get a batch for visualization
        batch = None
        for b in loader:
            batch = _to_device(b, device)
            break
        
        if batch is None:
            return
        
        outputs = model(
            gas=batch.get("gas"),
            mixture=batch.get("mixture"),
            fg=batch.get("fg"),
            stage=cfg.stage.stage,
        )
        
        if cfg.stage.stage in (0, 1):
            # Stage 1: Translation interpretability
            # Get dataset indices if available (from dataset's indices attribute)
            sample_indices = None
            if dataset is not None and hasattr(dataset, "indices"):
                # Get the actual dataset indices for the batch
                # Note: This is approximate since we don't track exact batch indices
                # We'll use the first batch's indices as a best guess
                if len(dataset.indices) > 0:
                    # Use first few indices from validation set
                    sample_indices = dataset.indices[:min(batch["gas"].shape[0], len(dataset.indices))]
            
            plot_translation_examples(
                batch["gas"],
                outputs["liq_pred"],
                batch["liq"],
                cfg.viz,
                f"stage{cfg.stage.stage}_final",
                max_samples=cfg.viz.max_samples,
                sample_indices=sample_indices,
            )
        else:
            # Stage 2: Mixture unmixing interpretability
            for idx in range(min(batch["mixture"].shape[0], cfg.viz.max_samples)):
                components_pred = outputs["components"][idx].cpu()
                components_true = batch["components"][idx].cpu()  # Ground truth components
                mixture = batch["mixture"][idx].cpu()
                plot_mixture_breakdown(
                    mixture,
                    components_pred,
                    cfg.viz,
                    f"stage{cfg.stage.stage}_sample{idx}",
                    components_true=components_true,
                )
            
    model.train()


def visualize_batch(model: SlotUnmixer, batch: Dict[str, torch.Tensor], cfg: TrainingConfig) -> None:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(
            gas=batch.get("gas"),
            mixture=batch.get("mixture"),
            stage=cfg.stage.stage,
        )
    if cfg.stage.stage >= 2:
        components_pred = outputs["components"][0].cpu()
        components_true = batch["components"][0].cpu()  # Ground truth components
        mixture = batch["mixture"][0].cpu()
        plot_mixture_breakdown(mixture, components_pred, cfg.viz, tag="sample0", 
                              components_true=components_true)
    model.train()

