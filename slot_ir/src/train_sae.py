"""Training script for SAE interpretability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import TrainingConfig, load_config
from .data import (
    PureSpectrumDataset,
    MixtureDataset,
    build_dataloader,
    split_indices,
)
from .models import SlotUnmixer
from .sae import BatchTopKSAE, AuxiliaryLoss


def collect_activations(
    model: SlotUnmixer,
    dataloader: DataLoader,
    device: torch.device,
    stage: int = 1,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Collect activations from trained model for SAE training."""
    model.eval()

    all_activations = []
    all_fg_labels = []
    all_slot_indices = []
    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            if max_samples and sample_count >= max_samples:
                break
            if stage == 1:
                # Stage 1: Collect encoder outputs
                gas = batch["gas"].to(device)
                fg = batch["fg"].to(device)
                
                tokens = model.tokenizer(gas)["tokens"]
                memory = model.encoder(tokens)  # [B, T, 512]
                
                # Pool over sequence
                activations = memory.mean(dim=1)  # [B, 512]
                
                all_activations.append(activations.cpu())
                all_fg_labels.append(fg.cpu())
                sample_count += gas.shape[0]
                
            else:
                # Stage 2: Collect slot embeddings
                mixture = batch["mixture"].to(device)
                
                outputs = model(mixture=mixture, stage=2)
                slots = outputs["slots"]  # [B, K, 512]
                
                B, K, d = slots.shape
                slots_flat = slots.reshape(B * K, d)
                
                all_activations.append(slots_flat.cpu())
                
                # Slot indices
                slot_idx = torch.arange(K).repeat(B)
                all_slot_indices.append(slot_idx.cpu())
                sample_count += mixture.shape[0]

    data = {
        "activations": torch.cat(all_activations, dim=0),
    }
    
    if stage == 1:
        data["fg_labels"] = torch.cat(all_fg_labels, dim=0)
    else:
        data["slot_indices"] = torch.cat(all_slot_indices, dim=0)
    
    return data


def train_sae(
    sae: BatchTopKSAE,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
) -> BatchTopKSAE:
    """Train BatchTopK SAE."""
    sae = sae.to(device)
    optimizer = Adam(sae.parameters(), lr=lr, weight_decay=0.0)
    aux_loss_fn = AuxiliaryLoss(dead_threshold=5)

    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Track feature usage for auxiliary loss
    feature_usage = torch.zeros(sae.d_latent, device=device)

    best_val_loss = float("inf")
    best_sae_state = None

    for epoch in range(epochs):
        # Reset feature usage each epoch
        feature_usage.zero_()

        # Training
        sae.train()
        train_losses = []

        for (batch_activations,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_activations = batch_activations.to(device)

            optimizer.zero_grad()

            # Forward pass (returns activations z in stats to avoid double forward)
            x_recon, stats = sae(batch_activations, return_stats=True)
            z = stats["z"]

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, batch_activations)

            # Update feature usage
            feature_usage += (z > 0).sum(dim=0).float()

            # Auxiliary loss for dead features
            residual = batch_activations - x_recon
            aux_loss = aux_loss_fn(residual, sae.encoder, feature_usage)

            # Total loss
            total_loss = recon_loss + 0.01 * aux_loss

            total_loss.backward()
            optimizer.step()

            # Normalize decoder after weight update (not inside forward to avoid in-place conflicts)
            if sae.step_count % 100 == 0:
                sae._normalize_decoder()

            train_losses.append(recon_loss.item())

        # Validation
        sae.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_activations,) in val_loader:
                batch_activations = batch_activations.to(device)
                x_recon = sae(batch_activations)
                val_loss = F.mse_loss(x_recon, batch_activations)
                val_losses.append(val_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_sae_state = sae.state_dict().copy()

    # Load best model
    if best_sae_state is not None:
        sae.load_state_dict(best_sae_state)

    return sae


def main():
    parser = argparse.ArgumentParser(description="Train SAE for interpretability")
    parser.add_argument("--config", type=str, default=None, help="Path to training config JSON file")
    parser.add_argument("--unmixer-checkpoint", type=str, required=True, help="Path to trained unmixer checkpoint")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True, help="Stage (1 or 2)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--d-latent", type=int, default=None, help="Latent dimension (auto if not specified)")
    parser.add_argument("--sparsity-percent", type=float, default=0.10, help="Sparsity percentage")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for SAE training")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset file (for stage 2)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config:
        import json
        with open(args.config) as f:
            cfg = load_config(**json.load(f))
    else:
        cfg = load_config()

    # Determine latent dimension
    if args.d_latent is None:
        d_latent = 2048 if args.stage == 1 else 1024
    else:
        d_latent = args.d_latent

    # Load unmixer model
    print(f"[Load] Loading unmixer from {args.unmixer_checkpoint}")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
        checkpoint = torch.load(args.unmixer_checkpoint, map_location=device, weights_only=False)
    checkpoint_state = checkpoint.get("model", checkpoint)

    # Auto-detect max_components from checkpoint via routing projection
    for key in checkpoint_state:
        if "routing.proj.3.weight" in key:
            out_features = checkpoint_state[key].shape[0]
            d_model = checkpoint_state[key].shape[1] // 2
            if d_model > 0:
                ckpt_max_components = out_features // d_model
                if ckpt_max_components != cfg.model.max_components:
                    print(f"[Load] Detected {ckpt_max_components} slots in checkpoint (config has {cfg.model.max_components}), adjusting")
                    cfg.model.max_components = ckpt_max_components
            break

    unmixer = SlotUnmixer(cfg.model).to(device)
    unmixer.load_state_dict(checkpoint_state, strict=False)
    unmixer.eval()

    # Collect activations
    print(f"[Collect] Collecting activations from Stage {args.stage}")
    if args.stage == 1:
        dataset = PureSpectrumDataset(cfg.data, n_functional_groups=cfg.model.functional_groups)
        train_idx, val_idx, _ = split_indices(len(dataset), seed=42)
        train_dataset = PureSpectrumDataset(cfg.data, train_idx, n_functional_groups=cfg.model.functional_groups)
        val_dataset = PureSpectrumDataset(cfg.data, val_idx, n_functional_groups=cfg.model.functional_groups)
    else:
        dataset_file = args.dataset or cfg.data.mixture_dataset
        base_dataset = MixtureDataset(cfg.data, dataset_file=dataset_file)
        train_idx, val_idx, _ = split_indices(len(base_dataset), seed=42)
        train_dataset = MixtureDataset(cfg.data, dataset_file=dataset_file, split_indices=train_idx)
        val_dataset = MixtureDataset(cfg.data, dataset_file=dataset_file, split_indices=val_idx)

    train_loader = build_dataloader(train_dataset, cfg, shuffle=False)
    val_loader = build_dataloader(val_dataset, cfg, shuffle=False)

    train_activations = collect_activations(unmixer, train_loader, device, stage=args.stage)
    val_activations = collect_activations(unmixer, val_loader, device, stage=args.stage)

    # Create SAE
    sae = BatchTopKSAE(
        d_input=512,
        d_latent=d_latent,
        sparsity_percent=args.sparsity_percent,
    )

    # Train SAE
    print(f"[Train] Training SAE (d_latent={d_latent}, sparsity={args.sparsity_percent})")
    trained_sae = train_sae(
        sae,
        train_activations["activations"],
        val_activations["activations"],
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save SAE
    sae_path = output_dir / "sae_model.pt"
    torch.save(trained_sae.state_dict(), sae_path)
    print(f"[Save] Saved SAE to {sae_path}")

    # Save metadata
    metadata = {
        "stage": args.stage,
        "d_input": 512,
        "d_latent": d_latent,
        "sparsity_percent": args.sparsity_percent,
        "n_train_samples": len(train_activations["activations"]),
        "n_val_samples": len(val_activations["activations"]),
    }
    import json
    with open(output_dir / "sae_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Complete] SAE training finished. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

