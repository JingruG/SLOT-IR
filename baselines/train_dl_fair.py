"""Train one deep learning baseline (CNNAEU, CyCU-Net or DeepTrans-HSU) under the shared protocol.

Usage (normally called by run_fair_baselines.py):
    python train_dl_fair.py --method cnnaeu --dataset binary --lr 1e-4 --rec-weight 1 --seed 0 --out-dir results/fair/final_seeds/seed0
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

_BASELINES_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASELINES_ROOT))
sys.path.insert(0, str(_BASELINES_ROOT.parent / "slot_ir"))

from utils.data_loader import get_unmix_dataloaders  # noqa: E402
from src.losses import hungarian_component_loss  # noqa: E402


def build_model(method: str, n_components: int):
    if method == "cnnaeu":
        from deep_learning.cnnaeu import CNNAEU
        return CNNAEU(n_components=n_components, input_dim=1250)
    if method == "cycunet":
        from deep_learning.cycunet import CyCUNet
        return CyCUNet(n_components=n_components, input_dim=1250)
    if method == "deeptrans_hsu":
        from deep_learning.deeptrans_hsu import DeepTransHSU
        return DeepTransHSU(n_components=n_components, input_dim=1250)
    raise ValueError(f"Unknown method: {method}")


def forward(model, method: str, mixture: torch.Tensor) -> dict:
    if method == "cycunet":
        return model(mixture)
    _, components, reconstruction = model(mixture)
    return {"components": components, "reconstruction": reconstruction}


def compute_loss(out: dict, mixture, target, method: str, rec_weight: float):
    comp = hungarian_component_loss(out["components"], target)
    rec = F.mse_loss(out["reconstruction"], mixture)
    loss = comp + rec_weight * rec
    if method == "cycunet":
        loss = loss + F.mse_loss(out["cycle_mixture"], mixture) \
                    + hungarian_component_loss(out["cycle_components"], target)
    return loss, comp, rec


def matched_cosine(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample mean cosine after Hungarian L2 matching (as in the evaluation)."""
    out = np.zeros(len(pred))
    for b in range(len(pred)):
        cost = np.linalg.norm(pred[b][:, None] - target[b][None], axis=-1)
        r, c = linear_sum_assignment(cost)
        p, t = pred[b][r], target[b][c]
        cos = (p * t).sum(-1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-8)
        out[b] = cos.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["cnnaeu", "cycunet", "deeptrans_hsu"])
    ap.add_argument("--dataset", required=True, choices=["binary", "ternary"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--rec-weight", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="../data/processed_unmix")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_{args.dataset}"
    model_path = out_dir / (f"{tag}_mix_ml_model.pt" if args.dataset == "binary" else f"{tag}_model.pt")
    summary_path = out_dir / f"{tag}_summary.json"
    history_path = out_dir / f"{tag}_history.jsonl"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("done"):
        print(f"Already done: {summary_path}")
        return
    # One process per job: a second runner on the same job exits instead of mixing files.
    lock_path = out_dir / f"{tag}.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print(f"Locked by another run ({lock_path}); exiting. Delete the lock only if no run is alive.")
        sys.exit(3)
    try:
        train(args, out_dir, tag, model_path, summary_path, history_path)
    finally:
        lock_path.unlink(missing_ok=True)


def train(args, out_dir, tag, model_path, summary_path, history_path):
    history_path.unlink(missing_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    n_components = 2 if args.dataset == "binary" else 3
    train_loader, val_loader, _ = get_unmix_dataloaders(
        data_dir=args.data_dir, dataset_type=args.dataset, batch_size=args.batch_size, seed=42,
    )
    model = build_model(args.method, n_components).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(len(train_loader), 1))
    config = vars(args) | {"n_params": sum(p.numel() for p in model.parameters()), "device": device}
    print(json.dumps(config, indent=2))

    best = {"val_loss": float("inf"), "epoch": -1, "val_comp_cos": None}
    wait = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_loss, n_tr = 0.0, 0
        for batch in train_loader:
            mixture = batch["mixture"].to(device)
            target = batch["components"].to(device)
            optimizer.zero_grad()
            loss, _, _ = compute_loss(forward(model, args.method, mixture), mixture, target,
                                      args.method, args.rec_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * len(mixture)
            n_tr += len(mixture)

        model.eval()
        va_loss, n_va, cos = 0.0, 0, []
        with torch.no_grad():
            for batch in val_loader:
                mixture = batch["mixture"].to(device)
                target = batch["components"].to(device)
                out = forward(model, args.method, mixture)
                loss, _, _ = compute_loss(out, mixture, target, args.method, args.rec_weight)
                va_loss += loss.item() * len(mixture)
                n_va += len(mixture)
                cos.append(matched_cosine(out["components"].cpu().numpy(), target.cpu().numpy()))
        va_loss /= n_va
        va_cos = float(np.concatenate(cos).mean())

        improved = va_loss < best["val_loss"]
        if improved:
            best = {"val_loss": va_loss, "epoch": epoch, "val_comp_cos": va_cos}
            torch.save(model.state_dict(), model_path)
            wait = 0
        else:
            wait += 1
        rec = {"epoch": epoch, "train_loss": tr_loss / n_tr, "val_loss": va_loss,
               "val_comp_cos": va_cos, "improved": improved, "elapsed_s": round(time.time() - t0, 1)}
        with open(history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        summary_path.write_text(json.dumps({"config": config, "best": best, "done": False}, indent=2))
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    summary_path.write_text(json.dumps(
        {"config": config, "best": best, "epochs_run": epoch + 1,
         "train_time_s": round(time.time() - t0, 1), "done": True}, indent=2))
    print(f"Best epoch {best['epoch']}: val loss {best['val_loss']:.6f}, val comp cos {best['val_comp_cos']:.4f}")


if __name__ == "__main__":
    main()
