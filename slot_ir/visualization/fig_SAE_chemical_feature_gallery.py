#!/usr/bin/env python3
"""Figure 2: SAE chemical feature gallery (activation-conditioned spectral differences and top-activating examples).

Usage (from slot_ir/):
    python -m visualization.fig_SAE_chemical_feature_gallery --plot-a OH,C-F --plot-b OH,C-F --features OH=1511,C-F=1752
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.data import MixtureDataset, split_indices, _load_mixture_dataset
from src.models import SlotUnmixer
from src.sae import BatchTopKSAE
from src.losses import match_slots_l2

# ── Style ────────────────────────────────────────────────────────────────────
FONT_SIZE = 8
TITLE_SIZE = 10
LABEL_SIZE = 7
PANEL_LABEL_SIZE = 12

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": LABEL_SIZE,
    "ytick.labelsize": LABEL_SIZE,
    "legend.fontsize": LABEL_SIZE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    # Math text (e.g. cm$^{-1}$) in the same font as the labels
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial",
    "mathtext.sf": "Arial",
    # Embed fonts as TrueType (Type 42) instead of Type 3
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

WN = np.linspace(400, 4000, 1250)       # low to high (data order)
WN_PLOT = np.linspace(4000, 400, 1250)   # high to low (plot order)

# Dashed vertical reference lines (common IR regions)
VLINE_POSITIONS = [3500, 3000, 2500, 2200, 1700, 1500, 1200, 800]

# SMARTS for FG substructure highlighting (Plot B molecules)
FG_SMARTS = {
    "NH":   "[NX3H]",
    "OH":   "[OX2H]",
    "CHO":  "[CX3H1](=O)",
    "COO":  "[CX3](=O)[OX2]",
    "Ar-O": "[c][OX2]",
    "N=N":  "[NX2]=[NX2]",
    "SO2":  "[SX4](=O)(=O)",
    "SH":   "[SX2H]",
    "COOH": "[CX3](=O)[OX2H]",
    "C#N":  "[CX2]#[NX1]",
    "Ar":   "[c]",
}

# Color palette for FGs (cycled if more FGs than colors)
FG_PALETTE = [
    "#E24A33", "#348ABD", "#FF9F00", "#988ED5", "#2CA02C",
    "#D4527B", "#56B4E9", "#E69F17", "#7B68EE", "#CC6677",
    "#44AA99", "#AA4499", "#882255", "#332288", "#117733",
    "#DDCC77", "#88CCEE", "#CC6600", "#6699CC", "#661100",
]


# ── IR band loading ──────────────────────────────────────────────────────────

def load_ir_bands() -> tuple:
    """Load IR bands from configs/ir_bands.json."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ir_bands.json"
    with open(config_path) as f:
        data = json.load(f)
    known_bands = {k: tuple(v) for k, v in data["known_ir_bands"].items()}
    fg_to_bands = {}
    for fg, band_names in data["fg_to_bands"].items():
        if fg.startswith("_"):
            continue
        bands = []
        for bn in band_names:
            if bn in known_bands:
                bands.append((bn, known_bands[bn]))
        fg_to_bands[fg] = bands
    return known_bands, fg_to_bands


# ── Model loading ────────────────────────────────────────────────────────────

def load_models(config_path, checkpoint_path, sae_path, device):
    """Load frozen SlotUnmixer and SAE."""
    with open(config_path) as f:
        cfg = json.load(f)
    config = load_config(**cfg)
    model = SlotUnmixer(config.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    sae_ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae_state = sae_ckpt.get("model_state_dict", sae_ckpt)
    d_latent = sae_state["encoder.weight"].shape[0]
    d_model = config.model.d_model
    sae = BatchTopKSAE(d_model, d_latent, sparsity_percent=0.05)
    sae.load_state_dict(sae_state)
    sae.to(device).eval()
    return model, sae, config


def load_all_results(results_dir: Path) -> dict:
    """Load results.json, SLI, and PR data."""
    data = {}
    with open(results_dir / "results.json") as f:
        data["main"] = json.load(f)

    sli_path = results_dir / "spectral_localization.json"
    if sli_path.exists():
        with open(sli_path) as f:
            data["sli"] = json.load(f)

    pr_path = results_dir / "precision_recall_per_fg.json"
    if pr_path.exists():
        with open(pr_path) as f:
            data["pr"] = json.load(f)

    return data


# ── Data collection helpers ──────────────────────────────────────────────────

def collect_per_slot_data(model, sae, dataset, device):
    """Collect per-slot activations and matched component spectra."""
    from scipy.optimize import linear_sum_assignment

    model.eval()
    sae.eval()
    all_activations = []
    all_comp_spectra = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0
    )
    with torch.no_grad():
        for batch in loader:
            mixture = batch["mixture"].to(device)
            components = batch["components"].to(device)  # [B, K, 1250]
            B = mixture.shape[0]

            out = model(mixture=mixture, stage=2)
            slots = out["slots"]           # [B, K, d_model]
            pred = out["components"]       # [B, K, 1250]
            K = slots.shape[1]

            # Match each slot to a true component (L2 Hungarian, as in training)
            matched = match_slots_l2(pred, components)
            for b in range(B):
                row_ind, col_ind = range(K), matched[b]

                slots_flat = slots[b]  # [K, d_model]
                acts = sae.get_activations(slots_flat)  # [K, d_latent]

                for r, c in zip(row_ind, col_ind):
                    all_activations.append(acts[r].cpu().numpy())
                    all_comp_spectra.append(components[b, c].cpu().numpy())

    return np.array(all_activations), np.array(all_comp_spectra)


def compute_activation_conditioned_spectrum(
    activations: np.ndarray,
    component_spectra: np.ndarray,
    feature_idx: int,
    top_percentile: float = 90.0,
) -> np.ndarray:
    """Activation-conditioned spectral difference for one SAE feature."""
    feat_acts = activations[:, feature_idx]  # [N]

    # Inactive: feature activation is zero
    inactive_mask = feat_acts == 0.0
    # Active: above top_percentile of nonzero activations
    nonzero = feat_acts[feat_acts > 0]
    if len(nonzero) < 5:
        # Too few activations; return zeros
        return np.zeros(component_spectra.shape[1])
    threshold = np.percentile(nonzero, top_percentile)
    active_mask = feat_acts >= threshold

    mean_active = component_spectra[active_mask].mean(axis=0)
    if inactive_mask.sum() > 0:
        mean_inactive = component_spectra[inactive_mask].mean(axis=0)
    else:
        # Fallback: use below-median nonzero as "inactive"
        median_val = np.median(nonzero)
        low_mask = (feat_acts > 0) & (feat_acts <= median_val)
        mean_inactive = component_spectra[low_mask].mean(axis=0)

    return mean_active - mean_inactive


def find_top_activating_samples(model, sae, dataset, feature_idx, device,
                                top_n=5, max_samples=2000):
    """Find top-N samples that maximally activate a specific feature."""
    import heapq
    heap = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    sample_offset = 0
    model.eval(); sae.eval()
    with torch.no_grad():
        for batch in loader:
            mixture = batch["mixture"].to(device)
            B = mixture.shape[0]
            out = model(mixture=mixture, stage=2)
            slots = out["slots"]
            K = slots.shape[1]
            slots_flat = slots.reshape(B * K, -1)
            acts = sae.get_activations(slots_flat).reshape(B, K, -1)
            for b in range(B):
                for k in range(K):
                    a = acts[b, k, feature_idx].item()
                    if a > 0:
                        entry = (a, sample_offset + b, k)
                        if len(heap) < top_n:
                            heapq.heappush(heap, entry)
                        elif a > heap[0][0]:
                            heapq.heapreplace(heap, entry)
            sample_offset += B
            if sample_offset >= max_samples:
                break
    results = sorted(heap, key=lambda x: -x[0])
    samples = []
    for act_val, sidx, kidx in results:
        sample = dataset[sidx]
        samples.append({
            "sample_idx": sidx,
            "slot_idx": kidx,
            "activation": act_val,
            "mixture": sample["mixture"].numpy(),
            "smiles": list(sample.get("smiles", [])),
        })
    return samples


def render_mol_with_fg(smiles: str, fg_smarts: str = None, size=(200, 160)):
    """Render molecule with optional FG substructure highlighting."""
    if not HAS_RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    highlight_atoms, highlight_bonds = [], []
    if fg_smarts:
        patt = Chem.MolFromSmarts(fg_smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            if matches:
                highlight_atoms = list(matches[0])
                for bond in mol.GetBonds():
                    if (bond.GetBeginAtomIdx() in highlight_atoms and
                            bond.GetEndAtomIdx() in highlight_atoms):
                        highlight_bonds.append(bond.GetIdx())
    return Draw.MolToImage(
        mol, size=size,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plot A: Activation-conditioned spectral difference for ALL FGs
# ══════════════════════════════════════════════════════════════════════════════

def make_plot_a(fg_data, output_dir: Path):
    """Each row: one FG's activation-conditioned spectral difference
    (mean component spectrum when feature is highly active minus when inactive)
    """
    n_rows = len(fg_data)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(5.5, 0.55 * n_rows + 0.6),
        sharex=True, gridspec_kw={"hspace": 0.08}
    )
    if n_rows == 1:
        axes = [axes]

    for i, entry in enumerate(fg_data):
        ax = axes[i]
        fg_name = entry["fg_name"]
        color = entry["color"]
        delta = entry["delta_spectrum"]
        bands = entry["bands"]  # list of (band_name, (lo, hi))
        feat_idx = entry["feature_idx"]
        corr = entry["correlation"]
        sli_ratio = entry.get("sli_ratio")
        f1 = entry.get("f1")

        # Flip to plot order (4000 -> 400)
        delta_plot = np.flip(delta)

        # Colored baseline
        ax.axhline(0, color=color, linewidth=0.8, alpha=0.5)

        # GT absorption bands (may have multiple per FG)
        for band_name, (lo, hi) in bands:
            ax.axvspan(lo, hi, alpha=0.18, color=color, zorder=0)

        # Spectrum as filled curve
        ax.fill_between(WN_PLOT, 0, delta_plot,
                        where=(delta_plot > 0), color=color, alpha=0.7)
        ax.fill_between(WN_PLOT, 0, delta_plot,
                        where=(delta_plot < 0), color=color, alpha=0.3)
        ax.plot(WN_PLOT, delta_plot, color=color, linewidth=0.7)

        # Dashed vertical reference lines
        for v in VLINE_POSITIONS:
            ax.axvline(v, color="#cccccc", linewidth=0.4, linestyle="--", zorder=0)

        # Label on left: FG name, feature idx
        ax.set_ylabel("")
        ax.set_yticks([])
        label_text = f"F#{feat_idx} ({fg_name})"
        ax.text(0.12, 0.5, label_text, transform=ax.transAxes,
                fontsize=FONT_SIZE, fontweight="bold", color=color,
                ha="right", va="center")

        # Metrics annotation on right
        parts = [f"r={corr:.2f}"]
        if f1 is not None:
            parts.append(f"F1={f1:.2f}")
        metric_text = "  ".join(parts)
        ax.text(0.99, 0.85, metric_text, transform=ax.transAxes,
                fontsize=9, ha="right", va="top", color="#555555",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        ax.set_xlim(4000, 400)
        ymax = max(np.max(np.abs(delta_plot)) * 1.3, 0.01)
        ax.set_ylim(-ymax * 0.15, ymax)

    axes[-1].set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=FONT_SIZE, ha="right", x=1.0)

    # axes[0].text(-0.02, 1.15, "(a)", transform=axes[0].transAxes,
    #              fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top")

    out = output_dir / "fig_SAE_gallery_feature.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out} / .png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Plot B: Top example spectra + molecules (selected FGs only)
# ══════════════════════════════════════════════════════════════════════════════

def make_plot_b(fg_data, output_dir: Path):
    """Each row: one FG with overlaid top-activating example spectra."""
    n_rows = len(fg_data)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(5.5, 0.55 * n_rows + 0.6),
        sharex=True, gridspec_kw={"hspace": 0.08}
    )
    if n_rows == 1:
        axes = [axes]

    for i, entry in enumerate(fg_data):
        ax = axes[i]
        fg_name = entry["fg_name"]
        color = entry["color"]
        bands = entry["bands"]
        top_samples = entry["top_samples"]

        ax.axhline(0, color=color, linewidth=0.8, alpha=0.4)
        for _, (lo, hi) in bands:
            ax.axvspan(lo, hi, alpha=0.15, color=color, zorder=0)

        for j, s in enumerate(top_samples):
            mix = np.flip(s["mixture"])
            alpha = max(0.9 - j * 0.12, 0.15)
            ax.plot(WN_PLOT, mix, color=color, linewidth=0.6, alpha=alpha)

        for v in VLINE_POSITIONS:
            ax.axvline(v, color="#cccccc", linewidth=0.4, linestyle="--", zorder=0)

        ax.set_xlim(4000, 400)
        ax.set_yticks([])
        ax.set_ylabel("")

        label_text = f"{fg_name}"
        ax.text(0.08, 0.5, label_text, transform=ax.transAxes,
                fontsize=FONT_SIZE, fontweight="bold", color=color,
                ha="right", va="center")

    axes[-1].set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=FONT_SIZE,
                        ha="right", x=1.0)

    out = output_dir / "fig_SAE_gallery_example.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out} / .png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main

def feature_f1(results_dir: Path, fg_names: list, feature_idx: int, fg_name: str):
    """Best F1 of one feature for one FG over the saved activation thresholds."""
    path = results_dir / "precision_recall.npz"
    if not path.exists() or fg_name not in fg_names:
        return None
    pr = np.load(path)
    g = fg_names.index(fg_name)
    vals = [pr[k][feature_idx, g] for k in pr.files if k.endswith("_f1")]
    return float(np.nanmax(vals)) if vals else None


def score_difference_spectrum(delta: np.ndarray, bands: list) -> dict:
    """How well an activation-conditioned difference spectrum matches its FG bands."""
    pos = np.clip(delta, 0, None) ** 2
    neg = np.clip(delta, None, 0) ** 2
    total = pos.sum() + neg.sum() + 1e-20
    mask = np.zeros_like(WN, dtype=bool)
    width = 0.0
    for _, (lo, hi) in bands:
        mask |= (WN >= lo) & (WN <= hi)
        width += hi - lo
    chance = width / (WN[-1] - WN[0]) if width > 0 else float("nan")
    in_band = pos[mask].sum() / (pos.sum() + 1e-20) if width > 0 else float("nan")
    peak_wn = float(WN[int(np.argmax(delta))])
    peak_in_band = bool(mask[int(np.argmax(delta))]) if width > 0 else False
    return {
        "in_band_fraction": float(in_band),
        "chance": float(chance),
        "in_band_ratio": float(in_band / chance) if width > 0 else float("nan"),
        "peak_wn": peak_wn,
        "peak_in_band": peak_in_band,
        "negative_fraction": float(neg.sum() / total),
    }

# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load IR bands from config
    _, fg_to_bands = load_ir_bands()

    # Load all analysis results
    all_results = load_all_results(results_dir)
    top_per_fg = all_results["main"]["feature_fg_correlation"]["top_features_per_fg"]

    # Build SLI lookup: feature_idx -> {fg, sli_ratio}
    sli_lookup = {}
    if "sli" in all_results:
        for feat in all_results["sli"]["per_feature"]:
            key = (feat["feature_idx"], feat["fg"])
            sli_lookup[key] = feat.get("sli_ratio", float("nan"))

    # Build PR lookup: fg_name -> best_f1
    pr_lookup = {}
    if "pr" in all_results:
        for item in all_results["pr"]:
            pr_lookup[item["fg_name"]] = item.get("best_f1", item.get("max_f1", 0))

    # Collect all FGs with nonzero top correlation
    all_fg_entries = []
    for fg_name, feats in top_per_fg.items():
        if not feats:
            continue
        f0 = feats[0]
        corr = f0["correlation"]
        if abs(corr) < 0.05:
            continue  # skip FGs with near-zero correlation
        feature_idx = f0["feature_idx"]
        bands = fg_to_bands.get(fg_name, [])
        sli_ratio = sli_lookup.get((feature_idx, fg_name))
        f1 = pr_lookup.get(fg_name)
        if args.features and fg_name in args.features:
            feature_idx = args.features[fg_name]
            corr = next((f["correlation"] for f in feats if f["feature_idx"] == feature_idx), corr)
            sli_ratio = sli_lookup.get((feature_idx, fg_name))
        fg_list = all_results["main"].get("fg_names", [])
        f1_feat = feature_f1(results_dir, fg_list, feature_idx, fg_name)
        if f1_feat is not None:
            f1 = f1_feat
        all_fg_entries.append({
            "fg_name": fg_name,
            "feature_idx": feature_idx,
            "correlation": corr,
            "bands": bands,
            "sli_ratio": sli_ratio,
            "f1": f1,
        })

    # Sort by SLI ratio (best first), then by correlation for those without SLI
    def sort_key(e):
        sli = e["sli_ratio"]
        if sli is not None and not np.isnan(sli):
            return (0, -sli)  # has SLI: sort by SLI descending
        return (1, -abs(e["correlation"]))  # no SLI: sort by correlation
    all_fg_entries.sort(key=sort_key)

    # Assign colors
    for i, entry in enumerate(all_fg_entries):
        entry["color"] = FG_PALETTE[i % len(FG_PALETTE)]

    print(f"Found {len(all_fg_entries)} FGs with |r| >= 0.05")
    for entry in all_fg_entries:
        sli_str = f"SLI={entry['sli_ratio']:.1f}x" if entry['sli_ratio'] is not None and not np.isnan(entry['sli_ratio']) else "no SLI"
        f1_str = f"F1={entry['f1']:.3f}" if entry['f1'] is not None else "no PR"
        n_bands = len(entry['bands'])
        print(f"  {entry['fg_name']:12s}: F#{entry['feature_idx']:4d}  r={entry['correlation']:+.3f}  {sli_str:12s}  {f1_str:10s}  bands={n_bands}")

    # Load models
    print("\nLoading models...")
    model, sae, config = load_models(args.config, args.checkpoint,
                                      args.sae_checkpoint, device)

    print("Loading dataset (test split only)...")
    mixture_data = _load_mixture_dataset(config.data)
    n_samples = mixture_data['mixture'].shape[0]
    _, _, test_indices = split_indices(n_samples, seed=config.seed)
    dataset = MixtureDataset(config.data, split_indices=test_indices)
    print(f"  Test split: {len(test_indices)} samples ({n_samples} total)")

    # Collect per-slot SAE activations and matched component spectra
    print("Collecting per-slot activations and component spectra...")
    activations, component_spectra = collect_per_slot_data(model, sae, dataset, device)
    print(f"  Collected {activations.shape[0]} slot embeddings, "
          f"activations shape {activations.shape}, spectra shape {component_spectra.shape}")

    if args.rank_candidates:
        fg_list = all_results["main"].get("fg_names", [])
        rows = []
        for fg_name, feats in top_per_fg.items():
            bands = fg_to_bands.get(fg_name, [])
            if not bands:
                continue
            for f in feats[:3]:
                if abs(f["correlation"]) < 0.2:
                    continue
                delta = compute_activation_conditioned_spectrum(
                    activations, component_spectra, f["feature_idx"])
                sc = score_difference_spectrum(delta, bands)
                rows.append({"fg": fg_name, "feature_idx": int(f["feature_idx"]),
                             "correlation": float(f["correlation"]),
                             "f1": feature_f1(results_dir, fg_list, f["feature_idx"], fg_name), **sc})
        rows.sort(key=lambda r: -r["in_band_ratio"])
        out = output_dir / "gallery_candidates.json"
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\n{'FG':12s} {'feat':>5s} {'r':>5s} {'F1':>5s} {'in-band x chance':>17s} {'peak cm-1':>9s} {'peak in band':>12s} {'neg share':>9s}")
        for r in rows:
            f1s = f"{r['f1']:.2f}" if r["f1"] is not None else "  -  "
            print(f"{r['fg']:12s} {r['feature_idx']:5d} {r['correlation']:5.2f} {f1s:>5s} {r['in_band_ratio']:17.1f} {r['peak_wn']:9.0f} {str(r['peak_in_band']):>12s} {r['negative_fraction']:9.2f}")
        print(f"\nSaved {out}")
        return

    # Compute activation-conditioned spectral difference for all FGs
    print(f"\nComputing activation-conditioned spectra for {len(all_fg_entries)} FGs...")
    for entry in all_fg_entries:
        entry["delta_spectrum"] = compute_activation_conditioned_spectrum(
            activations, component_spectra, entry["feature_idx"]
        )

    # Optional fixed colors, given in the same order as --plot-a (or --plot-b)
    if args.colors:
        color_list = [c.strip() for c in args.colors.split(",")]
        order_src = args.plot_a or args.plot_b
        if order_src:
            order_fgs = [fg.strip() for fg in order_src.split(",")]
            color_map = dict(zip(order_fgs, color_list))
            for entry in all_fg_entries:
                if entry["fg_name"] in color_map:
                    entry["color"] = color_map[entry["fg_name"]]

    # Generate Plot A (selected or all FGs)
    if args.plot_a:
        plot_a_fgs = [fg.strip() for fg in args.plot_a.split(",")]
        plot_a_data = [e for e in all_fg_entries if e["fg_name"] in plot_a_fgs]
        # Preserve user-specified order
        fg_order = {fg: i for i, fg in enumerate(plot_a_fgs)}
        plot_a_data.sort(key=lambda e: fg_order.get(e["fg_name"], 999))
        missing = set(plot_a_fgs) - {e["fg_name"] for e in plot_a_data}
        if missing:
            print(f"  WARNING: FGs not found for Plot A: {missing}")
    else:
        plot_a_data = all_fg_entries
    print(f"\nGenerating Plot A ({len(plot_a_data)} FGs)...")
    make_plot_a(plot_a_data, output_dir)

    # Generate Plot B (selected FGs only, if requested)
    if args.plot_b:
        plot_b_fgs = [fg.strip() for fg in args.plot_b.split(",")]
        plot_b_data = [e for e in all_fg_entries if e["fg_name"] in plot_b_fgs]
        # Preserve user-specified order
        fg_order_b = {fg: i for i, fg in enumerate(plot_b_fgs)}
        plot_b_data.sort(key=lambda e: fg_order_b.get(e["fg_name"], 999))
        missing = set(plot_b_fgs) - {e["fg_name"] for e in plot_b_data}
        if missing:
            print(f"  WARNING: FGs not found for Plot B: {missing}")

        print(f"\nFinding top-activating samples for Plot B ({len(plot_b_data)} FGs)...")
        for entry in plot_b_data:
            print(f"  {entry['fg_name']}: Feature #{entry['feature_idx']}...")
            entry["top_samples"] = find_top_activating_samples(
                model, sae, dataset, entry["feature_idx"], device, top_n=10
            )

        print(f"Generating Plot B...")
        make_plot_b(plot_b_data, output_dir)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAE Chemical Feature Gallery"
    )
    parser.add_argument("--config", type=str,
                        default="configs/stage2_binary.json")
    parser.add_argument("--checkpoint", type=str,
                        default="results/main/binary/seed_0/checkpoints/stage2_best.pt")
    parser.add_argument("--sae-checkpoint", type=str,
                        default="results/sae/trained_sae.pt")
    parser.add_argument("--results-dir", type=str,
                        default="results/sae")
    parser.add_argument("--output-dir", type=str,
                        default="visualization/figures")
    parser.add_argument("--plot-a", type=str, default=None,
                        help="Comma-separated FG names for Plot A (e.g. OH,C-F). Default: all FGs.")
    parser.add_argument("--plot-b", type=str, default=None,
                        help="Comma-separated FG names for Plot B (e.g. OH,C-F)")
    parser.add_argument("--rank-candidates", action="store_true",
                        help="Score the top-3 features of every FG against its IR bands, save gallery_candidates.json, and exit")
    parser.add_argument("--features", type=str, default=None,
                        help="Override the plotted feature per FG, e.g. 'OH=1511,C-F=1752' (use the last '=' to split)")
    parser.add_argument("--colors", type=str, default=None,
                        help="Comma-separated hex colors in the order of --plot-a (e.g. '#E24A33,#348ABD,#988ED5')")
    args = parser.parse_args()
    if args.features:
        args.features = {item.rsplit("=", 1)[0].strip(): int(item.rsplit("=", 1)[1])
                         for item in args.features.split(",")}
    main(args)
