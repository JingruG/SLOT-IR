#!/usr/bin/env python
"""Score every SLOT-IR checkpoint with the baselines' evaluation code.

Usage (from slot_ir/):
    python -m analysis.rescore_with_baseline_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "slot_ir"))

import evaluate_baselines as eb  # noqa: E402
from src.config import DataConfig, load_config  # noqa: E402
from src.data import kfold_split_indices, split_indices  # noqa: E402
from src.evaluate import _load_model  # noqa: E402
from src.matching import LiquidIRDatabase  # noqa: E402

SLOT = ROOT / "slot_ir"
SEEDS = [0, 1, 2, 3, 4]


def experiments():
    """(name, config, [(run_id, checkpoint)], [dataset keys])"""
    E = []
    for task, cfg, ds in [("binary", "stage2_binary.json", ["binary_test", "drug_all"]),
                          ("ternary", "stage2_ternary.json", ["ternary_test"])]:
        E.append((f"main_{task}", f"configs/{cfg}",
                  [(f"seed{s}", f"results/main/{task}/seed_{s}/checkpoints/stage2_best.pt") for s in SEEDS], ds))
    for name, cfg in [("no_s1_tuned", "no_s1_tuned.json"), ("no_s1_tuned_p20", "no_s1_tuned_p20.json")]:
        E.append((name, f"configs/ablation/{cfg}",
                  [(f"seed{s}", f"results/{name}/no_s1/seed_{s}/checkpoints/stage2_best.pt") for s in SEEDS],
                  ["binary_test", "drug_all"]))
    E.append(("plus_drug", "configs/stage2_binary.json",
              [(f"seed{s}", f"results/ablation/plus_drug/seed_{s}/checkpoints/stage2_best.pt") for s in SEEDS],
              ["binary_test", "drug_test"]))
    E.append(("plus_synthetic", "configs/stage2_binary.json",
              [(f"seed{s}", f"results/ablation/plus_synthetic/seed_{s}/checkpoints/stage2_best.pt") for s in SEEDS],
              ["binary_test", "drug_all"]))
    for task, cfg in [("binary", "stage2_binary.json"), ("ternary", "stage2_ternary.json")]:
        E.append((f"cv_{task}", f"configs/{cfg}",
                  [(f"fold{f}", f"results/cross_val_{task}/fold_{f}/checkpoints/stage2_best.pt") for f in range(5)],
                  [f"cv_{task}"]))
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true", help="Only print the summary of saved scores")
    args = ap.parse_args()
    out_path = SLOT / "results" / "rescore.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else {}

    if not args.summary_only:
        data_dir = ROOT / "data" / "processed_unmix"
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sf = eb.compute_global_scale_factor(str(data_dir))
        with h5py.File(data_dir / "combined_molecule_database.h5") as f:
            liq = np.array(f["liq_spectra"][:], dtype=np.float32)
            smiles = [s.decode() if isinstance(s, bytes) else str(s) for s in f["smiles"][:]]
        cats = eb.build_molecule_categories(data_dir, seed=42)
        db = LiquidIRDatabase(DataConfig(processed_unmix_dir=data_dir), apply_normalization=True,
                              device=str(dev), database_file="combined_molecule_database.h5")

        def n_of(fn):
            with h5py.File(data_dir / fn) as f:
                return f["mixture_spectra"].shape[0]
        n_drug, n_bin, n_ter = n_of("mix_drugs_ml_instances.h5"), n_of("mix_ml_instances.h5"), n_of("mix_ternary_ml_instances.h5")
        cache = {}

        def dataset(key, run_id):
            if key == "binary_test":
                return eb.load_binary_test(data_dir, liq, sf, seed=42)
            if key == "ternary_test":
                return eb.load_ternary_test(data_dir, liq, sf, seed=42)
            if key == "drug_all":
                return eb.load_drug_all(data_dir, liq, sf)
            if key == "drug_test":  # as run_ablations.py / training.py: drug split uses the MODEL seed
                _, _, idx = split_indices(n_drug, seed=int(run_id.replace("seed", "")))
                return eb._load_dataset_raw(data_dir, "mix_drugs_ml_instances.h5", liq, sf, np.array(idx))
            if key.startswith("cv_"):  # as run_cross_val.py: test fold = folds[fold]
                fn, n = ("mix_ml_instances.h5", n_bin) if key == "cv_binary" else ("mix_ternary_ml_instances.h5", n_ter)
                folds = kfold_split_indices(n, k=5, seed=42)
                return eb._load_dataset_raw(data_dir, fn, liq, sf, np.array(folds[int(run_id[-1])]))
            raise KeyError(key)

        for name, cfg_path, runs, keys in experiments():
            cfg = load_config(**json.load(open(SLOT / cfg_path)))
            for run_id, ck in runs:
                for key in keys:
                    tag = f"{name}|{run_id}|{key}"
                    if tag in done:
                        continue
                    ck_path = SLOT / ck
                    if not ck_path.exists():
                        print(f"[skip] missing {ck_path}")
                        continue
                    model = _load_model(ck_path, cfg, dev).eval()

                    def pf(x):
                        with torch.no_grad():
                            return model(mixture=torch.from_numpy(x).float().to(dev), stage=2)["components"].cpu().numpy()

                    ck_key = (key, run_id if (key.startswith("cv_") or key == "drug_test") else "")
                    if ck_key not in cache:
                        cache[ck_key] = dataset(key, run_id)
                    r = eb.evaluate_on_data(pf, cache[ck_key], cats, 64, database=db, smiles_list=smiles)
                    m = r["matching"]
                    done[tag] = {"n": r["n_samples"], "cos": r["unmixing"]["component_cosine"]["mean"],
                                 "r1": m["component_top1_accuracy"], "r5": m["component_top5_accuracy"],
                                 "r10": m["component_top10_accuracy"], "mix": m["mixture_top1_accuracy"]}
                    out_path.write_text(json.dumps(done, indent=1))
                    print(tag, json.dumps({k: round(v, 4) for k, v in done[tag].items()}), flush=True)
                    del model
                    torch.cuda.empty_cache()

    # Summary: mean and population std over seeds / folds
    groups = {}
    for tag, v in done.items():
        name, _, key = tag.split("|")
        groups.setdefault((name, key), []).append(v)
    print(f"\n{'experiment':22s} {'dataset':13s} n  " + "  ".join(f"{k:>12s}" for k in ["cos", "R@1", "R@5", "R@10", "Mix"]))
    for (name, key), L in sorted(groups.items()):
        cells = []
        for k in ["cos", "r1", "r5", "r10", "mix"]:
            a = np.array([x[k] for x in L]) * (1 if k == "cos" else 100)
            cells.append(f"{a.mean():6.3f}±{a.std():.3f}" if k == "cos" else f"{a.mean():6.1f}±{a.std():4.1f}")
        print(f"{name:22s} {key:13s} {len(L)}  " + "  ".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    main()
