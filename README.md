# SLOT-IR

Code for *SLOT-IR: Learning Disentangled Slot Representations for Infrared Spectral Unmixing* (NeurIPS 2026).
SLOT-IR splits the infrared spectrum of a liquid mixture into the spectra of its components, using one slot per component.

## Install

```bash
conda create -n slot-ir python=3.11
conda activate slot-ir
pip install -r requirements.txt
```

## Data

Download the processed data from [Google Drive](https://drive.google.com/file/d/1ysMG6zzXE05lL6bx9c3McjPWkyZM_4a6/view?usp=sharing) (`SLOT-IR_data.zip`, 387 MB) and unzip it in the repository root. This creates `data/processed/` and `data/processed_unmix/`.
The spectra are from Melle et al. (2026). `data/preprocess.py` and `data/preprocess_unmix.py` show how the processed files were built from the raw simulation data.

## Reproduce

SLOT-IR commands run from `slot_ir/`. Run `run_main.py` first: the other Stage 2 runs load its Stage 1 checkpoint (`results/main/stage1/`).

| Result | Command |
|---|---|
| Table 1 and Table 2, SLOT-IR default | `python run_main.py` |
| Table 1, baselines (from `baselines/`) | `python run_fair_baselines.py <stage>` for each stage `dl-grid sunsal-grid select dl-seeds evaluate classical summary` |
| Table 2, w/o Stage 1 (grid) | `python -m src.hyperparam_search --mode no_s1` |
| Table 2, w/o Stage 1 (5 seeds) | `python run_ablations.py --config configs/ablation/no_s1_tuned.json --experiments no_s1 --results-dir results/no_s1_tuned` |
| Appendix, w/o Stage 1 with patience 20 | `python run_ablations.py --config configs/ablation/no_s1_tuned_p20.json --experiments no_s1 --results-dir results/no_s1_tuned_p20` |
| Table 2, + Drug and + Synthetic | `python run_ablations.py --experiments plus_drug plus_synthetic` |
| Table 3, Stage 1 objectives | `python run_stage1_controls.py` |
| Appendix, cross-validation | `python run_cross_val.py --config configs/stage2_binary.json --results-dir results/cross_val_binary` (same with `stage2_ternary.json` and `results/cross_val_ternary`) |
| Appendix, molecule holdout | `python run_holdout.py` |
| Appendix, hyperparameter grids | `python -m src.hyperparam_search --mode stage1` (also `binary`, `ternary`) |
| Section 5, SAE | `python -m src.interpret_sae --config configs/stage2_binary.json --checkpoint results/main/binary/seed_0/checkpoints/stage2_best.pt` |
| Figure 2 | `python -m visualization.fig_SAE_chemical_feature_gallery --plot-a OH,C-F --plot-b OH,C-F --features OH=1511,C-F=1752` |
| Copy-the-mixture reference | `python analysis/copy_mixture_reference.py` |

All SLOT-IR rows in the paper use the same scoring code as the baselines. After training, score all checkpoints with:

```bash
python -m analysis.rescore_with_baseline_eval
```

## Check numbers

`paper_numbers.py` reads the saved results (no model runs) and prints each paper number with MATCH or MISMATCH:

```bash
python slot_ir/analysis/paper_numbers.py   # from the repo root
```

## Citation

```bibtex
@inproceedings{gan2026slotir,
  title     = {{SLOT-IR}: Learning Disentangled Slot Representations for Infrared Spectral Unmixing},
  author    = {Gan, Jingru and Melle, Yannah J.U. and Zhu, Yanqiao and Schwalbe-Koda, Daniel and Wang, Wei},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see `LICENSE`.
