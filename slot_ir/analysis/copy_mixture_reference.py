"""Copy-the-mixture reference: use the mixture spectrum as every predicted component and score it like the baselines.

Usage (from the repo root):
    python slot_ir/analysis/copy_mixture_reference.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "slot_ir"))
import evaluate_baselines as eb  # noqa: E402

d = ROOT / "data" / "processed_unmix"
sf = eb.compute_global_scale_factor(str(d))
with h5py.File(d / "combined_molecule_database.h5") as f:
    liq = np.array(f["liq_spectra"][:], dtype=np.float32)
cats = eb.build_molecule_categories(d, seed=42)
for name, data in [("binary_test", eb.load_binary_test(d, liq, sf, seed=42)), ("drug_all", eb.load_drug_all(d, liq, sf))]:
    r = eb.evaluate_on_data(lambda x: np.repeat(x[:, None, :], 2, axis=1), data, cats, 256)
    print(name, r["n_samples"], round(r["unmixing"]["component_cosine"]["mean"], 4), flush=True)
