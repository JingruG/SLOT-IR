"""Check whether the top-activating test slots of an SAE feature come from molecules that contain its group.

Usage (from slot_ir/):
    python analysis/sae_group_presence.py
"""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from src.data import MixtureDataset, split_indices
from visualization.fig_SAE_chemical_feature_gallery import load_models
dev = 'cuda'
model, sae, cfg = load_models('configs/stage2_binary.json', 'results/main/binary/seed_0/checkpoints/stage2_best.pt',
                              'results/sae/trained_sae.pt', dev)
base = MixtureDataset(cfg.data)
_, _, te = split_indices(len(base), seed=42)
ds = MixtureDataset(cfg.data, split_indices=te)
acts, smi = [], []
with torch.no_grad():
    for i in range(0, len(ds), 64):
        items = [ds[j] for j in range(i, min(i + 64, len(ds)))]
        mix = torch.stack([it['mixture'] for it in items]).to(dev)
        comp = torch.stack([it['components'] for it in items]).to(dev)
        out = model(mixture=mix, stage=2)
        a = sae.get_activations(out['slots'].reshape(-1, out['slots'].shape[-1])).reshape(len(items), -1, sae.d_latent if hasattr(sae,'d_latent') else 2048)
        for b, it in enumerate(items):
            K = comp.shape[1]
            cost = torch.cdist(out['components'][b, :K], comp[b]).cpu().numpy()
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                acts.append(a[b, ri].cpu().numpy()); smi.append(list(it['smiles'])[ci])
acts = np.array(acts); print('slots', len(acts))
pat = {'C-F': Chem.MolFromSmarts('[#6]F'), 'OH': Chem.MolFromSmarts('[OX2H]'), 'COO': Chem.MolFromSmarts('[#6]C(=O)O[#6]'), 'CONH': Chem.MolFromSmarts('C(=O)[NX3]')}
has = {k: np.array([Chem.MolFromSmiles(s).HasSubstructMatch(p) if Chem.MolFromSmiles(s) else False for s in smi]) for k, p in pat.items()}
res = {}
for feat, g in [(1752, 'C-F'), (262, 'C-F'), (1511, 'OH'), (902, 'COO'), (160, 'COO'), (1460, 'CONH')]:
    f = acts[:, feat]; nz = f > 0
    order = np.argsort(-f)
    top20 = order[:20]; thr = np.percentile(f[nz], 90) if nz.sum() > 5 else np.inf
    top10pct = f >= thr
    row = dict(feature=feat, group=g, base_rate=float(has[g].mean()), n_active=int(nz.sum()),
               top20_with_group=int(has[g][top20].sum()), top10pct_n=int(top10pct.sum()),
               top10pct_with_group=float(has[g][top10pct].mean()), active_with_group=float(has[g][nz].mean()),
               distinct_top20=len(set(smi[i] for i in top20)), distinct_active_with_group=len(set(smi[i] for i in np.where(nz & has[g])[0])), distinct_group_mols=len(set(s for s,h in zip(smi,has[g]) if h)), top5_smiles=[smi[i] for i in order[:5]])
    res[f'{feat}_{g}'] = row; print(json.dumps(row))
json.dump(res, open('results/sae/group_presence_check.json', 'w'), indent=1)
