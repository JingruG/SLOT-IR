"""Print every number in the paper tables and key text next to the saved result, with a MATCH / MISMATCH flag.

Reads only saved result files (no model is run). Mean and population std over seeds or folds.

Usage (from the repo root):
    python slot_ir/analysis/paper_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESCORE = ROOT / "slot_ir/results/rescore.json"
BASELINES = ROOT / "baselines/results/fair/summary.json"
HOLDOUT = ROOT / "slot_ir/results/holdout/aggregated/per_molecule_aggregated.json"
SAE = ROOT / "slot_ir/results/sae"

# ---------------------------------------------------------------- paper values
# Each entry: metric -> (mean, std) or mean. Cos raw, others in %.

# Table 1 (tab:main_results, experiments.tex)
PAPER_T1 = {
    "MCR-ALS": ("mcr_als_dict", dict(cos=.685, r1=41.7, r5=52.1, r10=55.7, mix=17.6), dict(r1=77.7, r5=85.7, mix=48.2)),
    "VCA+FCLS": ("vca_fcls_dict", dict(cos=.825, r1=39.6, r5=59.1, r10=66.4, mix=15.3), dict(r1=51.0, r5=61.9, mix=12.1)),
    "NNLS": ("nnls", dict(cos=.942, r1=80.2, r5=85.1, r10=87.6, mix=64.0), dict(r1=79.0, r5=83.8, mix=52.8)),
    "SUnSAL": ("sunsal_admm", dict(cos=.943, r1=80.5, r5=85.2, r10=87.7, mix=64.2), dict(r1=79.0, r5=83.9, mix=52.4)),
    "CNNAEU": ("cnnaeu", dict(cos=(.899, .002), r1=(61.0, 1.0), r5=(78.6, 0.9), r10=(83.2, 0.9), mix=(37.1, 1.4)),
               dict(r1=(57.8, 1.0), r5=(70.4, 0.6), mix=(15.7, 2.1))),
    "CyCU-Net": ("cycunet", dict(cos=(.898, .001), r1=(60.5, 0.6), r5=(77.5, 0.6), r10=(82.2, 0.5), mix=(37.7, 1.1)),
                 dict(r1=(57.5, 0.9), r5=(70.1, 0.2), mix=(15.3, 1.2))),
    "DeepTrans": ("deeptrans_hsu", dict(cos=(.925, .013), r1=(29.4, 8.4), r5=(57.0, 9.1), r10=(72.4, 8.8), mix=(6.8, 4.4)),
                  dict(r1=(43.2, 4.0), r5=(55.9, 4.9), mix=(4.1, 1.7))),
}
PAPER_T1_OURS_BIN = dict(cos=(.983, .004), r1=(81.9, 5.8), r5=(96.6, 1.9), r10=(98.5, 0.7), mix=(67.1, 9.9))
PAPER_T1_OURS_TER = dict(r1=(94.3, 2.6), r5=(98.3, 1.2), mix=(83.9, 7.0))

# Table 2 (tab:ablation, experiments.tex). (run name, drug split key)
PAPER_T2 = {
    "default": ("main_binary", "drug_all",
                dict(cos=(.983, .004), r1=(81.9, 5.8), r5=(96.6, 1.9), mix=(67.1, 9.9)),
                dict(cos=(.752, .003), r1=(1.1, 0.1), r5=(4.9, 0.5), mix=(0.0, 0.0))),
    "w/o Stage 1 (tuned)": ("no_s1_tuned", "drug_all",
                dict(cos=(.815, .203), r1=(44.7, 36.6), r5=(56.9, 45.8), mix=(32.5, 27.2)),
                dict(cos=(.629, .105), r1=(0.4, 0.3), r5=(2.3, 1.5), mix=(0.0, 0.0))),
    "+ Drug train": ("plus_drug", "drug_test",
                dict(cos=(.987, .002), r1=(86.2, 2.3), r5=(98.4, 0.5), mix=(74.2, 4.1)),
                dict(cos=(.963, .005), r1=(83.2, 3.9), r5=(93.7, 1.6), mix=(71.2, 5.5))),
    "+ Synthetic drug": ("plus_synthetic", "drug_all",
                dict(cos=(.983, .003), r1=(80.4, 3.7), r5=(96.3, 1.5), mix=(64.3, 6.2)),
                dict(cos=(.893, .007), r1=(51.9, 5.7), r5=(68.2, 4.3), mix=(28.3, 6.5))),
}

# Table 3 (tab:s1_controls, experiments.tex): results/stage1_controls/<variant>/seed_<s>/test_results.json
S1CTRL = ROOT / "slot_ir/results/stage1_controls"
PAPER_T3 = {
    "translation (default)": ("s1ctrl_baseline", dict(cos=(.987, .005), r1=(86.8, 4.4), mix=(75.4, 7.8))),
    "liquid autoencoder": ("s1ctrl_autoencoder", dict(cos=(.989, .002), r1=(88.1, 2.7), mix=(77.7, 4.9))),
    "gas autoencoder": ("s1ctrl_gas_autoencoder", dict(cos=(.985, .007), r1=(84.6, 7.5), mix=(72.0, 12.5))),
    "shuffled pairs": ("s1ctrl_shuffled", dict(cos=(.585, .015), r1=(0.0, 0.0), mix=(0.0, 0.0))),
}
PAPER_WELCH = {"s1ctrl_autoencoder": 0.63, "s1ctrl_gas_autoencoder": 0.62}  # experiments.tex sec:s1_controls

# Cross validation (tab:cross_val and tab:cross_val_detail, appendix.tex app:cross_val)
PAPER_CV = {
    "cv_binary": dict(cos=(.984, .002), r1=(84.3, 3.3), r5=(98.0, 1.0), r10=(99.4, 0.3), mix=(70.9, 5.9)),
    "cv_ternary": dict(cos=(.979, .005), r1=(94.5, 1.6), r5=(98.7, 0.6), r10=(99.1, 0.5), mix=(84.6, 4.1)),
}
PAPER_CV_FOLDS = {  # n, cos, r1, r5, mix
    "cv_binary": [(4221, .982, 79.7, 96.6, 62.8), (4221, .981, 80.9, 97.0, 64.8), (4221, .984, 86.1, 98.3, 74.1),
                  (4221, .986, 87.4, 99.0, 76.3), (4220, .987, 87.4, 99.0, 76.5)],
    "cv_ternary": [(1597, .985, 96.2, 99.4, 89.3), (1597, .981, 94.7, 99.0, 85.2), (1597, .983, 96.1, 99.1, 88.6),
                   (1597, .973, 92.9, 98.2, 81.4), (1597, .974, 92.5, 97.9, 78.8)],
}

# Holdout (tab:holdout, appendix.tex app:holdout). N, cos mean, cos std, R@1 %
PAPER_HOLDOUT = {"seen": (163, .986, .010, 86.3), "held out": (47, .865, .165, 11.0), "drug": (115, .742, .113, 1.1)}

# No Stage 1 per seed R@1 (appendix.tex app:no_s1)
PAPER_NO_S1 = {
    "no_s1_tuned": [71.0, 0.4, 81.6, 0.0, 70.6],
    "no_s1_tuned_p20": [79.7, 0.0, 92.8, 0.0, 82.3],
}
PAPER_NO_S1_P20_MEAN = (51.0, 41.8)
PAPER_NO_S1_GOOD3 = 74.4  # experiments.tex: other three seeds average

# SAE (analysis.tex and appendix.tex app:sae)
PAPER_SAE = dict(n_selective=118, n_perm=500, n_band=106, n_above=70, mean_sli=0.203, mean_chance=0.065,
                 oh_n=29, oh_above=29, oh_min=1.9, oh_max=8.2, coo=(11, 15), so2=(1, 11), cho=(0, 3),
                 excl=(44, 67), wilcoxon_p=5e-8, wilcoxon_p_excl=5e-7)
UNSURE_BANDS = {"N=N", "Ar-O", "SO3H", "P=O"}

# Copy-the-mixture numbers, printed by analysis/copy_mixture_reference.py and not saved (reported as MISSING)
PAPER_NO_SOURCE = {
    "copy mixture cos, binary test": 0.769,
    "copy mixture cos, drug set": 0.798,
    "copy mixture cos, held out molecules": 0.751,
}

# ---------------------------------------------------------------- helpers
counts = {"MATCH": 0, "MISMATCH": 0, "MISSING": 0}
mismatches = []


def fmt(v, kind):
    return f"{v:.3f}" if kind == "cos" else f"{v:.1f}"


def check(label, got, want, kind="pct"):
    """kind: 'cos' (3 decimals), 'pct' (1 decimal), 'int', or a float tolerance."""
    if got is None:
        counts["MISSING"] += 1
        print(f"  {label:<48} paper {want!s:>8}  file {'?':>8}  MISSING")
        mismatches.append((label, want, None))
        return
    if kind == "int":
        ok, g, w = int(got) == int(want), str(int(got)), str(int(want))
    elif isinstance(kind, float):
        ok, g, w = abs(got - want) <= kind, f"{got:.3g}", f"{want:.3g}"
    else:
        g, w = fmt(got, kind), fmt(want, kind)
        ok = g == w
    flag = "MATCH" if ok else "MISMATCH"
    counts[flag] += 1
    if not ok:
        mismatches.append((label, w, g))
    print(f"  {label:<48} paper {w:>8}  file {g:>8}  {flag}")


def check_ms(label, values, want, metric):
    """values: per seed raw (fractions for accuracy). want: mean or (mean, std)."""
    kind = "cos" if metric == "cos" else "pct"
    scale = 1 if metric == "cos" else 100
    arr = np.asarray(values, dtype=float) * scale
    if isinstance(want, tuple):
        check(f"{label} {metric} mean", arr.mean(), want[0], kind)
        check(f"{label} {metric} std", arr.std(ddof=0), want[1], kind)
    else:
        check(f"{label} {metric}", arr.mean(), want, kind)


def load(path):
    if not path.exists():
        print(f"  MISSING FILE: {path.relative_to(ROOT)}")
        return None
    return json.loads(path.read_text())


def per_seed(rs, run, split, metric):
    keys = sorted(k for k in rs if k.startswith(run + "|") and k.endswith("|" + split))
    return [rs[k][metric] for k in keys]


# ---------------------------------------------------------------- main
def main():
    rs = load(RESCORE)
    bl = load(BASELINES)

    print("\n== Table 1: main results (baselines)")
    if bl is not None:
        for name, (key, pb, pt) in PAPER_T1.items():
            for split, want in (("binary_test", pb), ("ternary_test", pt)):
                for m, w in want.items():
                    entry = bl.get(f"{key}:{split}", {}).get(m)
                    scale = 1 if m == "cos" else 100
                    kind = "cos" if m == "cos" else "pct"
                    if entry is None:
                        check(f"{name} {split} {m}", None, w)
                        continue
                    if isinstance(w, tuple):
                        check(f"{name} {split} {m} mean", entry[0] * scale, w[0], kind)
                        check(f"{name} {split} {m} std", entry[1] * scale, w[1], kind)
                    else:
                        check(f"{name} {split} {m}", entry[0] * scale, w, kind)

    if rs is None:
        print("Rescore json missing, cannot check SLOT-IR rows.")
        return
    print("\n== Table 1: SLOT-IR")
    for m, w in PAPER_T1_OURS_BIN.items():
        check_ms("SLOT-IR binary", per_seed(rs, "main_binary", "binary_test", m), w, m)
    for m, w in PAPER_T1_OURS_TER.items():
        check_ms("SLOT-IR ternary", per_seed(rs, "main_ternary", "ternary_test", m), w, m)

    print("\n== Table 2: ablation")
    for name, (run, drug_split, wb, wd) in PAPER_T2.items():
        for m, w in wb.items():
            check_ms(f"{name} binary", per_seed(rs, run, "binary_test", m), w, m)
        for m, w in wd.items():
            check_ms(f"{name} {drug_split}", per_seed(rs, run, drug_split, m), w, m)

    print("\n== Table 3: Stage 1 controls")
    s1ctrl_vals = {}
    for name, (run, want) in PAPER_T3.items():
        variant = run.replace("s1ctrl_", "")
        files = [S1CTRL / variant / f"seed_{s}" / "test_results.json" for s in range(5)]
        if not all(f.exists() for f in files):
            for m, w in want.items():
                check(f"{name} {m}", None, w)
            continue
        res = [json.loads(f.read_text()) for f in files]
        s1ctrl_vals[run] = {"cos": [r["unmixing"]["component_cosine"]["mean"] for r in res],
                   "r1": [r["matching"]["top1_accuracy"] for r in res],
                   "mix": [r["matching"]["mixture_match_rate"] for r in res]}
        for m, w in want.items():
            check_ms(name, s1ctrl_vals[run][m], w, m)
    try:
        from scipy.stats import ttest_ind
        base = np.array(s1ctrl_vals.get("s1ctrl_baseline", {}).get("r1", []))
        for run, want in PAPER_WELCH.items():
            other = np.array(s1ctrl_vals.get(run, {}).get("r1", []))
            p = ttest_ind(base, other, equal_var=False).pvalue
            check(f"Welch p translation vs {run} (R@1)", round(p, 2), want, 0.005)
    except ImportError:
        print("  scipy not installed, Welch test skipped")

    print("\n== Cross validation (appendix app:cross_val)")
    for run, want in PAPER_CV.items():
        for m, w in want.items():
            check_ms(run, per_seed(rs, run, run, m), w, m)
        for i, (n, cos, r1, r5, mix) in enumerate(PAPER_CV_FOLDS[run]):
            e = rs.get(f"{run}|fold{i}|{run}")
            if e is None:
                check(f"{run} fold{i}", None, r1)
                continue
            check(f"{run} fold{i} n", e["n"], n, "int")
            check(f"{run} fold{i} cos", e["cos"], cos, "cos")
            check(f"{run} fold{i} r1", e["r1"] * 100, r1)
            check(f"{run} fold{i} r5", e["r5"] * 100, r5)
            check(f"{run} fold{i} mix", e["mix"] * 100, mix)

    print("\n== No Stage 1 per seed R@1 (appendix app:no_s1)")
    for run, want in PAPER_NO_S1.items():
        vals = per_seed(rs, run, "binary_test", "r1")
        for i, w in enumerate(want):
            check(f"{run} seed{i} r1", vals[i] * 100 if i < len(vals) else None, w)
    check_ms("no_s1_tuned_p20 binary", per_seed(rs, "no_s1_tuned_p20", "binary_test", "r1"), PAPER_NO_S1_P20_MEAN, "r1")
    vals = sorted(per_seed(rs, "no_s1_tuned", "binary_test", "r1"))[2:]
    check("no_s1_tuned mean of 3 working seeds r1", np.mean(vals) * 100, PAPER_NO_S1_GOOD3)

    print("\n== Holdout (appendix app:holdout, run_holdout.py outputs)")
    ho = load(HOLDOUT)
    if ho is not None:
        # Pool each molecule over all test mixtures it appears in, weighted by count.
        groups = {"seen": {}, "held out": {}, "drug": {}}
        for cat, mols in ho.items():
            for smi, rec in mols.items():
                for co, s in rec["by_co_category"].items():
                    if cat.startswith("drug"):
                        g = "drug"
                    elif cat == "S12_S12":
                        g = "seen"
                    elif cat == "S1_S1" or (cat == "S12_S1" and co == "S12"):
                        g = "held out"
                    else:
                        continue  # seen molecules mixed with held-out ones, not in the table
                    acc = groups[g].setdefault(smi, [0.0, 0.0, 0.0])
                    acc[0] += s["n_total"]
                    acc[1] += s["n_total"] * s["cosine_mean"]
                    acc[2] += s["n_total"] * s["top1_rate_mean"]
        for g, (n, cm, cs, r1) in PAPER_HOLDOUT.items():
            rows = np.array([[c / w, t / w] for w, c, t in groups[g].values()])
            check(f"holdout {g} N", len(rows), n, "int")
            check(f"holdout {g} cos mean", rows[:, 0].mean(), cm, "cos")
            check(f"holdout {g} cos std", rows[:, 0].std(ddof=0), cs, "cos")
            check(f"holdout {g} R@1", rows[:, 1].mean() * 100, r1)

    print("\n== SAE (analysis.tex, appendix app:sae)")
    perm = load(SAE / "permutation_test.json")
    loc = load(SAE / "spectral_localization.json")
    if perm is not None:
        check("selective features", perm["real_monosemantic_count"], PAPER_SAE["n_selective"], "int")
        check("permutations", perm["n_permutations"], PAPER_SAE["n_perm"], "int")
        check("null max selective", perm["null_max"], 0, "int")
    if loc is not None:
        feats = [f for f in loc["per_feature"] if f["baseline_sli"] > 0]
        above = [f for f in feats if f["sli"] > f["baseline_sli"]]
        check("features with a textbook band", len(feats), PAPER_SAE["n_band"], "int")
        check("of those, above chance", len(above), PAPER_SAE["n_above"], "int")
        check("mean in-band fraction", np.mean([f["sli"] for f in feats]), PAPER_SAE["mean_sli"], "cos")
        check("mean chance fraction", np.mean([f["baseline_sli"] for f in feats]), PAPER_SAE["mean_chance"], "cos")
        oh = [f["sli"] / f["baseline_sli"] for f in feats if f["fg"] == "OH"]
        check("OH features", len(oh), PAPER_SAE["oh_n"], "int")
        check("OH features above chance", sum(r > 1 for r in oh), PAPER_SAE["oh_above"], "int")
        check("OH min ratio", round(min(oh), 1), PAPER_SAE["oh_min"], 0.05)
        check("OH max ratio", round(max(oh), 1), PAPER_SAE["oh_max"], 0.05)
        for fg, key in (("COO", "coo"), ("SO2", "so2"), ("CHO", "cho")):
            sub = [f for f in feats if f["fg"] == fg]
            check(f"{fg} above chance", sum(f["sli"] > f["baseline_sli"] for f in sub), PAPER_SAE[key][0], "int")
            check(f"{fg} features", len(sub), PAPER_SAE[key][1], "int")
        sub = [f for f in feats if f["fg"] not in UNSURE_BANDS]
        check("excl. unsure bands, above chance", sum(f["sli"] > f["baseline_sli"] for f in sub), PAPER_SAE["excl"][0], "int")
        check("excl. unsure bands, features", len(sub), PAPER_SAE["excl"][1], "int")
        try:
            from scipy.stats import wilcoxon
            for lab, s, key in (("all", feats, "wilcoxon_p"), ("excl. unsure", sub, "wilcoxon_p_excl")):
                d = np.array([f["sli"] - f["baseline_sli"] for f in s])
                p = wilcoxon(d, alternative="greater").pvalue
                check(f"Wilcoxon p ({lab}), first digit", float(f"{p:.0e}"), PAPER_SAE[key], PAPER_SAE[key] * 0.5)
        except ImportError:
            print("  scipy not installed, Wilcoxon test skipped")

    print("\n== Text numbers with no saved source in this repo")
    for label, want in PAPER_NO_SOURCE.items():
        check(label, None, want)

    print(f"\nSummary: {counts['MATCH']} MATCH, {counts['MISMATCH']} MISMATCH, {counts['MISSING']} MISSING")
    for label, w, g in mismatches:
        print(f"  {label}: paper {w}, file {g if g is not None else 'not found'}")
    return 1 if counts["MISMATCH"] else 0


if __name__ == "__main__":
    sys.exit(main())
