"""Run all Table 1 baselines under one protocol (from baselines/).

Stages, in order (each skips finished work, so it can be resumed):
    dl-grid      validation grid for CNNAEU, CyCU-Net, DeepTrans-HSU (seed 0)
    sunsal-grid  SUnSAL lambda on the validation split
    select       print the DL grid and pick the SUnSAL lambda by validation component cosine
    dl-seeds     train the DL baselines with the shared setting (lr 1e-4, rec_weight 1), seeds 0 to 4
    evaluate     test evaluation of the DL baselines and SUnSAL
    classical    test evaluation of NNLS, MCR-ALS and VCA+FCLS (CPU is enough)
    summary      mean and std over seeds, written to results/fair/summary.json
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("results/fair")
DL_METHODS = ["cnnaeu", "cycunet", "deeptrans_hsu"]
DATASETS = ["binary", "ternary"]
LRS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
# Grid extensions for cases whose best lr was at an edge of LRS
EXTRA_LRS = {("cnnaeu", "ternary"): [1e-2], ("deeptrans_hsu", "ternary"): [1e-5]}


REC_WEIGHTS = [0.0, 1.0]
SUNSAL_LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SEEDS = [0, 1, 2, 3, 4]
FIXED_LR = 1e-4
FIXED_REC_WEIGHT = 1.0
FINAL = ROOT / "final_seeds"
CLASSICAL = ["nnls", "sunsal_admm", "mcr_als_dict", "vca_fcls_dict"]
PY = sys.executable


def lrs_for(method, dataset):
    return sorted(LRS + EXTRA_LRS.get((method, dataset), []))


def tag_config(lr, rec):
    return f"lr{lr:g}_rec{rec:g}"


def run_pool(jobs, gpus, jobs_per_gpu=1):
    """Run (name, cmd, done_marker, log) jobs over GPUs; skip jobs whose marker exists."""
    pending = [j for j in jobs if not j[2]()]
    print(f"{len(jobs) - len(pending)} of {len(jobs)} jobs already done; running {len(pending)}")
    slots = [g for g in gpus for _ in range(jobs_per_gpu)]
    running = {}
    while pending or running:
        for slot in [s for s in range(len(slots)) if s not in running]:
            if not pending:
                break
            name, cmd, _, log = pending.pop(0)
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[slot]))
            fh = open(log, "w")
            print(f"[start] GPU {slots[slot]}: {name}", flush=True)
            running[slot] = (name, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh)
        time.sleep(10)
        for slot, (name, proc, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                print(f"[{'done' if proc.returncode == 0 else 'FAILED ' + str(proc.returncode)}] {name}",
                      flush=True)
                del running[slot]


def dl_train_job(method, dataset, lr, rec, seed, out_dir):
    summary = Path(out_dir) / f"{method}_{dataset}_summary.json"
    done = lambda: summary.exists() and json.loads(summary.read_text()).get("done", False)
    cmd = [PY, "train_dl_fair.py", "--method", method, "--dataset", dataset, "--lr", str(lr),
           "--rec-weight", str(rec), "--seed", str(seed), "--out-dir", str(out_dir)]
    return (f"{method} {dataset} {tag_config(lr, rec)} seed{seed}", cmd, done, str(Path(out_dir) / "train.log"))


def eval_job(name, methods, datasets, out_dir, models_dir, extra=(), log_name="eval.log"):
    out_dir = Path(out_dir)
    done = lambda: all((out_dir / f"{m}_results.json").exists() for m in methods)
    cmd = [PY, "evaluate_baselines.py", "--methods", *methods, "--datasets", *datasets,
           "--models_dir", str(models_dir), "--output_dir", str(out_dir), *extra]
    return (name, cmd, done, str(out_dir / log_name))


def val_cos(result_json, key):
    r = json.loads(Path(result_json).read_text())["results"]
    return r[key]["unmixing"]["component_cosine"]["mean"]


def load_selected():
    return json.loads((ROOT / "selected.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dl-grid", "sunsal-grid", "select", "dl-seeds", "evaluate", "classical",
                                      "summary"])
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--jobs-per-gpu", type=int, default=1)
    args = ap.parse_args()

    if args.stage == "dl-grid":
        jobs = [dl_train_job(m, d, lr, rec, 0, ROOT / "grid" / f"{m}_{d}_{tag_config(lr, rec)}")
                for m, d in itertools.product(DL_METHODS, DATASETS)
                for lr, rec in itertools.product(lrs_for(m, d), REC_WEIGHTS)]
        run_pool(jobs, args.gpus, args.jobs_per_gpu)

    elif args.stage == "sunsal-grid":
        jobs = [eval_job(f"sunsal val lambda={lam:g}", ["sunsal_admm"], DATASETS,
                         ROOT / "sunsal_val" / f"lambda{lam:g}", ROOT / "models_classical",
                         ["--split", "val", "--sunsal_lambda", str(lam)])
                for lam in SUNSAL_LAMBDAS]
        run_pool(jobs, args.gpus, args.jobs_per_gpu)

    elif args.stage == "select":
        selected = {"criterion": "validation component cosine", "sunsal_lambda": {}}
        for m, d in itertools.product(DL_METHODS, DATASETS):
            for lr, rec in itertools.product(lrs_for(m, d), REC_WEIGHTS):
                f = ROOT / "grid" / f"{m}_{d}_{tag_config(lr, rec)}" / f"{m}_{d}_summary.json"
                if f.exists() and json.loads(f.read_text()).get("done"):
                    cos = json.loads(f.read_text())["best"]["val_comp_cos"]
                    print(f"  {m:14s} {d:8s} {tag_config(lr, rec):16s} val cos {cos:.4f}")
        for d in DATASETS:
            key = f"{d}_test"
            rows = [(val_cos(ROOT / "sunsal_val" / f"lambda{lam:g}" / "sunsal_admm_results.json", key), lam)
                    for lam in SUNSAL_LAMBDAS]
            for cos, lam in rows:
                print(f"  sunsal {d:8s} lambda {lam:g}: val cos {cos:.4f}")
            best = max(rows)
            selected["sunsal_lambda"][d] = {"lambda": best[1], "val_comp_cos": best[0]}
        (ROOT / "selected.json").write_text(json.dumps(selected, indent=2))
        print(f"Wrote {ROOT / 'selected.json'}")

    elif args.stage == "dl-seeds":
        jobs = [dl_train_job(m, d, FIXED_LR, FIXED_REC_WEIGHT, s, FINAL / f"seed{s}")
                for m, d in itertools.product(DL_METHODS, DATASETS) for s in SEEDS]
        run_pool(jobs, args.gpus, args.jobs_per_gpu)

    elif args.stage == "evaluate":
        sel = load_selected()
        jobs = [eval_job(f"dl test seed{s}", DL_METHODS, ["binary", "drug", "ternary"],
                         ROOT / "test" / f"dl_seed{s}", FINAL / f"seed{s}") for s in SEEDS]
        # SUnSAL is run per dataset, each with its own selected lambda
        for d, ds in [("binary", ["binary", "drug"]), ("ternary", ["ternary"])]:
            lam = sel["sunsal_lambda"][d]["lambda"]
            jobs.append(eval_job(f"sunsal test {d} lambda={lam:g}", ["sunsal_admm"], ds,
                                 ROOT / "test" / f"sunsal_{d}", ROOT / "models_classical",
                                 ["--sunsal_lambda", str(lam)]))
        run_pool(jobs, args.gpus, args.jobs_per_gpu)

    elif args.stage == "classical":
        jobs = [eval_job(f"{m} test", [m], ["binary", "drug", "ternary"], ROOT / "test" / "classical",
                         ROOT / "models_classical", log_name=f"{m}_eval.log")
                for m in ["mcr_als_dict", "nnls", "vca_fcls_dict"]]
        run_pool(jobs, args.gpus, args.jobs_per_gpu)

    elif args.stage == "summary":
        def metrics(f, key):
            r = json.loads(Path(f).read_text())["results"].get(key)
            if not r:
                return None
            m = r.get("matching", {})
            return {"cos": r["unmixing"]["component_cosine"]["mean"],
                    "r1": m.get("component_top1_accuracy"), "r5": m.get("component_top5_accuracy"),
                    "r10": m.get("component_top10_accuracy"), "mix": m.get("mixture_top1_accuracy")}

        out = {}
        sources = {m: [ROOT / "test" / f"dl_seed{s}" / f"{m}_results.json" for s in SEEDS] for m in DL_METHODS}
        for m in ["nnls", "mcr_als_dict", "vca_fcls_dict"]:
            sources[m] = [ROOT / "test" / "classical" / f"{m}_results.json"]
        for m, files in sources.items():
            for key in ["binary_test", "drug", "ternary_test"]:
                vals = [metrics(f, key) for f in files if f.exists()]
                vals = [v for v in vals if v]
                if vals:
                    out[f"{m}:{key}"] = {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals])),
                                             len(vals)] for k in vals[0]}
        for d, keys in [("binary", ["binary_test", "drug"]), ("ternary", ["ternary_test"])]:
            f = ROOT / "test" / f"sunsal_{d}" / "sunsal_admm_results.json"
            if f.exists():
                for key in keys:
                    v = metrics(f, key)
                    if v:
                        out[f"sunsal_admm:{key}"] = {k: [v[k], 0.0, 1] for k in v}
        (ROOT / "summary.json").write_text(json.dumps(out, indent=2))
        print(f"{'method:dataset':32s} {'Cos':>13s} {'R@1':>13s} {'R@5':>13s} {'R@10':>13s} {'Mix':>13s}  n")
        for k, v in out.items():
            cells = "".join(f" {v[x][0]:.3f}+/-{v[x][1]:.3f}" if v[x][0] is not None else "            -"
                            for x in ["cos", "r1", "r5", "r10", "mix"])
            print(f"{k:32s}{cells}  {v['cos'][2]}")


if __name__ == "__main__":
    main()
