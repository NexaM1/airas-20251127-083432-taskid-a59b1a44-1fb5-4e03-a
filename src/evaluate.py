"""Independent evaluation script.
Fetches histories & summaries from WandB and creates JSON + figures.
Supports both positional CLI arguments and Hydra-style key=value overrides."""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy import stats

sns.set(style="whitegrid")
PRIMARY_METRIC_KEY = "best_val_fid"  # lower = better


# ---------------------------------------------------------------------------
# Utility helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(path)


def _parse_cli() -> argparse.Namespace:
    kv_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    kv = {}
    positional: List[str] = []
    for arg in os.sys.argv[1:]:
        m = kv_pattern.match(arg)
        if m:
            kv[m.group(1)] = m.group(2)
        else:
            positional.append(arg)

    if positional:
        parser = argparse.ArgumentParser()
        parser.add_argument("results_dir")
        parser.add_argument("run_ids", help="JSON list, e.g. '[\"run1\", \"run2\"]'")
        args = parser.parse_args(positional)
        kv.setdefault("results_dir", args.results_dir)
        kv.setdefault("run_ids", args.run_ids)

    if "results_dir" not in kv or "run_ids" not in kv:
        raise ValueError("Both results_dir and run_ids must be supplied (positional or key=value)")

    ns = argparse.Namespace()
    ns.results_dir = kv["results_dir"]
    ns.run_ids = kv["run_ids"]
    return ns


# ---------------------------------------------------------------------------
# Per-run processing --------------------------------------------------------
# ---------------------------------------------------------------------------

def _plot_learning_curves(history: pd.DataFrame, out_dir: str, run_id: str):
    keys = [c for c in history.columns if c.startswith("train_") or c.startswith("val_")]
    if not keys:
        return
    plt.figure(figsize=(7, 4))
    for k in keys:
        plt.plot(history[k].dropna().values, label=k)
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("metric value")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    plt.savefig(path)
    plt.close()
    print(path)


def _plot_confusion_matrix(summary: Dict, out_dir: str, run_id: str):
    needed = {f"confusion_{k}" for k in ("tp", "fp", "tn", "fn")}
    if not needed.issubset(summary):
        return
    matrix = np.array([
        [summary["confusion_tp"], summary["confusion_fp"]],
        [summary["confusion_fn"], summary["confusion_tn"]],
    ])
    plt.figure(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pos", "neg"], yticklabels=["pos", "neg"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    plt.savefig(path)
    plt.close()
    print(path)


def extract_and_save(api: wandb.Api, entity: str, project: str, run_id: str, out_root: str) -> Dict:
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(keys=[], pandas=True)
    summary = dict(run.summary._json_dict)
    config = dict(run.config)

    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    _save_json({"summary": summary, "config": config, "history": history.to_dict(orient="list")},
               os.path.join(run_dir, "metrics.json"))

    _plot_learning_curves(history, run_dir, run_id)
    _plot_confusion_matrix(summary, run_dir, run_id)
    return summary


# ---------------------------------------------------------------------------
# Aggregated analysis -------------------------------------------------------
# ---------------------------------------------------------------------------

def aggregated_analysis(summaries: Dict[str, Dict], out_root: str):
    comp_dir = os.path.join(out_root, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    metric_table: Dict[str, Dict[str, float]] = defaultdict(dict)
    for rid, summ in summaries.items():
        for k, v in summ.items():
            if isinstance(v, (int, float)):
                metric_table[k][rid] = float(v)

    prim_vals = metric_table[PRIMARY_METRIC_KEY]

    best_prop = min(((r, v) for r, v in prim_vals.items() if re.search(r"proposed|hices", r, re.I)),
                    key=lambda t: t[1])
    best_base = min(((r, v) for r, v in prim_vals.items() if re.search(r"baseline|comparative|afder", r, re.I)),
                    key=lambda t: t[1])
    gap = (best_base[1] - best_prop[1]) / best_base[1] * 100.0

    aggregated = {
        "primary_metric": "FID @ImageNet64 with A-steps≈1.7 (≈57 % compute saving)",
        "metrics": metric_table,
        "best_proposed": {"run_id": best_prop[0], "value": best_prop[1]},
        "best_baseline": {"run_id": best_base[0], "value": best_base[1]},
        "gap": gap,
    }
    _save_json(aggregated, os.path.join(comp_dir, "aggregated_metrics.json"))

    # Figure: bar chart ----------------------------------------------------
    plt.figure(figsize=(max(6, 0.8 * len(prim_vals)), 4))
    sns.barplot(x=list(prim_vals.keys()), y=list(prim_vals.values()), palette="crest")
    for i, (k, v) in enumerate(prim_vals.items()):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("FID (↓)")
    plt.tight_layout()
    path_bar = os.path.join(comp_dir, "comparison_primary_metric_bar.pdf")
    plt.savefig(path_bar)
    plt.close()
    print(path_bar)

    # Box plot + Welch-t ---------------------------------------------------
    groups = {"proposed": [], "baseline": []}
    for rid, val in prim_vals.items():
        if re.search(r"proposed|hices", rid, re.I):
            groups["proposed"].append(val)
        else:
            groups["baseline"].append(val)
    if groups["proposed"] and groups["baseline"]:
        t_stat, p_val = stats.ttest_ind(groups["proposed"], groups["baseline"], equal_var=False)
        plt.figure(figsize=(4, 4))
        sns.boxplot(data=[groups["proposed"], groups["baseline"]], palette="pastel")
        plt.xticks([0, 1], ["proposed", "baseline"])
        plt.ylabel("FID (↓)")
        plt.title(f"Welch t-test p={p_val:.4f}")
        plt.tight_layout()
        path_box = os.path.join(comp_dir, "comparison_primary_metric_box.pdf")
        plt.savefig(path_box)
        plt.close()
        print(path_box)


# ---------------------------------------------------------------------------
# Entry-point ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    args = _parse_cli()
    run_ids: List[str] = json.loads(args.run_ids)

    with open(os.path.join("config", "config.yaml"), "r") as f:
        root_cfg = yaml.safe_load(f)
    entity = root_cfg["wandb"]["entity"]
    project = root_cfg["wandb"]["project"]

    api = wandb.Api()
    summaries = {}
    for rid in run_ids:
        summaries[rid] = extract_and_save(api, entity, project, rid, args.results_dir)

    aggregated_analysis(summaries, args.results_dir)


if __name__ == "__main__":
    main()