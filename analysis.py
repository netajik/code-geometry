#!/usr/bin/env python3
"""
Code Geometry Error Analysis

Loads saved answers and labels from the code-generation pipeline,
aggregates error categories (syntax, logic, timeout, garbage), and
produces plots + JSON summary for manifold / REMA-style analysis.

**level_run_id** names one saved batch of tasks
(on-disk: ``level_run_<id>.json`` / ``level_run_<id>_layer{L}.npy``). **Population** = correct vs wrong (outcomes), not ``level_run_id``.

No GPU required.

Usage:
    python analysis.py
    python analysis.py --config path.yaml
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ====================================================================
# 1. CONFIG + LOGGING
# ====================================================================

def get_dataset_name(cfg):
    """Return folder name for outputs. Must match pipeline.py."""
    d = cfg.get("dataset", {})
    if d.get("output_name"):
        return d["output_name"]
    source = d.get("source", "json")
    if source == "json":
        return "custom"
    if source == "huggingface":
        repo = d.get("hf_repo", "")
        return repo.split("/")[-1] if repo else "hf"
    if source == "json_levels":
        return "levels"
    return "custom"


def _resolve_output_roots(workspace_str, data_root_str, config_base):
    """Same logic as pipeline.resolve_output_roots (keep in sync)."""
    base = Path(config_base)
    ws = Path(workspace_str).expanduser()
    dr = Path(data_root_str).expanduser()
    if not ws.is_absolute():
        ws = (base / ws).resolve()
    else:
        ws = ws.resolve()
    if not dr.is_absolute():
        dr = (base / dr).resolve()
    else:
        dr = dr.resolve()
    return ws, dr


def load_config(config_path=None):
    """Load config and derive paths. Outputs live under dataset-named subdirs.
    Relative paths are resolved against the config file's directory (repo root).
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    config_path = Path(config_path)
    base = config_path.resolve().parent
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ws, dr = _resolve_output_roots(
        cfg["paths"]["workspace"], cfg["paths"]["data_root"], base
    )
    dset = get_dataset_name(cfg)
    cfg["paths"]["dataset_name"] = dset
    cfg["paths"]["labels_dir"] = str(ws / dset / "labels")
    cfg["paths"]["logs_dir"] = str(ws / dset / "logs")
    cfg["paths"]["plots_dir"] = str(ws / dset / "plots")
    cfg["paths"]["activations_dir"] = str(dr / dset / "activations")
    cfg["paths"]["answers_dir"] = str(dr / dset / "answers")
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["plots_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(cfg):
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("code_analysis")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    fh = RotatingFileHandler(logs_dir / "analysis.log", maxBytes=10 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)
    return logger


# ====================================================================
# 2. DATA LOADING
# ====================================================================

def discover_level_run_ids(answers_dir):
    """List level_run_id strings from ``level_run_<id>.json`` in answers_dir."""
    ans_dir = Path(answers_dir)
    if not ans_dir.exists():
        return []
    names = []
    for f in ans_dir.glob("level_run_*.json"):
        name = f.stem.replace("level_run_", "", 1)
        names.append(name)
    return sorted(names)


def load_answers(cfg, logger):
    ans_dir = Path(cfg["paths"]["answers_dir"])
    stems = discover_level_run_ids(ans_dir)
    answers = {}
    for stem_id in stems:
        path = ans_dir / f"level_run_{stem_id}.json"
        with open(path) as f:
            data = json.load(f)
        rid = data.get("level_run_id") or stem_id
        answers[rid] = data
        n = len(answers[rid]["results"])
        nc = answers[rid]["n_correct"]
        acc = answers[rid]["accuracy"]
        logger.info(f"Level run {rid}: {n} results, accuracy {acc:.1%} ({nc}/{n})")
    return answers


def load_labels(cfg, logger):
    lab_dir = Path(cfg["paths"]["labels_dir"])
    labels = {}
    for path in sorted(lab_dir.glob("level_run_*.json")):
        stem_id = path.stem.replace("level_run_", "", 1)
        with open(path) as f:
            data = json.load(f)
        rid = data.get("level_run_id") or stem_id
        rows = []
        for p in data["problems"]:
            row = dict(p.get("labels") or {})
            if "prompt" in p:
                row["prompt"] = p["prompt"]
            rows.append(row)
        labels[rid] = rows
        logger.info(f"Level run {rid}: loaded {len(labels[rid])} label sets")
    return labels


def merge_data(answers, labels, logger):
    merged = {}
    for rid in answers:
        results = answers[rid]["results"]
        labs = labels.get(rid, [])
        if len(labs) != len(results):
            logger.warning(f"Level run {rid}: label count {len(labs)} != result count {len(results)}")
            labs = [{}] * len(results)  # pad
        merged[rid] = [{**r, **lab} for r, lab in zip(results, labs)]
        logger.info(f"Level run {rid}: merged {len(merged[rid])} entries")
    return merged


# ====================================================================
# 3. ERROR CATEGORY SUMMARY (already in results; aggregate)
# ====================================================================

def summarize_error_categories(merged, logger):
    for rid in sorted(merged):
        counts = Counter()
        for p in merged[rid]:
            if p.get("correct"):
                counts["correct"] += 1
            else:
                cat = p.get("error_category", "unknown")
                counts[cat] += 1
        n_wrong = sum(v for k, v in counts.items() if k != "correct")
        logger.info(f"Level run {rid}: {n_wrong} wrong — {dict(counts)}")


# ====================================================================
# 4. PLOTTING
# ====================================================================

ERROR_COLORS = {
    "correct": "#4CAF50",
    "syntax_error": "#F44336",
    "logic_error": "#FF9800",
    "timeout": "#FFC107",
    "wrong_output": "#9C27B0",
    "garbage": "#212121",
    "no_tests": "#757575",
    "unknown": "#9E9E9E",
}


def plot_error_categories(merged, cfg, logger):
    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    level_run_ids = sorted(merged)
    categories = ["correct", "syntax_error", "logic_error", "timeout", "wrong_output", "garbage", "no_tests", "unknown"]
    counts = {cat: [] for cat in categories}
    for rid in level_run_ids:
        c = Counter()
        for p in merged[rid]:
            if p.get("correct"):
                c["correct"] += 1
            else:
                cat = p.get("error_category") or "unknown"
                c[cat] += 1
        for cat in categories:
            counts[cat].append(c.get(cat, 0))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(level_run_ids))
    bottom = np.zeros(len(level_run_ids))
    for cat in categories:
        vals = np.array(counts[cat], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=cat, color=ERROR_COLORS.get(cat, "#999"), width=0.6)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(level_run_ids)
    ax.set_ylabel("Count")
    ax.set_title("Code generation error categories by level run")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "error_categories.png", dpi=150)
    plt.close(fig)
    logger.info("Saved error_categories.png")


def plot_error_category_heatmap(merged, cfg, logger):
    """Heatmap: level_run × error category (counts); correct shown as separate row label."""
    plots_dir = Path(cfg["paths"]["plots_dir"])
    level_run_ids = sorted(merged.keys())
    if not level_run_ids:
        return
    cats = set()
    for rid in level_run_ids:
        for p in merged[rid]:
            if p.get("correct"):
                cats.add("correct")
            else:
                cats.add(p.get("error_category") or "unknown")
    categories = sorted(cats, key=lambda x: (x != "correct", x))
    mat = np.zeros((len(level_run_ids), len(categories)), dtype=float)
    for i, rid in enumerate(level_run_ids):
        for j, cat in enumerate(categories):
            c = 0
            for p in merged[rid]:
                if cat == "correct" and p.get("correct"):
                    c += 1
                elif cat != "correct" and not p.get("correct") and (p.get("error_category") or "unknown") == cat:
                    c += 1
            mat[i, j] = c
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(categories) + 2), max(4, 0.4 * len(level_run_ids) + 2)))
    im = ax.imshow(mat.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(level_run_ids)))
    ax.set_xticklabels(level_run_ids, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("Level run")
    ax.set_ylabel("Category")
    ax.set_title("Counts: level run × (correct / error category)")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / "error_category_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved error_category_heatmap.png")


def plot_num_test_lines_vs_correctness(merged, cfg, logger):
    """Box-ish comparison: num_test_lines for correct vs wrong (per level run)."""
    plots_dir = Path(cfg["paths"]["plots_dir"])
    level_run_ids = sorted(merged.keys())
    if not level_run_ids:
        return
    fig, axes = plt.subplots(1, len(level_run_ids), figsize=(4.5 * len(level_run_ids), 4.5), squeeze=False)
    for idx, rid in enumerate(level_run_ids):
        ax = axes[0][idx]
        correct_vals = []
        wrong_vals = []
        for p in merged[rid]:
            ntl = p.get("num_test_lines")
            if ntl is None:
                continue
            if p.get("correct"):
                correct_vals.append(ntl)
            else:
                wrong_vals.append(ntl)
        datasets, labels = [], []
        if correct_vals:
            datasets.append(correct_vals)
            labels.append("correct")
        if wrong_vals:
            datasets.append(wrong_vals)
            labels.append("wrong")
        if not datasets:
            ax.text(0.5, 0.5, "no num_test_lines", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(rid)
            continue
        ax.violinplot(datasets, positions=list(range(1, len(datasets) + 1)), showmeans=True, showmedians=True)
        ax.set_xticks(list(range(1, len(datasets) + 1)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("num_test_lines (from labels)")
        ax.set_title(rid)
    fig.suptitle("Test size vs correctness", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "num_test_lines_vs_correctness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved num_test_lines_vs_correctness.png")


def plot_outcome_distribution_pooled(merged, cfg, logger):
    """Horizontal bar: pooled counts over all level runs (correct + each error type)."""
    plots_dir = Path(cfg["paths"]["plots_dir"])
    c = Counter()
    for rid in merged:
        for p in merged[rid]:
            if p.get("correct"):
                c["correct"] += 1
            else:
                c[p.get("error_category") or "unknown"] += 1
    if not c:
        return
    cats = sorted(c.keys(), key=lambda x: (x != "correct", x))
    vals = [c[k] for k in cats]
    colors = [ERROR_COLORS.get(k, "#78909C") for k in cats]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(cats))))
    ax.barh(cats, vals, color=colors)
    ax.set_xlabel("Count (all level runs pooled)")
    ax.set_title("Outcome distribution (code generation)")
    fig.tight_layout()
    fig.savefig(plots_dir / "outcome_distribution_pooled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved outcome_distribution_pooled.png")


def plot_error_distribution_by_level_run(merged, cfg, logger):
    """Grouped bars: wrong-only error categories per level run (code-native diagnostic)."""
    plots_dir = Path(cfg["paths"]["plots_dir"])
    level_run_ids = sorted(merged.keys())
    if not level_run_ids:
        return
    wrong_cats = set()
    for rid in level_run_ids:
        for p in merged[rid]:
            if not p.get("correct"):
                wrong_cats.add(p.get("error_category") or "unknown")
    wrong_cats = sorted(wrong_cats)
    if not wrong_cats:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No wrong samples", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(plots_dir / "error_distribution_by_level_run.png", dpi=150)
        plt.close(fig)
        logger.info("Saved error_distribution_by_level_run.png (empty)")
        return
    x = np.arange(len(level_run_ids), dtype=float)
    ncat = len(wrong_cats)
    width = min(0.8 / ncat, 0.25)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(level_run_ids)), 5))
    for i, cat in enumerate(wrong_cats):
        counts = [
            sum(
                1
                for p in merged[sp]
                if not p.get("correct") and (p.get("error_category") or "unknown") == cat
            )
            for sp in level_run_ids
        ]
        ax.bar(x + (i - ncat / 2 + 0.5) * width, counts, width, label=cat, color=ERROR_COLORS.get(cat, "#999"))
    ax.set_xticks(x)
    ax.set_xticklabels(level_run_ids, rotation=20, ha="right")
    ax.set_ylabel("Count (wrong only)")
    ax.set_title("Error category counts by level run")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_distribution_by_level_run.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved error_distribution_by_level_run.png")


def plot_accuracy_vs_num_test_lines(merged, cfg, logger):
    """Accuracy vs discrete num_test_lines (per level run)."""
    plots_dir = Path(cfg["paths"]["plots_dir"])
    level_run_ids = sorted(merged.keys())
    if not level_run_ids:
        return
    fig, axes = plt.subplots(1, len(level_run_ids), figsize=(4.8 * len(level_run_ids), 4.2), squeeze=False)
    for idx, rid in enumerate(level_run_ids):
        ax = axes[0][idx]
        stats = defaultdict(lambda: [0, 0])
        for p in merged[rid]:
            ntl = p.get("num_test_lines")
            if ntl is None:
                continue
            k = int(ntl)
            stats[k][1] += 1
            if p.get("correct"):
                stats[k][0] += 1
        if not stats:
            ax.text(0.5, 0.5, "no num_test_lines", ha="center", transform=ax.transAxes)
            ax.set_title(rid)
            continue
        keys = sorted(stats.keys())
        accs = [stats[k][0] / stats[k][1] if stats[k][1] else 0.0 for k in keys]
        ns = [stats[k][1] for k in keys]
        bars = ax.bar(range(len(keys)), accs, color="steelblue", alpha=0.85)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([str(k) for k in keys])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("num_test_lines")
        ax.set_ylabel("Accuracy")
        ax.set_title(rid)
        for j, (b, n) in enumerate(zip(bars, ns)):
            ax.text(b.get_x() + b.get_width() / 2, min(b.get_height() + 0.04, 1.0), f"n={n}", ha="center", fontsize=7)
    fig.suptitle("Accuracy vs number of assert lines (by level run)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_num_test_lines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy_vs_num_test_lines.png")


def plot_accuracy_vs_prompt_length(merged, cfg, logger):
    """Accuracy vs prompt character length (quantile bins per level run)."""
    plots_dir = Path(cfg["paths"]["plots_dir"])
    level_run_ids = sorted(merged.keys())
    if not level_run_ids:
        return
    fig, axes = plt.subplots(1, len(level_run_ids), figsize=(4.8 * len(level_run_ids), 4.2), squeeze=False)
    for idx, rid in enumerate(level_run_ids):
        ax = axes[0][idx]
        lens = []
        corr = []
        for p in merged[rid]:
            lens.append(len(p.get("prompt") or ""))
            corr.append(1.0 if p.get("correct") else 0.0)
        if not lens:
            ax.set_title(rid)
            continue
        lens_a = np.array(lens, dtype=float)
        corr_a = np.array(corr, dtype=float)
        if np.all(lens_a == lens_a[0]) or len(lens_a) < 5:
            ax.text(0.5, 0.5, "uniform or n<5", ha="center", transform=ax.transAxes)
            ax.set_title(rid)
            continue
        qs = [0.0, 0.25, 0.5, 0.75, 1.0]
        edges = np.quantile(lens_a, qs)
        edges = np.unique(edges)
        if len(edges) < 2:
            ax.text(0.5, 0.5, "need varying lengths", ha="center", transform=ax.transAxes)
            ax.set_title(rid)
            continue
        accs = []
        counts = []
        labels = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i < len(edges) - 2:
                m = (lens_a >= lo) & (lens_a < hi)
            else:
                m = (lens_a >= lo) & (lens_a <= hi)
            cnt = int(m.sum())
            counts.append(cnt)
            accs.append(float(corr_a[m].mean()) if cnt else 0.0)
            labels.append(f"{int(lo)}–{int(hi)}")
        x = np.arange(len(accs))
        ax.bar(x, accs, color="darkslateblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Prompt length (chars, quartile bins)")
        ax.set_title(rid)
        for j, n in enumerate(counts):
            ax.text(j, min(accs[j] + 0.05, 1.0), f"n={n}", ha="center", fontsize=7)
    fig.suptitle("Accuracy vs prompt length (character quartiles)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_prompt_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy_vs_prompt_length.png")


# ====================================================================
# 5. SUMMARY JSON
# ====================================================================

def build_summary(answers, merged):
    summary = {"level_runs": {}}
    for rid in sorted(merged):
        n = len(merged[rid])
        n_correct = sum(1 for p in merged[rid] if p.get("correct"))
        cats = Counter(p.get("error_category", "unknown") for p in merged[rid] if not p.get("correct"))
        summary["level_runs"][rid] = {
            "n_problems": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n else 0,
            "error_categories": dict(cats),
        }
    return summary


def save_summary(summary, cfg, logger):
    for dest_key in ("answers_dir", "labels_dir"):
        dest = Path(cfg["paths"][dest_key])
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "analysis_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved {path}")


# ====================================================================
# 6. MAIN
# ====================================================================

def main(config_path=None):
    t_start = time.time()
    cfg = load_config(config_path)
    logger = setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("Code Geometry Error Analysis")
    logger.info("=" * 60)

    logger.info("--- Loading data ---")
    answers = load_answers(cfg, logger)
    if not answers:
        logger.error("No answer files found. Run pipeline first.")
        sys.exit(1)
    labels = load_labels(cfg, logger)
    merged = merge_data(answers, labels, logger)

    logger.info("--- Error category summary ---")
    summarize_error_categories(merged, logger)

    logger.info("--- Generating plots ---")
    plot_error_categories(merged, cfg, logger)
    plot_error_category_heatmap(merged, cfg, logger)
    plot_num_test_lines_vs_correctness(merged, cfg, logger)
    plot_outcome_distribution_pooled(merged, cfg, logger)
    plot_error_distribution_by_level_run(merged, cfg, logger)
    plot_accuracy_vs_num_test_lines(merged, cfg, logger)
    plot_accuracy_vs_prompt_length(merged, cfg, logger)

    logger.info("--- Summary ---")
    summary = build_summary(answers, merged)
    save_summary(summary, cfg, logger)

    logger.info("=" * 60)
    for rid in sorted(summary["level_runs"]):
        s = summary["level_runs"][rid]
        logger.info(f"  {rid}: {s['accuracy']:.1%} ({s['n_correct']}/{s['n_problems']})")
    logger.info(f"Analysis complete in {time.time()-t_start:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Geometry Error Analysis")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
