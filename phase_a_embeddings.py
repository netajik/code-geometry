#!/usr/bin/env python3
"""Phase A — UMAP/t-SNE embeddings for code activations, colored by correct/error_category.

Loads activations (``level_run_<id>_layer{L}.npy``) and answers;
builds coloring DataFrame per **level_run_id** (difficulty tier / batch, not outcome population).
Runs UMAP and t-SNE when sample count is within configured bounds; computes interestingness (e.g. correct vs wrong separation);
saves embedding CSVs and 2D plots (``--skip-plots`` writes CSVs without PNGs).

Usage:
  python phase_a_embeddings.py --config config.yaml
  python phase_a_embeddings.py --config config.yaml --skip-plots
"""

import argparse
import json
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from umap import UMAP
except ImportError:
    UMAP = None
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None

# t-SNE is unstable and sklearn defaults assume n ≫ perplexity; skip below this unless overridden in config.
# Default 30+ aligns with padded level JSONs (see scripts/pad_level_json_to_count.py).
DEFAULT_MIN_TSNE_SAMPLES = 30


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config_paths(cfg, config_path):
    """Resolve workspace/data_root: expand ~ (Google Drive etc.), then repo-relative or absolute."""
    base = Path(config_path).resolve().parent
    for key in ("workspace", "data_root"):
        p = Path(cfg["paths"][key]).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        else:
            p = p.resolve()
        cfg["paths"][key] = str(p)


def get_dataset_name(cfg):
    """Must match pipeline.py."""
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


def min_tsne_samples_from_cfg(cfg):
    """Minimum n_samples to run t-SNE; raise ``dataset.max_problems`` or add tasks per level if below."""
    pa = cfg.get("phase_a") or {}
    return max(2, int(pa.get("min_tsne_samples", DEFAULT_MIN_TSNE_SAMPLES)))


def derive_paths(cfg):
    ws = Path(cfg["paths"]["workspace"])
    dr = Path(cfg["paths"]["data_root"])
    dset = get_dataset_name(cfg)
    ws_dset = ws / dset
    return {
        "workspace": ws,
        "workspace_dataset": ws_dset,
        "data_root": dr,
        "dataset_name": dset,
        "labels_dir": ws_dset / "labels",
        "answers_dir": dr / dset / "answers",
        "act_dir": dr / dset / "activations",
        "phase_a_data": dr / dset / "phase_a",
        "phase_a_plots": ws_dset / "plots" / "phase_a",
    }


def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_a_code")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_a_embeddings.log", maxBytes=10_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Activations: level_run_<id>_layer<L>.npy (e.g. level_run_level_01_layer8.npy).
# Parse id with regex; do not use naive stem.split("_") (breaks on underscores in id).
_ACT_FILE_RE = re.compile(r"^level_run_(.+)_layer\d+$")


def discover_level_run_ids(act_dir):
    """level_run_id strings from activation filenames (matches answers JSON stem)."""
    act_dir = Path(act_dir)
    if not act_dir.exists():
        return []
    ids = set()
    for f in act_dir.glob("level_run_*_layer*.npy"):
        m = _ACT_FILE_RE.match(f.stem)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def load_answers_for_level_run(level_run_id, answers_dir):
    path = Path(answers_dir) / f"level_run_{level_run_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_activations(level_run_id, layer, act_dir):
    path = Path(act_dir) / f"level_run_{level_run_id}_layer{layer}.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def build_coloring_df(level_run_id, answers_data, n_samples):
    """Build DataFrame with correct, error_category, problem index for coloring."""
    results = answers_data.get("results", [])
    if len(results) != n_samples:
        return None
    rows = []
    for i, r in enumerate(results):
        rows.append({
            "problem_idx": i,
            "correct": r.get("correct", False),
            "error_category": r.get("error_category", "unknown"),
            "task_id": r.get("task_id", ""),
        })
    return pd.DataFrame(rows)


def run_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, logger=None):
    if UMAP is None:
        if logger:
            logger.warning("umap-learn not installed; skipping UMAP")
        return None
    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(X)


def run_tsne(X, n_components=2, perplexity=30, random_state=42, logger=None):
    if TSNE is None:
        if logger:
            logger.warning("sklearn not installed or no TSNE; skipping t-SNE")
        return None
    n = int(X.shape[0])
    # sklearn requires perplexity < n_samples (strict). Do not force perplexity >= 5 on tiny sets.
    if n < 2:
        if logger:
            logger.warning("t-SNE skipped: need at least 2 samples, got %s", n)
        return None
    max_perp = n - 1
    target = min(perplexity, max(1, n // 4))
    eff_perp = max(1, min(target, max_perp))
    embed = TSNE(n_components=n_components, perplexity=eff_perp, random_state=random_state)
    return embed.fit_transform(X)


def divergence_score(df, x_col, y_col, label_col="correct"):
    """Simple separation: mean distance between correct and wrong centroids (normalized by spread)."""
    if label_col not in df.columns or df[label_col].nunique() < 2:
        return 0.0
    g = df.groupby(label_col)[[x_col, y_col]].mean()
    if len(g) < 2:
        return 0.0
    pts = g.values
    diff = np.linalg.norm(pts[0] - pts[1])
    spread = df[[x_col, y_col]].std().mean()
    return diff / (spread + 1e-8)


def _scatter_correct_wrong(df, x0, x1, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for correct_val, color in [(True, "green"), (False, "red")]:
        sub = df[df["correct"] == correct_val]
        ax.scatter(sub[x0], sub[x1], c=color, alpha=0.5, s=12, label="correct" if correct_val else "wrong")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_error_category(df, x0, x1, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = sorted(df["error_category"].astype(str).unique().tolist())
    if len(cats) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    palette = plt.cm.tab10(np.linspace(0, 0.99, 10))
    for i, c in enumerate(cats):
        sub = df[df["error_category"].astype(str) == c]
        color = palette[i % len(palette)]
        lab = c if len(c) <= 24 else c[:21] + "..."
        ax.scatter(sub[x0], sub[x1], alpha=0.55, s=14, color=color, label=lab)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_divergence_heatmaps(scores_df, plot_dir, logger):
    """Level run × layer heatmaps of correct-vs-wrong divergence (per method)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_col = "level_run_id"
    for method, sub in scores_df.groupby("method"):
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=run_col, columns="layer", values="divergence", aggfunc="first")
        pivot = pivot.sort_index()
        col_order = sorted(pivot.columns, key=lambda x: (isinstance(x, (int, float)), x))
        pivot = pivot.reindex(columns=col_order)
        fig_h = max(4.0, 0.35 * len(pivot.index) + 1.5)
        fig_w = max(6.0, 0.45 * len(pivot.columns) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Level run")
        ax.set_title(f"Correct vs wrong divergence ({method})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = plot_dir / f"divergence_heatmap_{method}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out)


def main():
    parser = argparse.ArgumentParser(description="Phase A: UMAP/t-SNE for code geometry")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--skip-plots", action="store_true", help="Only write CSVs, no PNGs")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)
    resolve_config_paths(cfg, config_path)
    paths = derive_paths(cfg)
    for k, v in paths.items():
        if hasattr(v, "mkdir"):
            v.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["workspace_dataset"])

    if UMAP is None:
        logger.warning(
            "umap-learn not installed; UMAP will be skipped. "
            "Run: pip install umap-learn  (or pip install -r requirements.txt)"
        )

    layers = cfg["model"]["layers"]
    level_run_ids = discover_level_run_ids(paths["act_dir"])
    if not level_run_ids:
        logger.error("No activation files found; run pipeline first.")
        sys.exit(1)

    min_tsne_n = min_tsne_samples_from_cfg(cfg)
    logger.info(
        "Phase A: Building embeddings for level_run_ids=%s layers=%s (t-SNE when %s <= n <= 2000; "
        "set phase_a.min_tsne_samples in config to change floor)",
        level_run_ids,
        layers,
        min_tsne_n,
    )

    phase_a_data = paths["phase_a_data"]
    (phase_a_data / "embeddings").mkdir(parents=True, exist_ok=True)
    (phase_a_data / "coloring_dfs").mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        paths["phase_a_plots"].mkdir(parents=True, exist_ok=True)

    scores = []
    tsne_skipped_logged = set()
    for rid in level_run_ids:
        answers_data = load_answers_for_level_run(rid, paths["answers_dir"])
        if not answers_data:
            logger.warning("No answers for level_run_id %s", rid)
            continue
        results = answers_data.get("results", [])
        n = len(results)

        for layer in layers:
            act = load_activations(rid, layer, paths["act_dir"])
            if act is None or act.shape[0] != n:
                continue
            X = np.asarray(act, dtype=np.float32)
            df = build_coloring_df(rid, answers_data, n)
            if df is None:
                continue
            df.to_pickle(phase_a_data / "coloring_dfs" / f"{rid}_layer{layer}_coloring.pkl")

            # UMAP
            emb_umap = run_umap(X, logger=logger)
            if emb_umap is not None:
                df["umap_0"] = emb_umap[:, 0]
                df["umap_1"] = emb_umap[:, 1]
                div = divergence_score(df, "umap_0", "umap_1", "correct")
                scores.append({"level_run_id": rid, "layer": layer, "method": "umap", "divergence": div})
                out_csv = phase_a_data / "embeddings" / f"{rid}_layer{layer}_umap.csv"
                df.to_csv(out_csv, index=False)
                logger.info("Level run %s layer %s UMAP divergence (populations: correct vs wrong) = %.3f", rid, layer, div)
                if not args.skip_plots and df["correct"].nunique() > 1:
                    _scatter_correct_wrong(
                        df,
                        "umap_0",
                        "umap_1",
                        paths["phase_a_plots"] / f"{rid}_layer{layer}_umap.png",
                        f"{rid} layer {layer} UMAP (correct vs wrong)",
                    )
                if not args.skip_plots and df["error_category"].astype(str).nunique() > 1:
                    _scatter_error_category(
                        df,
                        "umap_0",
                        "umap_1",
                        paths["phase_a_plots"] / f"{rid}_layer{layer}_umap_by_category.png",
                        f"{rid} layer {layer} UMAP (error category)",
                    )

            # t-SNE (slower than UMAP); skipped when n is outside [min_tsne_samples, 2000]
            if n > 2000:
                pass
            elif n < min_tsne_n:
                if rid not in tsne_skipped_logged:
                    tsne_skipped_logged.add(rid)
                    logger.warning(
                        "t-SNE skipped for level_run_id=%s: n=%s < phase_a.min_tsne_samples=%s. "
                        "Increase tasks in that level's JSON, set dataset.max_problems higher, or lower "
                        "phase_a.min_tsne_samples (not recommended below ~20).",
                        rid,
                        n,
                        min_tsne_n,
                    )
            else:
                emb_tsne = run_tsne(X, logger=logger)
                if emb_tsne is not None:
                    df_tsne = df.copy()
                    df_tsne["tsne_0"] = emb_tsne[:, 0]
                    df_tsne["tsne_1"] = emb_tsne[:, 1]
                    div_tsne = divergence_score(df_tsne, "tsne_0", "tsne_1", "correct")
                    scores.append({"level_run_id": rid, "layer": layer, "method": "tsne", "divergence": div_tsne})
                    df_tsne.to_csv(phase_a_data / "embeddings" / f"{rid}_layer{layer}_tsne.csv", index=False)
                    if not args.skip_plots and df_tsne["correct"].nunique() > 1:
                        _scatter_correct_wrong(
                            df_tsne,
                            "tsne_0",
                            "tsne_1",
                            paths["phase_a_plots"] / f"{rid}_layer{layer}_tsne.png",
                            f"{rid} layer {layer} t-SNE (correct vs wrong)",
                        )
                    if not args.skip_plots and df_tsne["error_category"].astype(str).nunique() > 1:
                        _scatter_error_category(
                            df_tsne,
                            "tsne_0",
                            "tsne_1",
                            paths["phase_a_plots"] / f"{rid}_layer{layer}_tsne_by_category.png",
                            f"{rid} layer {layer} t-SNE (error category)",
                        )

    if scores:
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(phase_a_data / "interestingness_scores.csv", index=False)
        logger.info("Phase A complete. Scores in %s", phase_a_data / "interestingness_scores.csv")
        if not args.skip_plots:
            save_divergence_heatmaps(scores_df, paths["phase_a_plots"], logger)
    else:
        logger.warning(
            "No embedding scores written. Causes: (1) umap-learn/sklearn missing, "
            "(2) no matching answers for each level_run_id under answers_dir, "
            "(3) activation row counts != answer counts. "
            "Use the same config.yaml paths and dataset as pipeline.py (e.g. json_levels -> …/levels/)."
        )


if __name__ == "__main__":
    main()
