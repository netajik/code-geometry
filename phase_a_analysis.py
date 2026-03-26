#!/usr/bin/env python3
"""Phase A — Summary analysis: aggregate interestingness, report correct/wrong divergence.

Reads activations + answers and writes CKA heatmaps and norm profiles (all samples +
correct vs wrong). If `interestingness_scores.csv` exists (from `phase_a_embeddings.py`),
also prints divergence summary and saves divergence-by-layer line plots.

Usage:
  python phase_a_analysis.py --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _level_run_column(df):
    """Column name for level run in ``interestingness_scores.csv``."""
    if "level_run_id" not in df.columns:
        raise ValueError("interestingness_scores.csv must include column 'level_run_id'")
    return "level_run_id"


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


def linear_cka(X, Y):
    """Linear CKA between two sample matrices (same n rows). Range ~[0, 1]."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return float("nan")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xt_y = X.T @ Y
    xt_x = X.T @ X
    yt_y = Y.T @ Y
    hsic = np.sum(xt_y ** 2)
    denom = np.sqrt(np.sum(xt_x ** 2) * np.sum(yt_y ** 2)) + 1e-15
    return float(hsic / denom)


def plot_norm_profile_all_samples(cfg, dset, dr, plot_dir):
    """Mean L2 activation norm vs layer (all samples) — one curve per level_run_id."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = cfg.get("model", {}).get("layers") or []
    if not layers:
        return
    act_dir = dr / dset / "activations"
    ans_dir = dr / dset / "answers"
    if not act_dir.exists() or not ans_dir.exists():
        return
    level_run_ids = sorted(f.stem.replace("level_run_", "", 1) for f in ans_dir.glob("level_run_*.json"))
    if not level_run_ids:
        return
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for rid in level_run_ids:
        means = []
        ok = True
        for layer in layers:
            fp = act_dir / f"level_run_{rid}_layer{layer}.npy"
            if not fp.exists():
                ok = False
                break
            arr = np.load(fp, mmap_mode="r")
            means.append(float(np.linalg.norm(np.asarray(arr, dtype=np.float32), axis=1).mean()))
        if ok and means:
            ax.plot(layers, means, marker="o", label=rid)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 norm (all samples)")
    ax.set_title("Phase A: activation norm profile (all samples)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "norm_profile_all_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cka_layer_heatmaps(cfg, dset, dr, plot_dir):
    """Pairwise linear CKA between configured layers (representation similarity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = cfg.get("model", {}).get("layers") or []
    if len(layers) < 2:
        return
    act_dir = dr / dset / "activations"
    ans_dir = dr / dset / "answers"
    if not act_dir.exists() or not ans_dir.exists():
        return
    level_run_ids = sorted(f.stem.replace("level_run_", "", 1) for f in ans_dir.glob("level_run_*.json"))
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for rid in level_run_ids:
        mats = []
        n_ref = None
        bad = False
        for layer in layers:
            fp = act_dir / f"level_run_{rid}_layer{layer}.npy"
            if not fp.exists():
                bad = True
                break
            arr = np.asarray(np.load(fp, mmap_mode="r"), dtype=np.float64)
            if n_ref is None:
                n_ref = arr.shape[0]
            elif arr.shape[0] != n_ref:
                bad = True
                break
            mats.append(arr)
        if bad or len(mats) != len(layers):
            continue
        L = len(layers)
        cka = np.eye(L, dtype=np.float64)
        for i in range(L):
            for j in range(i + 1, L):
                v = linear_cka(mats[i], mats[j])
                cka[i, j] = cka[j, i] = v
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(cka, vmin=0.0, vmax=1.0, cmap="magma", aspect="equal")
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels([str(x) for x in layers])
        ax.set_yticklabels([str(x) for x in layers])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"Linear CKA — {rid}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plot_dir / f"cka_layer_similarity_{rid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_divergence_by_layer(df, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rc = _level_run_column(df)
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        fig, ax = plt.subplots(figsize=(10, 5))
        for rid in sorted(sub[rc].unique()):
            s2 = sub[sub[rc] == rid].sort_values("layer")
            ax.plot(s2["layer"], s2["divergence"], marker="o", label=rid)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Divergence (correct vs wrong)")
        ax.set_title(f"Phase A: divergence by layer ({method})")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / f"divergence_by_layer_{method}.png", dpi=150)
        plt.close(fig)


def plot_norm_profile_correct_wrong(cfg, dset, dr, plot_dir):
    """Mean L2 norm vs layer for correct vs wrong populations (loaded from disk)."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = cfg.get("model", {}).get("layers") or []
    if not layers:
        return
    act_dir = dr / dset / "activations"
    ans_dir = dr / dset / "answers"
    if not act_dir.exists() or not ans_dir.exists():
        return
    level_run_ids = sorted(
        f.stem.replace("level_run_", "", 1) for f in ans_dir.glob("level_run_*.json")
    )
    if not level_run_ids:
        return
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(level_run_ids)
    ncols = min(3, max(1, n_runs))
    nrows = math.ceil(n_runs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for idx, rid in enumerate(level_run_ids):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        path = ans_dir / f"level_run_{rid}.json"
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            ax.set_title(f"{rid} (empty)")
            continue
        correct_mask = np.array([bool(r.get("correct")) for r in results], dtype=bool)
        for flag, name, color in [(True, "correct", "green"), (False, "wrong", "red")]:
            m = correct_mask if flag else ~correct_mask
            if m.sum() == 0:
                continue
            means = []
            for layer in layers:
                fp = act_dir / f"level_run_{rid}_layer{layer}.npy"
                if not fp.exists():
                    means.append(np.nan)
                    continue
                arr = np.load(fp, mmap_mode="r")
                if arr.shape[0] != len(results):
                    means.append(np.nan)
                    continue
                means.append(float(np.linalg.norm(arr[m], axis=1).mean()))
            ax.plot(layers, means, marker="o", label=name, color=color)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean L2 norm")
        ax.set_title(rid)
        ax.legend(fontsize=8)
    for j in range(len(level_run_ids), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("Phase A: activation norm (correct vs wrong)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "norm_profile_correct_wrong.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase A summary analysis")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)
    resolve_config_paths(cfg, config_path)
    dr = Path(cfg["paths"]["data_root"])
    ws = Path(cfg["paths"]["workspace"])
    dset = get_dataset_name(cfg)
    phase_a_data = dr / dset / "phase_a"
    plot_dir = ws / dset / "plots" / "phase_a"
    plot_dir.mkdir(parents=True, exist_ok=True)
    act_dir = dr / dset / "activations"
    ans_dir = dr / dset / "answers"
    if not act_dir.exists() or not ans_dir.exists() or not list(ans_dir.glob("level_run_*.json")):
        print("No activations/answers found; run pipeline.py first.", file=sys.stderr)
        sys.exit(1)

    print("Saving Phase A geometry plots (CKA, norm profiles)...")
    plot_norm_profile_all_samples(cfg, dset, dr, plot_dir)
    plot_norm_profile_correct_wrong(cfg, dset, dr, plot_dir)
    plot_cka_layer_heatmaps(cfg, dset, dr, plot_dir)

    scores_path = phase_a_data / "interestingness_scores.csv"
    if scores_path.exists():
        df = pd.read_csv(scores_path)
        rc = _level_run_column(df)
        print("=== Phase A Summary (Code Geometry) ===")
        print(df.groupby([rc, "method"])["divergence"].agg(["mean", "max", "count"]).to_string())
        print("\nMax correct vs wrong divergence by (level_run_id, layer):")
        umap = df[df["method"] == "umap"]
        if not umap.empty:
            idx = umap["divergence"].idxmax()
            row = umap.loc[idx]
            print(f"  {row[rc]} layer {row['layer']}: divergence = {row['divergence']:.3f}")
        plot_divergence_by_layer(df, plot_dir)
    else:
        print(
            "(No interestingness_scores.csv — run phase_a_embeddings.py for UMAP/t-SNE divergence line plots.)",
            file=sys.stderr,
        )

    print(f"Plots under {plot_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
