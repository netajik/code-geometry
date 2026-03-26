#!/usr/bin/env python3
"""Phase D — Supervised linear discriminants (LDA) for correct vs wrong in activation space.

Complements Phase C (unsupervised variance within correct/wrong): LDA finds the direction
that best *separates* labels. Includes:

  - Stratified k-fold accuracy (default 5 folds, ``dataset.seed`` for splits)
  - **Shuffle null:** empirical p-value for CV accuracy vs label-shuffled repeats
  - Same linear nuisance removal on activations as Phase C (when labels JSON exists)

Outputs under ``{data_root}/{dataset}/phase_d/{level_run_id}/layer_{L}/``:
  ``lda_coef.npy``, ``lda_projection.npy`` (n×1 scores on full data), ``metrics.json``.

Usage:
  python phase_d_lda.py --config config.yaml
  python phase_d_lda.py --config config.yaml --pilot
"""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geometry_common import (
    derive_geometry_paths,
    load_activations,
    load_answers,
    load_config,
    load_labels_dataset,
    prepare_activations_like_phase_c,
    resolve_config_paths,
)

# Match phase_c activation file naming
import re

_ACT_FILE_RE = re.compile(r"^level_run_(.+)_layer\d+$")


def setup_logging(workspace: Path) -> logging.Logger:
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_d_code")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(
        log_dir / "phase_d_lda.log", maxBytes=10_000_000, backupCount=3
    )
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def discover_level_run_ids(act_dir: Path):
    ids = set()
    for f in Path(act_dir).glob("level_run_*_layer*.npy"):
        m = _ACT_FILE_RE.match(f.stem)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def lda_cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> float:
    """Mean stratified k-fold accuracy; binary y required."""
    if np.unique(y).size < 2:
        return float("nan")
    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=False)),
        ]
    )
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        scores.append(float(pipe.score(X[test_idx], y[test_idx])))
    return float(np.mean(scores))


def shuffle_null_p_value(
    X: np.ndarray,
    y: np.ndarray,
    observed_acc: float,
    n_shuffle: int,
    n_splits: int,
    seed: int,
) -> tuple[float, list]:
    """Empirical p-value: fraction of shuffled-label CV accuracies >= observed."""
    rng = np.random.default_rng(seed + 91)
    null_scores = []
    for _ in range(n_shuffle):
        y_s = rng.permutation(y)
        acc = lda_cv_accuracy(X, y_s, n_splits, seed=seed + rng.integers(1_000_000))
        if np.isfinite(acc):
            null_scores.append(acc)
    if not null_scores:
        return float("nan"), []
    ge = sum(1 for a in null_scores if a >= observed_acc - 1e-9)
    p = (1 + ge) / (1 + len(null_scores))
    return float(p), null_scores


def main():
    parser = argparse.ArgumentParser(description="Phase D: LDA discriminants for code geometry")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--pilot", action="store_true", help="One level_run_id, one layer")
    args = parser.parse_args()
    config_path = Path(args.config or Path(__file__).parent / "config.yaml")
    cfg = load_config(config_path)
    resolve_config_paths(cfg, config_path)
    paths = derive_geometry_paths(cfg)
    paths["phase_d_dir"].mkdir(parents=True, exist_ok=True)
    paths["phase_d_plots"].mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["workspace_dataset"])

    pd_cfg = cfg.get("phase_d") or {}
    nuisance_cols = list(
        pd_cfg.get("nuisance_columns") or cfg.get("phase_c", {}).get("nuisance_columns") or ["prompt_len", "num_test_lines"]
    )
    n_splits = int(pd_cfg.get("cv_folds", 5))
    n_shuffle = int(pd_cfg.get("shuffle_null_n", 80))
    min_per_class = int(pd_cfg.get("min_per_class", 15))

    seed = int(cfg.get("dataset", {}).get("seed", 42))
    layers = cfg["model"]["layers"]
    level_run_ids = discover_level_run_ids(paths["act_dir"])
    if not level_run_ids:
        logger.error("No activations found; run pipeline first.")
        return
    if args.pilot:
        level_run_ids = level_run_ids[:1]
        layers = layers[:1] if layers else [8]

    summary_rows = []
    for rid in level_run_ids:
        answers = load_answers(rid, paths["answers_dir"])
        if not answers:
            continue
        results = answers.get("results") or []
        y_bool = np.array([bool(r.get("correct", False)) for r in results], dtype=bool)
        if y_bool.sum() < min_per_class or (~y_bool).sum() < min_per_class:
            logger.warning("Phase D %s: skip (need ≥%s per class)", rid, min_per_class)
            continue
        y = y_bool.astype(np.int32)
        labels_ds = load_labels_dataset(paths["labels_dir"], rid)

        for layer in layers:
            Xraw = load_activations(rid, layer, paths["act_dir"])
            if Xraw is None:
                continue
            X = prepare_activations_like_phase_c(
                np.asarray(Xraw, dtype=np.float64), labels_ds, nuisance_cols, logger
            )
            if X.shape[0] != len(y):
                logger.warning("Phase D %s L%s: n mismatch", rid, layer)
                continue

            observed = lda_cv_accuracy(X, y, n_splits, seed)
            if not np.isfinite(observed):
                logger.warning("Phase D %s L%s: skip (CV not defined)", rid, layer)
                continue
            p_shuf, null_scores = shuffle_null_p_value(
                X, y, observed, n_shuffle, n_splits, seed
            )

            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=False)),
                ]
            )
            pipe.fit(X, y)
            lda = pipe.named_steps["lda"]
            coef = np.asarray(lda.coef_, dtype=np.float64).ravel()
            proj = np.asarray(pipe.decision_function(X), dtype=np.float64).ravel()

            out_dir = paths["phase_d_dir"] / rid / f"layer_{layer}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "lda_coef_scaled_space.npy", coef)
            np.save(out_dir / "lda_projection.npy", proj.astype(np.float64))

            metrics = {
                "level_run_id": rid,
                "layer": int(layer),
                "n_samples": int(len(y)),
                "n_correct": int(y_bool.sum()),
                "n_wrong": int((~y_bool).sum()),
                "cv_accuracy_mean": observed,
                "cv_folds": n_splits,
                "shuffle_null_n": n_shuffle,
                "shuffle_null_p_value": p_shuf,
                "shuffle_null_acc_mean": float(np.mean(null_scores)) if null_scores else None,
                "shuffle_null_acc_std": float(np.std(null_scores)) if null_scores else None,
                "explained_variance_ratio": (
                    np.asarray(lda.explained_variance_ratio_, dtype=float).tolist()
                    if getattr(lda, "explained_variance_ratio_", None) is not None
                    else None
                ),
            }
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            summary_rows.append(metrics)
            logger.info(
                "Phase D %s L%s: CV acc=%.3f shuffle p=%.4f",
                rid,
                layer,
                observed,
                p_shuf,
            )

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summ_path = paths["phase_d_dir"] / "phase_d_summary.csv"
        df.to_csv(summ_path, index=False)
        logger.info("Phase D summary: %s", summ_path)
    else:
        logger.warning("Phase D: no metrics written.")


if __name__ == "__main__":
    main()
