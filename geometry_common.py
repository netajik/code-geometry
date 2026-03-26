#!/usr/bin/env python3
"""Shared path resolution, I/O, and activation nuisance removal for Phases C / D / Fourier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config_paths(cfg: dict, config_path: Path) -> None:
    base = Path(config_path).resolve().parent
    for key in ("workspace", "data_root"):
        p = Path(cfg["paths"][key]).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        else:
            p = p.resolve()
        cfg["paths"][key] = str(p)


def get_dataset_name(cfg: dict) -> str:
    d = cfg.get("dataset", {})
    if d.get("output_name"):
        return str(d["output_name"])
    source = d.get("source", "json")
    if source == "json":
        return "custom"
    if source == "huggingface":
        repo = d.get("hf_repo", "")
        return repo.split("/")[-1] if repo else "hf"
    if source == "json_levels":
        return "levels"
    return "custom"


def derive_geometry_paths(cfg: dict) -> Dict[str, Path]:
    ws = Path(cfg["paths"]["workspace"])
    dr = Path(cfg["paths"]["data_root"])
    dset = get_dataset_name(cfg)
    ws_dset = ws / dset
    return {
        "workspace": ws,
        "workspace_dataset": ws_dset,
        "data_root": dr,
        "dataset_name": dset,
        "answers_dir": dr / dset / "answers",
        "act_dir": dr / dset / "activations",
        "labels_dir": ws_dset / "labels",
        "phase_c_data": dr / dset / "phase_c",
        "subspaces_dir": dr / dset / "phase_c" / "subspaces",
        "summary_dir": dr / dset / "phase_c" / "summary",
        "phase_c_plots": ws_dset / "plots" / "phase_c",
        "phase_d_dir": dr / dset / "phase_d",
        "phase_d_plots": ws_dset / "plots" / "phase_d",
        "fourier_dir": dr / dset / "fourier",
        "fourier_plots": ws_dset / "plots" / "fourier",
    }


def load_activations(level_run_id: str, layer: int, act_dir: Path) -> Optional[np.ndarray]:
    path = Path(act_dir) / f"level_run_{level_run_id}_layer{layer}.npy"
    if not path.exists():
        return None
    return np.load(path)


def load_answers(level_run_id: str, answers_dir: Path) -> Optional[dict]:
    path = Path(answers_dir) / f"level_run_{level_run_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_labels_dataset(labels_dir: Path, level_run_id: str) -> Optional[dict]:
    path = Path(labels_dir) / f"level_run_{level_run_id}.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def prompt_lengths_from_labels(labels_dataset: Optional[dict], n_rows: int) -> Optional[np.ndarray]:
    if not labels_dataset:
        return None
    probs = labels_dataset.get("problems") or []
    if len(probs) != n_rows:
        return None
    return np.array([len(str(p.get("prompt", ""))) for p in probs], dtype=np.int64)


def build_nuisance_matrix_for_activations(
    labels_dataset: Optional[dict],
    n_rows: int,
    nuisance_cols: list,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    if not labels_dataset or not nuisance_cols:
        return None
    probs = labels_dataset.get("problems") or []
    if len(probs) != n_rows:
        logger.warning(
            "Nuisance matrix: labels have %s problems, activations have %s rows — skip",
            len(probs),
            n_rows,
        )
        return None
    parts = []
    for c in nuisance_cols:
        if c == "prompt_len":
            parts.append(
                np.array([len(str(p.get("prompt", ""))) for p in probs], dtype=np.float64)
            )
        elif c == "num_test_lines":
            parts.append(
                np.array(
                    [float((p.get("labels") or {}).get("num_test_lines", 0)) for p in probs],
                    dtype=np.float64,
                )
            )
        else:
            logger.warning("Unknown nuisance column %r (use prompt_len, num_test_lines)", c)
            return None
    if not parts:
        return None
    return np.column_stack(parts)


def residualize_activations_linear_nuisance(
    X: np.ndarray, Z: np.ndarray, logger: logging.Logger
) -> np.ndarray:
    """Per dimension: OLS residual w.r.t. intercept + standardized Z columns."""
    n, d = X.shape
    if Z.shape[0] != n:
        return X
    pz = Z.shape[1]
    Zs = Z.astype(np.float64).copy()
    for j in range(pz):
        col = Zs[:, j]
        v = np.isfinite(col)
        if v.sum() < 10:
            return X
        mu = float(np.mean(col[v]))
        sig = float(np.std(col[v]))
        if sig < 1e-12:
            return X
        Zs[:, j] = (col - mu) / sig
        Zs[~np.isfinite(col), j] = np.nan
    Zd = np.column_stack([np.ones(n, dtype=np.float64), Zs])
    X_out = X.astype(np.float64).copy()
    min_valid = max(10, Zd.shape[1] + 2)
    for j in range(d):
        xcol = X_out[:, j]
        valid = np.isfinite(xcol) & np.all(np.isfinite(Zd), axis=1)
        if valid.sum() < min_valid:
            continue
        try:
            beta, _, _, _ = np.linalg.lstsq(Zd[valid], xcol[valid], rcond=None)
        except np.linalg.LinAlgError:
            continue
        X_out[:, j] = xcol - Zd @ beta
    return X_out


def prepare_activations_like_phase_c(
    X: np.ndarray,
    labels_ds: Optional[dict],
    nuisance_cols: list,
    logger: logging.Logger,
) -> np.ndarray:
    """Apply the same linear nuisance removal as Phase C (when labels exist)."""
    n = X.shape[0]
    if labels_ds is None:
        logger.warning("prepare_activations: no labels — raw X")
        return np.asarray(X, dtype=np.float64)
    Z = build_nuisance_matrix_for_activations(labels_ds, n, nuisance_cols, logger)
    if Z is None:
        logger.warning("prepare_activations: could not build Z — raw X")
        return np.asarray(X, dtype=np.float64)
    return residualize_activations_linear_nuisance(np.asarray(X, dtype=np.float64), Z, logger)
