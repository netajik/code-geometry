#!/usr/bin/env python3
"""Fourier screening — low-frequency power along the layer axis for activation norms.

For each problem, we form a 1D signal across layers (default: L2 norm of the residual-stream
vector per layer). A **concentrated** low-frequency spectrum vs layer index can indicate
smooth / coherent evolution across depth (as opposed to noise-like layer-to-layer changes).

**Null model:** apply the **same random permutation of layer indices** to every row (destroys
alignment of depth structure). Compare observed mean low-band power to the null distribution;
report z-score and empirical exceedance p-value.

Outputs: ``{data_root}/{dataset}/fourier/{level_run_id}_fourier.json`` and spectrum PNG under ``plots/fourier/`` when plotting runs.

Usage:
  python fourier_screening.py --config config.yaml
  python fourier_screening.py --config config.yaml --pilot
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Optional, Tuple
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

from geometry_common import (
    derive_geometry_paths,
    load_activations,
    load_answers,
    load_config,
    resolve_config_paths,
)

import re

_ACT_FILE_RE = re.compile(r"^level_run_(.+)_layer\d+$")


def setup_logging(workspace: Path) -> logging.Logger:
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fourier_code")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(
        log_dir / "fourier_screening.log", maxBytes=10_000_000, backupCount=3
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


def norm_matrix_from_layers(
    rid: str, layers: list, act_dir: Path, logger: logging.Logger
) -> Tuple[Optional[np.ndarray], int]:
    """Shape (n_samples, n_layers) of row L2 norms."""
    mats = []
    n0 = None
    for layer in layers:
        X = load_activations(rid, layer, act_dir)
        if X is None:
            logger.warning("Fourier %s: missing layer %s", rid, layer)
            return None, 0
        X = np.asarray(X, dtype=np.float64)
        if n0 is None:
            n0 = X.shape[0]
        elif X.shape[0] != n0:
            return None, 0
        mats.append(np.linalg.norm(X, axis=1))
    if not mats:
        return None, 0
    M = np.column_stack(mats)
    return M, n0


def mean_low_band_power(M: np.ndarray, k_low_max: int) -> float:
    """Mean over rows of sum of |rFFT|^2 for frequency bins 1..k_low_max (exclude DC)."""
    x = M - M.mean(axis=1, keepdims=True)
    L = x.shape[1]
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    k2 = min(k_low_max, spec.shape[1] - 1)
    if k2 < 1:
        return float("nan")
    return float(spec[:, 1 : k2 + 1].sum(axis=1).mean())


def main():
    parser = argparse.ArgumentParser(description="Fourier screening along layer axis")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--pilot", action="store_true", help="One level_run_id only")
    args = parser.parse_args()
    config_path = Path(args.config or Path(__file__).parent / "config.yaml")
    cfg = load_config(config_path)
    resolve_config_paths(cfg, config_path)
    paths = derive_geometry_paths(cfg)
    paths["fourier_dir"].mkdir(parents=True, exist_ok=True)
    paths["fourier_plots"].mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["workspace_dataset"])

    fc = cfg.get("fourier") or {}
    n_null = int(fc.get("n_layer_permutations", 400))
    k_low_max = int(fc.get("low_freq_bins_max", 3))
    min_samples = int(fc.get("min_samples", 20))

    seed = int(cfg.get("dataset", {}).get("seed", 42))
    rng = np.random.default_rng(seed + 401)
    layers = sorted(cfg["model"]["layers"])
    if len(layers) < 3:
        logger.error("Fourier needs at least 3 layers in config model.layers")
        return

    level_run_ids = discover_level_run_ids(paths["act_dir"])
    if not level_run_ids:
        logger.error("No activations found.")
        return
    if args.pilot:
        level_run_ids = level_run_ids[:1]

    summary_rows = []
    for rid in level_run_ids:
        answers = load_answers(rid, paths["answers_dir"])
        n_expect = len((answers or {}).get("results") or [])
        M, n = norm_matrix_from_layers(rid, layers, paths["act_dir"], logger)
        if M is None or n < min_samples:
            logger.warning("Fourier %s: skip (bad matrix or n=%s)", rid, n)
            continue
        if n_expect and n_expect != n:
            logger.warning("Fourier %s: answers n=%s vs activations n=%s", rid, n_expect, n)

        obs = mean_low_band_power(M, k_low_max)
        L = M.shape[1]
        null_stats = []
        for _ in range(n_null):
            perm = rng.permutation(L)
            Msh = M[:, perm]
            null_stats.append(mean_low_band_power(Msh, k_low_max))
        null_arr = np.array([s for s in null_stats if np.isfinite(s)], dtype=np.float64)
        if null_arr.size < 10:
            continue
        mu_n = float(np.mean(null_arr))
        sig_n = float(np.std(null_arr)) + 1e-20
        z = float((obs - mu_n) / sig_n)
        p_emp = float((1 + np.sum(null_arr >= obs - 1e-12)) / (1 + null_arr.size))

        record = {
            "level_run_id": rid,
            "n_samples": int(n),
            "n_layers": int(L),
            "layers": [int(x) for x in layers],
            "scalar": "l2_row_norm",
            "low_freq_bins_max": k_low_max,
            "observed_lowfreq_power_mean": obs,
            "null_mean": mu_n,
            "null_std": float(np.std(null_arr)),
            "z_score_vs_layer_shuffle": z,
            "empirical_p_value_high_power": p_emp,
            "n_layer_permutations": n_null,
            "note": (
                "High z or low p suggests layer-axis spectrum is unlike random layer orderings; "
                "interpret together with Phase B confounds."
            ),
        }
        out_json = paths["fourier_dir"] / f"{rid}_fourier.json"
        with open(out_json, "w") as f:
            json.dump(record, f, indent=2)
        summary_rows.append(record)
        logger.info("Fourier %s: obs=%.4f z=%.2f p_emp=%.4f", rid, obs, z, p_emp)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(null_arr, bins=40, density=True, alpha=0.75, color="steelblue")
            ax.axvline(obs, color="darkred", lw=2, label="observed")
            ax.set_title(f"Fourier null ({rid}): low-band power")
            ax.legend()
            fig.tight_layout()
            fig.savefig(paths["fourier_plots"] / f"{rid}_fourier_null_hist.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.debug("Fourier plot skip: %s", e)

    if summary_rows:
        import pandas as pd

        pd.DataFrame(summary_rows).to_csv(paths["fourier_dir"] / "fourier_summary.csv", index=False)


if __name__ == "__main__":
    main()
