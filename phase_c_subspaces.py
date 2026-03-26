#!/usr/bin/env python3
"""Phase C — Concept subspace identification for code: correct/wrong and error_category.

Uses conditional covariance + SVD to find linear subspaces that encode correct vs wrong
and error_category. Saves basis matrices, eigenvalues, and projections.

**Validation:** i.i.d. label permutation null (default), **stratified** permutation
(shuffles labels within prompt-length quantiles to preserve length marginals), and
**bootstrap** relative std of the top eigenvalue² on the same slice (stability).
See ``phase_c`` keys in ``config.yaml``.

Usage:
  python phase_c_subspaces.py --config config.yaml
  python phase_c_subspaces.py --config config.yaml --pilot   # one level_run_id, one layer
"""

import argparse
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from geometry_common import (
    build_nuisance_matrix_for_activations,
    derive_geometry_paths,
    load_activations,
    load_answers,
    load_config,
    load_labels_dataset,
    prompt_lengths_from_labels,
    residualize_activations_linear_nuisance,
    resolve_config_paths,
)

MIN_POPULATION = 20
DEFAULT_N_PERMUTATIONS = 50  # was 200; full d×d SVD was O(d³) — use config phase_c.n_permutations for more
PERM_ALPHA = 0.01
CUMVAR_THRESHOLD = 0.95

# Linear nuisance columns for activation residualization (must match labels JSON fields).
DEFAULT_NUISANCE_COLUMNS = ["prompt_len", "num_test_lines"]

# level_run_level_01_layer8.npy -> level_run_id level_01 (not "level"); parse with regex.
_ACT_FILE_RE = re.compile(r"^level_run_(.+)_layer\d+$")


def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_c_code")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_c_subspaces.log", maxBytes=10_000_000, backupCount=3)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def discover_level_run_ids(act_dir):
    ids = set()
    for f in Path(act_dir).glob("level_run_*_layer*.npy"):
        m = _ACT_FILE_RE.match(f.stem)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def get_concept_series(answers_data, concept="correct"):
    results = answers_data.get("results", [])
    if concept == "correct":
        return pd.Series([r.get("correct", False) for r in results])
    if concept == "error_category":
        return pd.Series([r.get("error_category", "unknown") for r in results])
    return None


def centered_slice(X, labels, value, min_pop=None):
    """Rows of X where labels == value, row-centered. Returns (n, d) or None."""
    mp = MIN_POPULATION if min_pop is None else min_pop
    mask = labels == value
    if mask.sum() < mp:
        return None
    Y = X[mask].astype(np.float64)
    Y = Y - Y.mean(axis=0)
    n = Y.shape[0]
    if n < 2:
        return None
    return Y


def svd_subspace_from_centered_Y(Y, cumvar_threshold=CUMVAR_THRESHOLD):
    """Subspace of covariance C = Y.T @ Y / (n-1) without forming d×d matrix.

    For n << d (typical), economy SVD of Y is O(n²d) instead of O(d³).
    Matches previous statistics: eigenvalues λ_i = s_i²/(n-1); test stat uses λ_max²
    (same as old ``(S[0] from svd(C))**2`` when S were eigenvalues of C).

    Returns
    -------
    basis : (d, k) eigenvectors (columns)
    singvals : (k,) singular values of C (= eigenvalues λ_i for PSD C)
    cumvar : (k,) cumulative fraction using λ_i² / sum(λ_j²) (same weighting as old code)
    """
    n, _d = Y.shape
    denom = max(n - 1, 1)
    _, s, Vt = np.linalg.svd(Y, full_matrices=False)
    lam = (s.astype(np.float64) ** 2) / denom
    if lam.size == 0:
        return None, None, None
    var_weights = lam**2
    total_w = var_weights.sum() + 1e-20
    cumvar = np.cumsum(var_weights) / total_w
    k = int(np.searchsorted(cumvar, cumvar_threshold) + 1)
    k = min(max(k, 1), len(lam))
    basis = Vt.T[:, :k]
    singvals = lam[:k]
    return basis, singvals, cumvar[:k]


def top_eigenvalue_sq_stat(Y):
    """Scalar λ_max² for covariance of centered Y (permutation null); fast, no U/V."""
    n = Y.shape[0]
    if n < 2:
        return None
    s = np.linalg.svd(Y, full_matrices=False, compute_uv=False)
    lam0 = (float(s[0]) ** 2) / max(n - 1, 1)
    return lam0**2


def save_scree_plot(eval_s2, cumvar, out_path, title):
    """Eigenvalue spectrum + cumulative variance (subspace diagnostic)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_s2 = np.asarray(eval_s2, dtype=float)
    cumvar = np.asarray(cumvar, dtype=float)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    ax0.plot(np.arange(1, len(eval_s2) + 1), eval_s2, marker="o", markersize=3)
    ax0.set_yscale("log")
    ax0.set_xlabel("Component")
    ax0.set_ylabel("Eigenvalue (log)")
    ax1.plot(np.arange(1, len(cumvar) + 1), cumvar, marker="o", markersize=3, color="darkgreen")
    ax1.axhline(CUMVAR_THRESHOLD, color="gray", linestyle="--", alpha=0.7, label=f"threshold={CUMVAR_THRESHOLD}")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Cumulative variance frac")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phase_c_summary(df, plot_dir, logger):
    """Bar charts of top eigenvalue by layer / level_run_id; significance rate by layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_col = "level_run_id"
    for concept in sorted(df["concept"].unique()):
        sub = df[df["concept"] == concept]
        piv = sub.pivot(index="layer", columns=run_col, values="top_eval")
        fig, ax = plt.subplots(figsize=(10, 5))
        piv.sort_index().plot(kind="bar", ax=ax, alpha=0.88, width=0.82)
        ax.set_title(f"Phase C: top eigenvalue ({concept})")
        ax.set_ylabel("Top eigenvalue")
        ax.legend(title="level run", fontsize=8)
        ax.set_xlabel("Layer")
        fig.tight_layout()
        out = plot_dir / f"top_eigenvalue_bar_{concept}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    for concept in sorted(df["concept"].unique()):
        sub = df[df["concept"] == concept]
        rate = sub.groupby("layer")["significant"].mean()
        ax2.plot(rate.index, rate.values, marker="o", label=concept)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Fraction significant (perm. null)")
    ax2.set_title("Phase C: significance rate by layer")
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    fig2.tight_layout()
    out2 = plot_dir / "significance_fraction_by_layer.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    logger.info("Saved %s", out2)


def assign_stratum_codes(prompt_lens: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    """Quantile bins on prompt length for stratified permutation (preserves length marginals)."""
    if prompt_lens is None or len(prompt_lens) < max(20, 2 * n_bins):
        return None
    s = pd.Series(prompt_lens, dtype=float)
    try:
        q = min(n_bins, max(3, len(s) // 5))
        cat = pd.qcut(s, q=q, duplicates="drop")
    except (ValueError, TypeError):
        return None
    return cat.cat.codes.values.astype(np.int64)


def permutation_null(
    X, labels, value, n_perm, min_pop=None, rng: Optional[np.random.Generator] = None
):
    """Distribution of top eigenvalue² under i.i.d. label permutation null."""
    rng = rng or np.random.default_rng(42)
    top_evals: List[float] = []
    for _ in range(n_perm):
        shuf = pd.Series(rng.permutation(np.asarray(labels)))
        Y = centered_slice(X, shuf, value, min_pop=min_pop)
        if Y is not None:
            te = top_eigenvalue_sq_stat(Y)
            if te is not None:
                top_evals.append(te)
    return np.array(top_evals) if top_evals else np.array([0.0])


def permutation_null_stratified(
    X,
    labels,
    value,
    stratum_codes: np.ndarray,
    n_perm: int,
    min_pop: Optional[int],
    rng: np.random.Generator,
):
    """Permutation null with shuffles **within** each prompt-length stratum."""
    top_evals: List[float] = []
    lab_vals = np.asarray(labels)
    for _ in range(n_perm):
        shuf_vals = lab_vals.copy()
        for s in np.unique(stratum_codes):
            idx = np.where(stratum_codes == s)[0]
            if len(idx) < 2:
                continue
            shuf_vals[idx] = rng.permutation(shuf_vals[idx])
        shuf = pd.Series(shuf_vals)
        Y = centered_slice(X, shuf, value, min_pop=min_pop)
        if Y is not None:
            te = top_eigenvalue_sq_stat(Y)
            if te is not None:
                top_evals.append(te)
    return np.array(top_evals) if top_evals else np.array([0.0])


def bootstrap_top_eval_sq(
    X: np.ndarray,
    mask: np.ndarray,
    n_boot: int,
    frac: float,
    rng: np.random.Generator,
    min_pop: int,
) -> Optional[Dict[str, float]]:
    """Bootstrap stability of λ_max² on row-masked slice (e.g. correct-only). Lower rel_std ⇒ stabler."""
    idx_all = np.where(mask)[0]
    n = len(idx_all)
    if n < min_pop:
        return None
    m = max(min_pop, int(frac * n))
    stats: List[float] = []
    for _ in range(n_boot):
        sub = rng.choice(idx_all, size=m, replace=True)
        Y = X[sub].astype(np.float64)
        Y = Y - Y.mean(axis=0)
        if Y.shape[0] < 2:
            continue
        te = top_eigenvalue_sq_stat(Y)
        if te is not None:
            stats.append(te)
    if len(stats) < max(5, n_boot // 4):
        return None
    arr = np.array(stats, dtype=np.float64)
    mu = float(np.mean(arr))
    sig = float(np.std(arr))
    return {"mean": mu, "std": sig, "rel_std": float(sig / (mu + 1e-20))}


def main():
    parser = argparse.ArgumentParser(description="Phase C: concept subspaces for code")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--pilot", action="store_true", help="One level_run_id, one layer only")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=None,
        help="Override phase_c.n_permutations (fewer = faster, default from config or %s)" % DEFAULT_N_PERMUTATIONS,
    )
    args = parser.parse_args()
    config_path = args.config or Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)
    resolve_config_paths(cfg, config_path)
    paths = derive_geometry_paths(cfg)
    for p in (paths["subspaces_dir"], paths["summary_dir"], paths["phase_c_plots"]):
        p.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["workspace_dataset"])

    pc = cfg.get("phase_c") or {}
    n_perm = int(pc.get("n_permutations", DEFAULT_N_PERMUTATIONS))
    if args.n_permutations is not None:
        n_perm = int(args.n_permutations)
    n_perm = max(10, min(n_perm, 5000))
    min_pop = int(pc.get("min_population", MIN_POPULATION))
    min_pop = max(2, min_pop)
    logger.info(
        "Phase C: n_permutations=%s min_population=%s (set phase_c in config.yaml to tune)",
        n_perm,
        min_pop,
    )

    layers = cfg["model"]["layers"]
    level_run_ids = discover_level_run_ids(paths["act_dir"])
    if not level_run_ids:
        logger.error("No activations found; run pipeline first.")
        return

    if args.pilot:
        level_run_ids = level_run_ids[:1]
        layers = layers[:1] if layers else [4]

    act_nuis_cols = list(pc.get("nuisance_columns") or DEFAULT_NUISANCE_COLUMNS)
    seed = int(cfg.get("dataset", {}).get("seed", 42))
    rng_perm = np.random.default_rng(seed)
    rng_strat = np.random.default_rng(seed + 17)
    rng_boot = np.random.default_rng(seed + 29)
    do_strat = bool(pc.get("stratified_permutation", True))
    strat_bins = int(pc.get("stratified_n_bins", 10))
    do_boot = bool(pc.get("bootstrap_stability", True))
    n_boot = int(pc.get("bootstrap_n", 30))
    boot_frac = float(pc.get("bootstrap_frac", 0.85))

    rows = []
    for rid in level_run_ids:
        answers_data = load_answers(rid, paths["answers_dir"])
        if not answers_data:
            continue
        correct_ser = get_concept_series(answers_data, "correct")
        labels_ds = load_labels_dataset(paths["labels_dir"], rid)
        prompt_lens = (
            prompt_lengths_from_labels(labels_ds, len(correct_ser))
            if labels_ds is not None
            else None
        )
        strata = None
        if do_strat and prompt_lens is not None:
            strata = assign_stratum_codes(prompt_lens, strat_bins)

        for layer in layers:
            X = load_activations(rid, layer, paths["act_dir"])
            if X is None:
                continue
            X = np.asarray(X, dtype=np.float64)
            n, d = X.shape
            if n != len(correct_ser):
                continue

            if labels_ds is not None:
                Z = build_nuisance_matrix_for_activations(labels_ds, n, act_nuis_cols, logger)
                if Z is not None:
                    X = residualize_activations_linear_nuisance(X, Z, logger)
                    logger.info(
                        "Phase C %s layer %s: activations residualized w.r.t. %s",
                        rid,
                        layer,
                        act_nuis_cols,
                    )
                else:
                    logger.warning(
                        "Phase C %s layer %s: could not build nuisance matrix — using raw activations",
                        rid,
                        layer,
                    )
            else:
                logger.warning(
                    "Phase C %s layer %s: missing labels JSON under labels/ — using raw activations",
                    rid,
                    layer,
                )

            # Correct subspace
            Y_correct = centered_slice(X, correct_ser, True, min_pop=min_pop)
            if Y_correct is not None:
                U, S, cumvar = svd_subspace_from_centered_Y(Y_correct)
                if U is not None:
                    top_stat = float(S[0] ** 2)
                    null_evals = permutation_null(
                        X, correct_ser, True, n_perm, min_pop=min_pop, rng=rng_perm
                    )
                    thresh = np.percentile(null_evals, 100 * (1 - PERM_ALPHA)) if len(null_evals) else 0
                    sig = top_stat > thresh
                    if strata is not None:
                        null_strat = permutation_null_stratified(
                            X, correct_ser, True, strata, n_perm, min_pop, rng_strat
                        )
                        thresh_strat = (
                            np.percentile(null_strat, 100 * (1 - PERM_ALPHA))
                            if len(null_strat)
                            else 0
                        )
                        sig_strat = bool(top_stat > thresh_strat)
                    else:
                        thresh_strat = None
                        sig_strat = None
                    boot_info = None
                    if do_boot:
                        boot_info = bootstrap_top_eval_sq(
                            X,
                            np.asarray(correct_ser, dtype=bool),
                            n_boot,
                            boot_frac,
                            rng_boot,
                            min_pop,
                        )
                    boot_rel = float(boot_info["rel_std"]) if boot_info else float("nan")
                    dim = U.shape[1]
                    out_dir = paths["subspaces_dir"] / rid / f"layer_{layer}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / "correct_basis.npy", U)
                    np.save(out_dir / "correct_eigenvalues.npy", S ** 2)
                    eval_s2 = S ** 2
                    save_scree_plot(
                        eval_s2,
                        cumvar,
                        paths["phase_c_plots"] / f"scree_{rid}_layer{layer}_correct.png",
                        f"{rid} layer {layer} — correct subspace",
                    )
                    rows.append(
                        {
                            "level_run_id": rid,
                            "layer": layer,
                            "concept": "correct",
                            "dim": dim,
                            "top_eval": top_stat,
                            "null_threshold": thresh,
                            "significant": sig,
                            "null_threshold_stratified": thresh_strat,
                            "significant_stratified": sig_strat,
                            "bootstrap_rel_std": boot_rel,
                        }
                    )
                    logger.info(
                        "Level run %s layer %s correct: dim=%d top_eval=%.2e sig=%s sig_strat=%s",
                        rid,
                        layer,
                        dim,
                        top_stat,
                        sig,
                        sig_strat,
                    )

            # Wrong subspace (separate population)
            Y_wrong = centered_slice(X, correct_ser, False, min_pop=min_pop)
            if Y_wrong is not None:
                U, S, cumvar_w = svd_subspace_from_centered_Y(Y_wrong)
                if U is not None:
                    top_stat = float(S[0] ** 2)
                    null_evals = permutation_null(
                        X, correct_ser, False, n_perm, min_pop=min_pop, rng=rng_perm
                    )
                    thresh = np.percentile(null_evals, 100 * (1 - PERM_ALPHA)) if len(null_evals) else 0
                    sig = top_stat > thresh
                    if strata is not None:
                        null_strat = permutation_null_stratified(
                            X, correct_ser, False, strata, n_perm, min_pop, rng_strat
                        )
                        thresh_strat = (
                            np.percentile(null_strat, 100 * (1 - PERM_ALPHA))
                            if len(null_strat)
                            else 0
                        )
                        sig_strat = bool(top_stat > thresh_strat)
                    else:
                        thresh_strat = None
                        sig_strat = None
                    wrong_mask = ~np.asarray(correct_ser, dtype=bool)
                    boot_info = None
                    if do_boot:
                        boot_info = bootstrap_top_eval_sq(
                            X, wrong_mask, n_boot, boot_frac, rng_boot, min_pop
                        )
                    boot_rel = float(boot_info["rel_std"]) if boot_info else float("nan")
                    out_dir = paths["subspaces_dir"] / rid / f"layer_{layer}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / "wrong_basis.npy", U)
                    np.save(out_dir / "wrong_eigenvalues.npy", S ** 2)
                    eval_s2_w = S ** 2
                    save_scree_plot(
                        eval_s2_w,
                        cumvar_w,
                        paths["phase_c_plots"] / f"scree_{rid}_layer{layer}_wrong.png",
                        f"{rid} layer {layer} — wrong subspace",
                    )
                    rows.append(
                        {
                            "level_run_id": rid,
                            "layer": layer,
                            "concept": "wrong",
                            "dim": U.shape[1],
                            "top_eval": top_stat,
                            "null_threshold": thresh,
                            "significant": sig,
                            "null_threshold_stratified": thresh_strat,
                            "significant_stratified": sig_strat,
                            "bootstrap_rel_std": boot_rel,
                        }
                    )

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_path = paths["summary_dir"] / "phase_c_results.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Phase C complete. Summary: %s", summary_path)
        plot_phase_c_summary(summary_df, paths["phase_c_plots"], logger)
    else:
        logger.warning("No subspaces computed.")


if __name__ == "__main__":
    main()
