#!/usr/bin/env python3
"""Phase B — Label-level confounds for code geometry.

Quantifies how **correctness**, **error_category**, and cheap **dataset covariates**
(prompt length, number of test lines) co-vary per ``level_run_id``.  This stage does
**not** use activations; it prepares interpretation notes for Phase A (embeddings) and
Phase C (linear subspaces).

**One statistical stack (same estimators throughout):**
  - **Pearson** correlation for numeric columns (pairwise matrix).  Point-biserial
    vs ``correct`` is Pearson between a binary 0/1 column and a continuous column
    (implemented via ``scipy.stats.pointbiserialr``, which is that correlation).
  - **Linear nuisance removal:** ``numpy.linalg.lstsq`` — each non-nuisance column
    is regressed on an intercept plus standardized nuisance columns; residuals replace
    the column before recomputing Pearson *R* (multivariate extension when
    ``phase_b.residualize_wrt_columns`` lists several covariates, e.g. prompt length
    and test-line count together).
  - **Spearman** on the top-|r| residual pairs only as a **rank / monotonicity check**
    (``scipy.stats.spearmanr``); primary decisions stay Pearson-based.
  - **Cramér's V** from ``scipy.stats.chi2_contingency`` for categorical tables.
  - **Logistic regression** (``sklearn``) for **multivariate** adjustment of
    correctness vs several numeric factors in one model (always attempted; may record
    ``status: skipped`` when not estimable).

**Structural (definitional):** When ``correct`` is True, ``error_category`` is
``"correct"``. Strong association between those two is expected by construction.

**Wrong-only pairwise block:** Full pairwise correlation on incorrect rows only
(excludes constant ``correct`` column) whenever ``n_wrong`` ≥ ``min_rows_wrong_only_pairwise``.

**Correct vs wrong — “which factor?”** Writes ``{level_run_id}_correct_vs_wrong_factors.md``
and ``factor_attribution``: splits, Cohen's *d*, Mann–Whitney, and multivariate logistic.
All observational — not causal claims.

**Pairwise label correlation:** Pearson *R* raw and after linear nuisance removal,
classified pairs, Spearman top-*K* on residuals. Outputs under ``phase_b/correlation_matrices/``;
no default PNG heatmaps.

Outputs (under ``{data_root}/{dataset}/phase_b/`` only):

  - ``{level_run_id}_joint_table.csv`` — per-row labels + covariates
  - ``{level_run_id}_contingency_correct_vs_category.csv`` — 2×K counts
  - ``{level_run_id}_metrics.json`` — Cramér’s V, point-biserial *r*, severities, ``factor_attribution``
  - ``{level_run_id}_correct_vs_wrong_factors.md`` — human-readable correct vs wrong breakdown
  - ``summary.json`` — all runs
  - ``deconfounding_plan.json`` — short notes for Phase A/C readers
  - ``correlation_matrices/{level_run_id}_R_raw.csv`` / ``_R_residualized.csv``
  - ``{level_run_id}_classified_pairs.json``, ``{level_run_id}_spearman_top_k.json``
  - ``{level_run_id}_label_correlation_summary.md`` — short summary of matrices and pairs

Depends on: completed ``pipeline.py`` (``answers/`` and ``labels/`` on disk).

Usage:
  python phase_b_deconfounding.py --config config.yaml
  python phase_b_deconfounding.py --config config.yaml --level-run-id level_01
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config / paths (keep in sync with phase_a_embeddings.py / phase_c_subspaces.py)
# ---------------------------------------------------------------------------


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


def derive_paths(cfg: dict) -> Dict[str, Path]:
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
        "phase_b_data": dr / dset / "phase_b",
    }


def setup_logging(workspace: Path) -> logging.Logger:
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_b_code")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(
        log_dir / "phase_b_deconfounding.log",
        maxBytes=10_000_000,
        backupCount=3,
    )
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def discover_level_run_ids(answers_dir: Path) -> List[str]:
    if not answers_dir.is_dir():
        return []
    ids = []
    for f in sorted(answers_dir.glob("level_run_*.json")):
        stem = f.stem
        if stem.startswith("level_run_"):
            ids.append(stem[len("level_run_") :])
    return ids


def load_answers(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def load_labels_level_run(labels_dir: Path, level_run_id: str) -> Optional[dict]:
    path = labels_dir / f"level_run_{level_run_id}.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _is_code_stub_prompt(prompt: str) -> float:
    """1.0 if prompt looks like a code stub (def/class), else 0.0 (matches pipeline heuristic)."""
    s = (prompt or "").lstrip()
    return 1.0 if (s.startswith("def ") or s.startswith("class ")) else 0.0


def build_joint_dataframe(
    answers: dict,
    labels_dataset: Optional[dict],
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """One row per problem index, aligned with pipeline outputs."""
    results = answers.get("results") or []
    if not results:
        return None

    n = len(results)
    prompt_lens: List[Optional[int]] = [None] * n
    num_test_lines: List[Optional[int]] = [None] * n
    prompt_n_lines: List[Optional[float]] = [None] * n
    is_code_stub: List[Optional[float]] = [None] * n
    entry_point_str: List[str] = ["missing"] * n

    if labels_dataset and "problems" in labels_dataset:
        probs = labels_dataset["problems"]
        if len(probs) != n:
            logger.warning(
                "Labels/problems length %s != answers results %s; skipping label covariates",
                len(probs),
                n,
            )
        else:
            for i, entry in enumerate(probs):
                prompt = (entry.get("prompt") or "") if isinstance(entry, dict) else ""
                prompt_lens[i] = len(prompt)
                prompt_n_lines[i] = float(len(prompt.splitlines()) if prompt else 0)
                is_code_stub[i] = _is_code_stub_prompt(prompt)
                lab = entry.get("labels") or {}
                if isinstance(lab, dict):
                    num_test_lines[i] = int(lab.get("num_test_lines", 0) or 0)
                    entry_point_str[i] = str(lab.get("entry_point", "check") or "check")
    else:
        logger.warning("No labels JSON (or missing problems[]); prompt_len / num_test_lines will be NaN")

    rows = []
    for i, r in enumerate(results):
        rows.append(
            {
                "index": i,
                "correct": bool(r.get("correct", False)),
                "error_category": str(r.get("error_category", "unknown")),
                "task_id": str(r.get("task_id", "")),
                "prompt_len": prompt_lens[i],
                "num_test_lines": num_test_lines[i],
                "prompt_n_lines": prompt_n_lines[i],
                "is_code_stub": is_code_stub[i],
                "entry_point_str": entry_point_str[i],
            }
        )
    df = pd.DataFrame(rows)
    df["correct_f"] = df["correct"].astype(float)
    df["error_category_code"] = pd.factorize(df["error_category"].astype(str))[0].astype(np.float64)
    df["entry_point_code"] = pd.factorize(df["entry_point_str"].astype(str))[0].astype(np.float64)
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cramers_v(table: np.ndarray) -> float:
    """Cramér's V for a contingency table (chi-square association strength)."""
    table = np.asarray(table, dtype=float)
    if table.size == 0 or table.sum() == 0:
        return float("nan")
    try:
        chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    except ValueError:
        return float("nan")
    n = table.sum()
    r, k = table.shape
    if n <= 0 or r < 2 or k < 2:
        return float("nan")
    phi2 = max(0.0, chi2 / n)
    # bias-corrected phi2 (Bergsma 2013-style simplification used in sklearn docs)
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(phi2corr / denom))


def contingency_correct_vs_category(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """2 x K table: rows [incorrect, correct], columns sorted error_category."""
    cats = sorted(df["error_category"].astype(str).unique().tolist())
    row_correct = df["correct"].values
    # Rows: wrong (False), right (True)
    table = np.zeros((2, len(cats)), dtype=float)
    for j, c in enumerate(cats):
        m = df["error_category"].astype(str).values == c
        table[0, j] = ((~row_correct) & m).sum()
        table[1, j] = ((row_correct) & m).sum()
    row_labels = ["incorrect", "correct"]
    return table, row_labels, cats


def cramers_v_correct_vs_categorical(
    df: pd.DataFrame, col: str, max_categories: int = 25
) -> float:
    """Cramér's V for binary ``correct`` × categorical column ``col`` (empirical association)."""
    if col not in df.columns or len(df) < 5:
        return float("nan")
    sub = df[["correct", col]].dropna()
    if len(sub) < 5 or sub["correct"].nunique() < 2:
        return float("nan")
    n_u = int(sub[col].astype(str).nunique())
    if n_u < 2 or n_u > max_categories:
        return float("nan")
    try:
        tab = pd.crosstab(sub["correct"].astype(bool), sub[col].astype(str))
    except (TypeError, ValueError):
        return float("nan")
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    return cramers_v(tab.values.astype(float))


def cohens_d_two_groups(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges-style simple Cohen's d (mean diff / pooled SD)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if pooled < 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def continuous_factor_correct_vs_wrong(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Compare a numeric covariate between correct and incorrect rows."""
    sub = df[[col, "correct"]].dropna()
    if sub.empty:
        return {"factor": col, "status": "no_data"}
    ok = sub.loc[sub["correct"], col].values.astype(np.float64)
    bad = sub.loc[~sub["correct"], col].values.astype(np.float64)
    out: Dict[str, Any] = {
        "factor": col,
        "n_correct": int(len(ok)),
        "n_wrong": int(len(bad)),
    }
    if len(ok) < 2 or len(bad) < 2:
        out["status"] = "insufficient_both_groups"
        return out

    out["status"] = "ok"
    out["mean_correct"] = float(np.mean(ok))
    out["mean_wrong"] = float(np.mean(bad))
    out["median_correct"] = float(np.median(ok))
    out["median_wrong"] = float(np.median(bad))
    out["cohens_d_mean_correct_minus_wrong"] = cohens_d_two_groups(ok, bad)
    # Positive d => correct group has higher values on average than wrong group
    try:
        u_stat, p_mw = stats.mannwhitneyu(ok, bad, alternative="two-sided")
        out["mannwhitney_statistic"] = float(u_stat)
        out["mannwhitney_pvalue"] = float(p_mw)
    except ValueError:
        out["mannwhitney_pvalue"] = float("nan")

    diff = out["mean_correct"] - out["mean_wrong"]
    if abs(diff) < 1e-9:
        out["plain_language"] = f"{col}: similar means for correct vs wrong (no clear shift)."
    elif diff > 0:
        out["plain_language"] = (
            f"{col}: correct answers tend to have **higher** {col} than wrong "
            f"(mean {out['mean_correct']:.2f} vs {out['mean_wrong']:.2f})."
        )
    else:
        out["plain_language"] = (
            f"{col}: correct answers tend to have **lower** {col} than wrong "
            f"(mean {out['mean_correct']:.2f} vs {out['mean_wrong']:.2f})."
        )
    return out


def logistic_correct_vs_numeric_factors(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_rows: int = 15,
) -> Dict[str, Any]:
    """Multivariate logit P(correct=1) ~ standardized features. Always returns a dict."""
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 1:
        return {"status": "skipped", "reason": "no numeric feature columns in dataframe"}
    sub = df[cols + ["correct"]].dropna()
    if len(sub) < min_rows:
        return {
            "status": "skipped",
            "reason": f"need ≥{min_rows} complete rows (have {len(sub)})",
        }
    y = sub["correct"].astype(int).values
    if y.min() == y.max():
        return {"status": "skipped", "reason": "correct is constant in complete rows"}
    X = sub[cols].values.astype(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=0, solver="lbfgs")
    clf.fit(Xs, y)
    rows = []
    for i, name in enumerate(cols):
        beta = float(clf.coef_[0, i])
        rows.append(
            {
                "feature": name,
                "logistic_coef_standardized_X": beta,
                "odds_ratio_per_1sd_increase": float(np.exp(beta)),
            }
        )
    rows.sort(key=lambda r: abs(r["logistic_coef_standardized_X"]), reverse=True)
    return {
        "status": "ok",
        "model": "LogisticRegression on StandardScaler features; intercept not shown",
        "n_samples": int(len(sub)),
        "features_ranked_by_abs_coef": rows,
        "note": (
            "Coefficients are on standardized scales (1 SD). "
            "odds_ratio_per_1sd_increase is exp(coef). "
            "This is association conditional on other listed features, not causal."
        ),
    }


def write_correct_vs_wrong_factors_md(
    path: Path,
    level_run_id: str,
    factor_reports: List[Dict[str, Any]],
    logistic: Dict[str, Any],
    wrong_counts: Dict[str, int],
    n_total: int,
) -> None:
    lines = [
        f"# Correct vs wrong — factor breakdown (`{level_run_id}`)",
        "",
        "**Interpretation:** These are **statistical associations** on this run (task mix, "
        "difficulty, spurious correlations). They do **not** prove a single “cause” of wrong answers.",
        "",
        "## Continuous factors (correct vs incorrect rows)",
        "",
        "| Factor | n correct | n wrong | mean correct | mean wrong | Cohen's d (correct − wrong) | Mann–Whitney p |",
        "|--------|-----------|---------|--------------|------------|-------------------------------|----------------|",
    ]
    for fr in factor_reports:
        if fr.get("status") != "ok":
            lines.append(
                f"| {fr.get('factor')} | — | — | — | — | — | *{fr.get('status')}* |"
            )
            continue
        d = fr.get("cohens_d_mean_correct_minus_wrong", float("nan"))
        p = fr.get("mannwhitney_pvalue", float("nan"))
        lines.append(
            f"| {fr['factor']} | {fr['n_correct']} | {fr['n_wrong']} | "
            f"{fr['mean_correct']:.2f} | {fr['mean_wrong']:.2f} | "
            f"{d:.3f} | {p:.4g} |"
        )
    lines.append("")
    lines.append("### Plain language")
    for fr in factor_reports:
        if "plain_language" in fr:
            lines.append(f"- {fr['plain_language']}")
    lines.append("")
    lines.append("## Among **incorrect** rows only: `error_category`")
    if not wrong_counts:
        lines.append("- (no incorrect rows in this run)")
    else:
        nw = sum(wrong_counts.values())
        for cat, ct in sorted(wrong_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * ct / nw if nw else 0.0
            lines.append(f"- **{cat}**: {ct} ({pct:.1f}% of wrong)")
    lines.append("")
    lines.append("## Multivariate logistic (adjusted for other covariates)")
    if logistic.get("status") != "ok":
        lines.append(f"- Not fitted: {logistic.get('reason', '')}")
    else:
        lines.append(f"- {logistic.get('note', '')}")
        lines.append(f"- n = {logistic.get('n_samples')}")
        lines.append("")
        lines.append("| Feature | coef (1 SD scale) | OR per +1 SD |")
        lines.append("|---------|-------------------|--------------|")
        for r in logistic.get("features_ranked_by_abs_coef", []):
            lines.append(
                f"| {r['feature']} | {r['logistic_coef_standardized_X']:.4f} | "
                f"{r['odds_ratio_per_1sd_increase']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def point_biserial_bool_numeric(
    correct: np.ndarray, x: np.ndarray
) -> Tuple[float, float]:
    """Return (r, pvalue) or (nan, nan) if undefined.

    scipy.stats.pointbiserialr(x, y): first arg is dichotomous — we pass
    ``correct`` as 0/1 and the numeric covariate second (same convention as
    common usage for "correlation with correctness").
    """
    c = correct.astype(np.float64)
    xv = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(xv)
    if mask.sum() < 5 or np.unique(c[mask]).size < 2:
        return float("nan"), float("nan")
    r, p = stats.pointbiserialr(c[mask], xv[mask])
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Pairwise Pearson on label columns + residualization + classified pairs
# ---------------------------------------------------------------------------

# Default nuisances for pairwise residual Pearson (intersected with ``correlation_columns``).
DEFAULT_RESIDUALIZE_WRT_COLUMNS: List[str] = ["prompt_len", "num_test_lines"]

# Order matters: first columns are typical linear nuisances for ``residualize_wrt_columns``.
DEFAULT_CORRELATION_COLUMNS: List[str] = [
    "prompt_len",
    "prompt_n_lines",
    "num_test_lines",
    "is_code_stub",
    "entry_point_code",
    "error_category_code",
    "correct_f",
]

# Wrong-only / failure-focused pairwise block: omit ``correct_f`` (constant on this slice).
DEFAULT_CORRELATION_COLUMNS_WRONG_ONLY: List[str] = [
    c for c in DEFAULT_CORRELATION_COLUMNS if c != "correct_f"
]


def pairwise_pearson(values_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise Pearson r; NaN where n_valid < 10 or zero variance."""
    c = len(values_list)
    r_mat = np.eye(c, dtype=np.float64)
    n_mat = np.zeros((c, c), dtype=np.int64)
    for i in range(c):
        vi = values_list[i]
        for j in range(i + 1, c):
            vj = values_list[j]
            mask = ~(np.isnan(vi) | np.isnan(vj))
            n = int(mask.sum())
            n_mat[i, j] = n_mat[j, i] = n
            if n < 10:
                r_mat[i, j] = r_mat[j, i] = np.nan
                continue
            a = vi[mask].astype(np.float64)
            b = vj[mask].astype(np.float64)
            ac = a - a.mean()
            bc = b - b.mean()
            den = math.sqrt(float(ac @ ac) * float(bc @ bc))
            if den < 1e-12:
                r_mat[i, j] = r_mat[j, i] = 0.0
            else:
                rv = float((ac @ bc) / den)
                r_mat[i, j] = r_mat[j, i] = rv
    n_diag = [int(np.sum(~np.isnan(v))) for v in values_list]
    np.fill_diagonal(n_mat, n_diag)
    return r_mat, n_mat


def residualize_wrt_nuisance_columns(
    sub: pd.DataFrame,
    cols: List[str],
    nuisance_cols: List[str],
) -> List[np.ndarray]:
    """OLS residualize each non-nuisance column on intercept + standardized nuisances.

    Uses ``numpy.linalg.lstsq`` (same linear algebra recommended for Phase C nuisance
    removal). Nuisance columns are left unchanged.
    """
    values = [sub[c].values.astype(np.float64) for c in cols]
    nuis_j = [cols.index(c) for c in nuisance_cols if c in cols]
    if not nuis_j:
        return values
    n = len(sub)
    Z_parts: List[np.ndarray] = [np.ones(n, dtype=np.float64)]
    for j in nuis_j:
        u = values[j].copy()
        valid_u = np.isfinite(u)
        if valid_u.sum() < 10:
            return values
        mu = float(np.mean(u[valid_u]))
        sig = float(np.std(u[valid_u]))
        if sig < 1e-12:
            return values
        u_z = (u - mu) / sig
        u_z[~np.isfinite(u)] = np.nan
        Z_parts.append(u_z)
    Z = np.column_stack(Z_parts)
    out: List[np.ndarray] = []
    for j, v in enumerate(values):
        if j in nuis_j:
            out.append(np.asarray(v, dtype=np.float64, copy=True))
            continue
        vv = np.asarray(v, dtype=np.float64, copy=True)
        valid = np.isfinite(vv) & np.all(np.isfinite(Z), axis=1)
        if valid.sum() < 10:
            out.append(vv)
            continue
        try:
            beta, _, _, _ = np.linalg.lstsq(Z[valid], vv[valid], rcond=None)
        except np.linalg.LinAlgError:
            out.append(vv)
            continue
        pred = Z @ beta
        resid = vv - pred
        out.append(resid)
    return out


def classify_code_label_pair(a: str, b: str) -> str:
    """Coarse tags for label pair relationships in this pipeline."""
    x, y = sorted([a, b])
    prompt_shape = {"prompt_len", "prompt_n_lines"}
    if x in prompt_shape and y in prompt_shape:
        return "structural"
    if x == "is_code_stub" and y in prompt_shape:
        return "structural"
    if "correct_f" in (x, y) and "error_category_code" in (x, y):
        return "definitional"
    if "correct_f" in (x, y):
        return "outcome"
    return "none"


def compute_spearman_top_k_residual(
    values_resid: List[np.ndarray],
    names: List[str],
    r_resid: np.ndarray,
    k: int,
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[float, int, int, float]] = []
    c = len(names)
    for i in range(c):
        for j in range(i + 1, c):
            rv = r_resid[i, j]
            if np.isfinite(rv):
                pairs.append((abs(float(rv)), i, j, float(rv)))
    pairs.sort(reverse=True)
    out: List[Dict[str, Any]] = []
    for _, i, j, r_pearson in pairs[: max(0, k)]:
        vi, vj = values_resid[i], values_resid[j]
        mask = ~(np.isnan(vi) | np.isnan(vj))
        n = int(mask.sum())
        if n < 10:
            rho = None
        else:
            rho_f, _ = stats.spearmanr(vi[mask], vj[mask])
            rho = float(rho_f) if np.isfinite(rho_f) else None
        out.append(
            {
                "concept_a": names[i],
                "concept_b": names[j],
                "r_pearson_resid": r_pearson,
                "rho_spearman_resid": rho,
                "n_valid": n,
            }
        )
    return out


def classify_label_correlation_pairs(
    names: List[str],
    r_raw: np.ndarray,
    r_resid: np.ndarray,
    r_report: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    c = len(names)
    for i in range(c):
        for j in range(i + 1, c):
            raw = r_raw[i, j]
            res = r_resid[i, j]
            if not np.isfinite(raw) and not np.isfinite(res):
                continue
            mx = max(abs(float(raw)) if np.isfinite(raw) else 0.0,
                     abs(float(res)) if np.isfinite(res) else 0.0)
            if mx < r_report:
                continue
            cls = classify_code_label_pair(names[i], names[j])
            rows.append(
                {
                    "concept_a": names[i],
                    "concept_b": names[j],
                    "r_raw": float(raw) if np.isfinite(raw) else None,
                    "r_residualized": float(res) if np.isfinite(res) else None,
                    "classification": cls,
                    "note": {
                        "structural": "Driven by shared prompt text / format.",
                        "definitional": "correct vs error_category encoding (expected).",
                        "outcome": "Association with correctness (interpret with care).",
                        "none": "No canned label; investigate task mix.",
                    }.get(cls, ""),
                }
            )
    return rows


def run_pairwise_label_correlation(
    df: pd.DataFrame,
    phase_b_data: Path,
    level_run_id: str,
    pb_cfg: dict,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Pearson R, R after OLS residualization w.r.t. nuisance column(s), classified pairs, Spearman top-K."""
    phase_b_data.mkdir(parents=True, exist_ok=True)
    cols_cfg = pb_cfg.get("correlation_columns") or DEFAULT_CORRELATION_COLUMNS
    cols = [c for c in cols_cfg if c in df.columns]
    r_report = float(pb_cfg.get("r_report_threshold", 0.1))
    k_sp = int(pb_cfg.get("spearman_top_k", 20))

    raw_list = pb_cfg.get("residualize_wrt_columns") or DEFAULT_RESIDUALIZE_WRT_COLUMNS
    nuisance_cols = [str(c) for c in raw_list if c in cols]

    sub = df[cols].dropna()
    if len(sub) < 15 or len(cols) < 3:
        msg = "skip_pairwise_label_correlation: need ≥15 complete rows and ≥3 columns"
        logger.warning("%s (%s)", msg, level_run_id)
        return {"status": "skipped", "reason": msg}

    names = cols
    values = [sub[c].values.astype(np.float64) for c in cols]
    r_raw, _ = pairwise_pearson(values)

    if not nuisance_cols:
        logger.warning(
            "No nuisance columns from residualize_wrt_columns intersect correlation_columns; skipping residual matrix"
        )
        r_resid = np.full_like(r_raw, np.nan)
        values_resid = values
    else:
        values_resid = residualize_wrt_nuisance_columns(sub, cols, nuisance_cols)
        r_resid, _ = pairwise_pearson(values_resid)

    corr_dir = phase_b_data / "correlation_matrices"
    corr_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(r_raw, index=names, columns=names).to_csv(corr_dir / f"{level_run_id}_R_raw.csv")
    nuis_tag = "_".join(nuisance_cols) if nuisance_cols else "none"
    pd.DataFrame(r_resid, index=names, columns=names).to_csv(
        corr_dir / f"{level_run_id}_R_residualized_wrt_{nuis_tag}.csv"
    )

    classified = classify_label_correlation_pairs(names, r_raw, r_resid, r_report)
    spearman_rows = compute_spearman_top_k_residual(values_resid, names, r_resid, k_sp)

    with open(phase_b_data / f"{level_run_id}_classified_pairs.json", "w") as f:
        json.dump(classified, f, indent=2)
    with open(phase_b_data / f"{level_run_id}_spearman_top_k.json", "w") as f:
        json.dump(spearman_rows, f, indent=2)

    # Short markdown summary
    nuis_md = ", ".join(f"`{c}`" for c in nuisance_cols) if nuisance_cols else "—"
    lines = [
        f"# Pairwise label correlation (`{level_run_id}`)",
        "",
        f"**Nuisances removed (linear OLS):** {nuis_md} → see `R_residualized_wrt_{nuis_tag}.csv`.",
        f"**Pairs reported:** |r| ≥ {r_report} (raw or residualized).",
        "",
        "## Top residualized |Pearson| (Spearman check)",
        "",
        "| A | B | r_pearson_resid | rho_Spearman | n |",
        "|---|---|-----------------|--------------|---|",
    ]
    for row in spearman_rows[:15]:
        rho = row.get("rho_spearman_resid")
        rho_s = f"{rho:.4f}" if rho is not None else "—"
        lines.append(
            f"| {row['concept_a']} | {row['concept_b']} | {row['r_pearson_resid']:.4f} | "
            f"{rho_s} | {row['n_valid']} |"
        )
    lines.append("")
    lines.append("## Classified pairs (thresholded)")
    lines.append("")
    for row in classified[:40]:
        lines.append(
            f"- **{row['concept_a']}** vs **{row['concept_b']}**: "
            f"class={row['classification']}, r_raw={row.get('r_raw')}, "
            f"r_resid={row.get('r_residualized')} — {row.get('note', '')}"
        )
    if len(classified) > 40:
        lines.append(f"- … and {len(classified) - 40} more (see `_classified_pairs.json`)")
    (phase_b_data / f"{level_run_id}_label_correlation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    logger.info(
        "Phase B pairwise label correlation: %s pairs reported, Spearman rows=%s",
        len(classified),
        len(spearman_rows),
    )

    return {
        "status": "ok",
        "correlation_columns": names,
        "residualize_wrt_columns": nuisance_cols,
        "n_rows_used": int(len(sub)),
        "n_classified_pairs": len(classified),
        "n_spearman_rows": len(spearman_rows),
    }


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------


def analyze_level_run(
    level_run_id: str,
    df: pd.DataFrame,
    phase_b_data: Path,
    pb_cfg: dict,
    logger: logging.Logger,
) -> Dict[str, Any]:
    r_prompt = pb_cfg.get("confound_r_warn", 0.25)
    r_severe = pb_cfg.get("confound_r_severe", 0.45)

    n = len(df)
    acc = float(df["correct"].mean()) if n else 0.0

    table, row_labels, col_labels = contingency_correct_vs_category(df)
    v_ce = cramers_v(table)

    # Definitional check: all correct rows have category "correct"
    wrong_labels_when_correct = df.loc[df["correct"], "error_category"].astype(str).ne("correct").sum()
    structural_note = (
        "When correct=True, pipeline sets error_category='correct' (definitional). "
        "High Cramér's V for correct × error_category is expected."
    )

    metrics: Dict[str, Any] = {
        "level_run_id": level_run_id,
        "n": n,
        "accuracy": acc,
        "cramers_v_correct_vs_error_category": v_ce,
        "anomaly_correct_true_but_error_category_not_correct": int(wrong_labels_when_correct),
        "structural_note": structural_note,
    }

    wrong_df = df[~df["correct"]]
    # Empirical Cramér's V (not definitional): correctness × entry-point / task id mix.
    metrics["cramers_v_correct_vs_entry_point"] = cramers_v_correct_vs_categorical(
        df, "entry_point_str"
    )
    # task_id is often near-unique per row; skip unless a small set of distinct ids.
    metrics["cramers_v_correct_vs_task_id"] = cramers_v_correct_vs_categorical(
        df, "task_id", max_categories=25
    )

    # Numeric confounds (point-biserial: dichotomous correct vs continuous covariate)
    pb_cols = (
        "prompt_len",
        "num_test_lines",
        "prompt_n_lines",
        "is_code_stub",
        "entry_point_code",
    )
    for col in pb_cols:
        if col not in df.columns or df[col].notna().sum() < 5:
            metrics[f"point_biserial_correct_vs_{col}"] = {"r": None, "pvalue": None}
            metrics[f"confound_severity_{col}"] = "insufficient_data"
            continue
        r, p = point_biserial_bool_numeric(df["correct"].values, df[col].values)
        metrics[f"point_biserial_correct_vs_{col}"] = {"r": r, "pvalue": p}
        sev = "none"
        if np.isfinite(r) and abs(r) >= r_severe:
            sev = "severe"
        elif np.isfinite(r) and abs(r) >= r_prompt:
            sev = "moderate"
        metrics[f"confound_severity_{col}"] = sev

    # Wrong-only error mix
    if len(wrong_df) > 0:
        vc = wrong_df["error_category"].astype(str).value_counts().to_dict()
        metrics["wrong_only_error_category_counts"] = vc
    else:
        metrics["wrong_only_error_category_counts"] = {}

    # --- Correct vs wrong: which observable factors line up? (association, not causation) ---
    candidate_factors = (
        "prompt_len",
        "num_test_lines",
        "prompt_n_lines",
        "is_code_stub",
        "entry_point_code",
    )
    numeric_factors = tuple(
        c for c in candidate_factors if c in df.columns and df[c].notna().sum() >= 5
    )
    factor_reports = [continuous_factor_correct_vs_wrong(df, c) for c in numeric_factors]
    logistic_cols = [c for c in numeric_factors if c in df.columns]
    logistic = logistic_correct_vs_numeric_factors(df, logistic_cols)
    metrics["factor_attribution"] = {
        "disclaimer": (
            "Observational associations only. Wrong answers are *described* by error_category; "
            "prompt_len and num_test_lines are pre-generation covariates that may correlate with outcome."
        ),
        "continuous_correct_vs_wrong": factor_reports,
        "logistic_regression_standardized": logistic,
    }
    # Rank univariate |r| for a one-line hint
    r_rank: List[Tuple[str, float]] = []
    for col in pb_cols:
        pb = metrics.get(f"point_biserial_correct_vs_{col}") or {}
        rv = pb.get("r")
        if rv is not None and isinstance(rv, (int, float)) and np.isfinite(rv):
            r_rank.append((col, abs(float(rv))))
    r_rank.sort(key=lambda x: -x[1])
    if r_rank:
        top = r_rank[0]
        metrics["factor_attribution"]["strongest_univariate_by_abs_point_biserial"] = {
            "feature": top[0],
            "abs_r": top[1],
            "note": "Largest |point-biserial r| among numeric label factors (univariate only).",
        }
    else:
        metrics["factor_attribution"]["strongest_univariate_by_abs_point_biserial"] = None

    pairwise_corr = run_pairwise_label_correlation(df, phase_b_data, level_run_id, pb_cfg, logger)
    metrics["pairwise_label_correlation"] = pairwise_corr

    # Wrong-only sub-population: pairwise structure among incorrect rows (drops constant correct_f).
    min_wr = int(pb_cfg.get("min_rows_wrong_only_pairwise", 15))
    if len(wrong_df) >= min_wr:
        pb_wr = dict(pb_cfg)
        cols_wr = pb_wr.get("correlation_columns_wrong_only") or DEFAULT_CORRELATION_COLUMNS_WRONG_ONLY
        pb_wr["correlation_columns"] = cols_wr
        metrics["pairwise_label_correlation_wrong_only"] = run_pairwise_label_correlation(
            wrong_df, phase_b_data, f"{level_run_id}_wrong_only", pb_wr, logger
        )
    else:
        metrics["pairwise_label_correlation_wrong_only"] = {
            "status": "skipped",
            "reason": (
                f"n_wrong={len(wrong_df)} < min_rows_wrong_only_pairwise={min_wr} "
                "(wrong-only pairwise correlation needs enough errors)"
            ),
        }

    write_correct_vs_wrong_factors_md(
        phase_b_data / f"{level_run_id}_correct_vs_wrong_factors.md",
        level_run_id,
        factor_reports,
        logistic,
        metrics["wrong_only_error_category_counts"],
        n,
    )

    # Save joint CSV + contingency as CSV
    phase_b_data.mkdir(parents=True, exist_ok=True)
    df.to_csv(phase_b_data / f"{level_run_id}_joint_table.csv", index=False)
    tab_df = pd.DataFrame(table, index=row_labels, columns=col_labels)
    tab_df.to_csv(phase_b_data / f"{level_run_id}_contingency_correct_vs_category.csv")

    with open(phase_b_data / f"{level_run_id}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    v_ep = metrics.get("cramers_v_correct_vs_entry_point")
    v_ep_s = f"{v_ep:.3f}" if isinstance(v_ep, (int, float)) and np.isfinite(v_ep) else "nan"
    fa = metrics.get("factor_attribution") or {}
    su = fa.get("strongest_univariate_by_abs_point_biserial") or {}
    su_s = (
        f"{su.get('feature')}|r|={su.get('abs_r', 0):.3f}"
        if su.get("feature")
        else "—"
    )
    npc = pairwise_corr.get("n_classified_pairs", 0)
    wo = metrics.get("pairwise_label_correlation_wrong_only") or {}
    wo_s = wo.get("status", "?")
    wo_n = wo.get("n_classified_pairs", "—") if wo_s == "ok" else "—"
    logger.info(
        "Phase B %s: n=%s acc=%.3f | primary_factor=%s | pairwise_pairs=%s | "
        "wrong_only_pairwise=%s (classified=%s) | CramérV(entry×correct)=%s | "
        "CramérV(correct×error_cat)=%s (definitional sanity)",
        level_run_id,
        n,
        acc,
        su_s,
        npc,
        wo_s,
        wo_n,
        v_ep_s,
        f"{v_ce:.3f}" if np.isfinite(v_ce) else "nan",
    )

    return metrics


def build_deconfounding_plan(all_metrics: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    rec_global: List[str] = [
        "Phase C compares correct vs wrong linear structure; do not treat error_category as independent of correctness.",
        "If prompt_len or num_test_lines correlates with correctness, interpret Phase A separation partly as difficulty/coverage effects.",
        "Pairwise label correlation: phase_b/correlation_matrices/*_R_raw.csv and *_R_residualized_wrt_*.csv; nuisances in phase_b.residualize_wrt_columns.",
    ]
    runs_out: Dict[str, Any] = {}
    for m in all_metrics:
        rid = m["level_run_id"]
        notes: List[str] = []
        if m.get("anomaly_correct_true_but_error_category_not_correct", 0):
            notes.append(
                "Data anomaly: some rows have correct=True but error_category != 'correct' — inspect answers JSON."
            )
        for col in (
            "prompt_len",
            "num_test_lines",
            "prompt_n_lines",
            "is_code_stub",
            "entry_point_code",
        ):
            sev = m.get(f"confound_severity_{col}")
            if sev == "severe":
                notes.append(
                    f"Strong confound: correctness vs {col} (see point_biserial). Stratify or control before causal language."
                )
            elif sev == "moderate":
                notes.append(f"Moderate association: correctness vs {col} — mention as limitation in writeups.")

        fa = m.get("factor_attribution") or {}
        summary_line = None
        if fa.get("strongest_univariate_by_abs_point_biserial"):
            s = fa["strongest_univariate_by_abs_point_biserial"]
            summary_line = (
                f"Strongest univariate label factor vs correctness: {s.get('feature')} "
                f"(|r|≈{s.get('abs_r', 0):.3f}). See *_correct_vs_wrong_factors.md for splits + logistic."
            )
        runs_out[rid] = {
            "n": m.get("n"),
            "accuracy": m.get("accuracy"),
            "cramers_v_correct_vs_error_category": m.get("cramers_v_correct_vs_error_category"),
            "cramers_v_correct_vs_entry_point": m.get("cramers_v_correct_vs_entry_point"),
            "cramers_v_correct_vs_task_id": m.get("cramers_v_correct_vs_task_id"),
            "structural_note": m.get("structural_note"),
            "interpretation_notes": notes,
            "correct_vs_wrong_factor_summary": summary_line,
            "pairwise_label_correlation_wrong_only": m.get("pairwise_label_correlation_wrong_only"),
        }

    return {
        "version": 1,
        "dataset": dataset_name,
        "level_runs": runs_out,
        "recommendations_global": rec_global,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B: label confounds for code geometry")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument(
        "--level-run-id",
        default=None,
        help="Only analyze this level_run_id (default: all answer files)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config or Path(__file__).parent / "config.yaml")
    cfg = load_config(config_path)
    resolve_config_paths(cfg, config_path)
    paths = derive_paths(cfg)
    paths["phase_b_data"].mkdir(parents=True, exist_ok=True)

    logger = setup_logging(paths["workspace_dataset"])
    pb_cfg = cfg.get("phase_b") or {}

    logger.info("Phase B — label confounds (dataset=%s)", paths["dataset_name"])
    logger.info("Answers dir: %s", paths["answers_dir"])
    logger.info("Labels dir: %s", paths["labels_dir"])

    ids = discover_level_run_ids(paths["answers_dir"])
    if args.level_run_id:
        ids = [args.level_run_id] if args.level_run_id in ids else []
    if not ids:
        logger.error("No matching level_run_*.json under %s", paths["answers_dir"])
        return

    all_metrics: List[Dict[str, Any]] = []
    for rid in ids:
        ans_path = paths["answers_dir"] / f"level_run_{rid}.json"
        answers = load_answers(ans_path)
        if not answers:
            logger.warning("Missing or empty answers: %s", ans_path)
            continue
        labels_ds = load_labels_level_run(paths["labels_dir"], rid)
        df = build_joint_dataframe(answers, labels_ds, logger)
        if df is None or df.empty:
            logger.warning("No results[] for %s", rid)
            continue
        m = analyze_level_run(
            rid,
            df,
            paths["phase_b_data"],
            pb_cfg,
            logger,
        )
        all_metrics.append(m)

    if not all_metrics:
        logger.error("Phase B produced no metrics.")
        return

    summary = {
        "dataset": paths["dataset_name"],
        "n_level_runs": len(all_metrics),
        "runs": all_metrics,
    }
    with open(paths["phase_b_data"] / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plan = build_deconfounding_plan(all_metrics, paths["dataset_name"])
    with open(paths["phase_b_data"] / "deconfounding_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    logger.info("Phase B complete. Outputs under %s", paths["phase_b_data"])


if __name__ == "__main__":
    main()
