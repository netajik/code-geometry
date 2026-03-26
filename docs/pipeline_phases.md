# Pipeline and manifold analysis (code generation)

**Data generation first** (GPU): fixed prompts, activations at a chosen layer position, then generation and test-based evaluation. **Then CPU analysis**: error structure, low-dimensional views of activations (Phase A), linear subspaces / variance structure (Phase C). Run order matters—each step reads outputs from the previous one.

**Vocabulary:** **Level** (curriculum difficulty tier), **population** (correct / wrong / error_category). Artifacts use **`level_run_<id>.json`** / **`level_run_<id>_layer{L}.npy`**. Phase **D** uses stratified **k-fold** CV (`phase_d.cv_folds`) for LDA accuracy.

---

## Manifold analysis — how to use this pipeline well

1. **Fix the protocol** — Same decoding (e.g. greedy), same **prompt token position** for hooks, same test harness. Geometry is only comparable across runs if the forward pass up to the hook is identical for a given input.
2. **Stratify difficulty** — Use **Level** files (or separate **level_run_id** batches) so you report accuracy, errors, and geometry **per tier** (easy → hard). Manifold structure often changes when the task gets harder.
3. **Separate behavior from pictures** — Use **execution correctness** and **error_category** as primary labels. UMAP/t-SNE are for **hypothesis forming**; support claims with **divergence scores**, **norm profiles (correct vs wrong)**, and Phase C **eigenvalues + permutation nulls**.
4. **Check confounds** — Correlate (or stratify) **prompt length**, **number of tests**, and **level** with correctness before attributing separation to “understanding code.” Phase A/C outputs are easier to interpret when confounds are weak or controlled.
5. **Layer sweep** — Compare **early vs mid vs late** layers (`model.layers` in config). For code models, informative separation often peaks in mid-to-late layers; the heatmaps and norm profiles make that visible.
6. **Linear separability vs variance** — For **defensible** claims that correctness aligns with a **linear** direction in activation space, prioritize **Phase D (LDA + CV + shuffle null)** after nuisance control. Phase C summarizes **within-class** variance (anisotropy vs permutation nulls); use it as **supporting** evidence, not the primary separability test. **Fourier** is separate (layer-depth periodicity).

---

## Overview

| Stage | What it does | Script(s) | Needs GPU | Main outputs |
|-------|----------------|-----------|-----------|----------------|
| **Data generation** | Load problems → labels → model → activations at prompt → generate code → run tests → save answers | `pipeline.py` | Yes (recommended) | `labels/`, `activations/`, `answers/`, `plots/`, `logs/` |
| **Error analysis** | Summarize error kinds; plots from saved answers | `analysis.py` | No | `plots/`, `analysis_summary.json` |
| **Phase B: confounds** | Tables + JSON + Markdown: Cramér’s V, point-biserial, splits, multivariate logistic; pairwise Pearson on derived label columns, **R after OLS residualization** (`phase_b.residualize_wrt_columns`), wrong-only pairwise when enough errors, classified pairs, Spearman top-*K* (`correlation_matrices/*.csv`). No PNG heatmaps by default. | `phase_b_deconfounding.py` | No | `{data_root}/…/phase_b/` |
| **Phase A: embeddings** | UMAP/t-SNE on activations; correct/wrong and error-category views; divergence heatmaps | `phase_a_embeddings.py` | No | `{data_root}/…/phase_a/` + `{workspace}/…/plots/phase_a/` |
| **Phase A: summary** | Divergence lines + norm profiles (correct vs wrong) | `phase_a_analysis.py` | No | `{workspace}/…/plots/phase_a/` |
| **Phase C: subspaces** | Linear nuisance removal; economy SVD within correct/wrong; i.i.d. + stratified permutation nulls; bootstrap stability; scree + summary | `phase_c_subspaces.py` | No | `{data_root}/…/phase_c/`, `{workspace}/…/plots/phase_c/` |
| **Phase D: LDA** | Supervised correct vs wrong; CV accuracy + shuffle null p-value; LDA direction + projections | `phase_d_lda.py` | No | `{data_root}/…/phase_d/` |
| **Fourier** | Layer-axis low-frequency power of norms vs layer-order null | `fourier_screening.py` | No | `{data_root}/…/fourier/`, plots |

Paths live under `paths.workspace` and `paths.data_root` in `config.yaml`, inside a **dataset-named** subfolder (e.g. `custom`, `levels`, or a HuggingFace repo tail).

---

## 1. Data generation (`pipeline.py`)

**Input:** `config.yaml` → model, layers, generation settings, **dataset** (JSON, Level files, or HuggingFace).

**Steps (in order):**

1. **Load problems** — One list per **level_run_id** (e.g. `custom`, `level_01`, …).
2. **Labels** — Metadata per problem (`task_id`, `num_test_lines`, `entry_point`, …).
3. **Save datasets** — `{workspace}/{dataset}/labels/level_run_<id>.json`.
4. **Tokenizer + model**
5. **Activations** — Forward on **prompt only**; **last prompt token** hidden state per configured layer → `{data_root}/{dataset}/activations/level_run_<id>_layer{L}.npy`.
6. **Generation** — Greedy decode (`max_new_tokens` from config).
7. **Evaluation** — Runnable program (stub + body + tests), subprocess + timeout → **correct** or **error_category**.
8. **Save answers** — `{data_root}/{dataset}/answers/level_run_<id>.json`.
9. **Plots** — Accuracy, problem counts, activation norms (all samples + correct vs wrong) → `{workspace}/{dataset}/plots/`.

**Outcome:** Fixed activation tensors aligned row-wise with **correctness labels** for manifold / subspace analysis.

---

## 2. Error analysis (`analysis.py`)

**Input:** Saved **answers** and **labels**.

**Output:** Error breakdowns, stacked bars, heatmaps, `num_test_lines` vs correctness, pooled / per-level-run error distributions, accuracy vs test-line count and vs prompt length (quartiles), `analysis_summary.json` under `{workspace}/{dataset}/plots/` and related paths.

**Depends on:** Completed `pipeline.py` (answers on disk).

---

## 3. Phase B — label confounds (interpretation guardrails)

**Script:** `phase_b_deconfounding.py`.

**Input:** Saved **answers** (`level_run_<id>.json`) and **labels** (`level_run_<id>.json` under `{workspace}/{dataset}/labels/`) for prompt length and `num_test_lines`.

**Methods:** Contingency **correct × error_category** with **Cramér’s V** (high values are *expected* when the pipeline sets `error_category="correct"` for correct rows). **Point-biserial** *r* between boolean correctness and **prompt_len** / **num_test_lines**. Splits and multivariate logistic in `*_correct_vs_wrong_factors.md`. **Pairwise Pearson** on numeric label-derived columns (raw **R** and **R** after multivariate linear OLS residualization w.r.t. `phase_b.residualize_wrt_columns`), **wrong-only** pairwise block when enough errors, **classified pairs** JSON, **Spearman** top-*K* on residualized columns — outputs under `phase_b/correlation_matrices/` and `*_label_correlation_summary.md`.

**Output:** Per-run CSVs/JSON/Markdown, pooled `summary.json`, **`deconfounding_plan.json`**. No Phase B PNGs.

**Depends on:** Completed `pipeline.py` (answers + labels on disk). *Recommended* after `analysis.py`, before or alongside Phase A.

---

## 4. Phase A — visual / global geometry

**Scripts:** `phase_a_embeddings.py`, then `phase_a_analysis.py`.

**Input:** Activations `level_run_*_layer{L}.npy` + answers.

**Methods:** UMAP and t-SNE when **`phase_a.min_tsne_samples` ≤ n ≤ 2000** (default **30** — use **30+** tasks per level, e.g. `python scripts/pad_level_json_to_count.py`); color by **correct vs wrong** and by **error_category**; **divergence** between correct/wrong centroids in 2D (normalized by spread); heatmaps of divergence over **level_run × layer**.

**Output:** CSVs, pickles, `interestingness_scores.csv`; PNGs under `{workspace}/{dataset}/plots/phase_a/` (embeddings, category views, heatmaps). `phase_a_analysis.py` adds **linear CKA** heatmaps, **norm profile (all samples)**, **norm correct vs wrong**, and (if embeddings ran) **divergence_by_layer_*.png**.

**Depends on:** Data generation.

---

## 5. Phase C — linear subspaces within populations

**Script:** `phase_c_subspaces.py`.

**Input:** Same activations + answers as Phase A; **labels** JSON under `{workspace}/{dataset}/labels/` (for linear nuisance removal).

**Methods:** **Per-dimension linear nuisance removal** w.r.t. `phase_c.nuisance_columns` (same `lstsq` idea as Phase B) before slicing. Then **conditional covariance** of activations restricted to **correct** or **wrong**; **SVD** for dominant directions; **i.i.d. permutation null** on labels; **stratified permutation null** (shuffle within prompt-length quantiles; default on via `phase_c.stratified_permutation`); **bootstrap** relative std of the top eigenvalue² (`phase_c.bootstrap_stability`); **scree** plots. If labels are missing or malformed, Phase C logs a warning and uses raw activations for that run.

**Output:** Bases and eigenvalues under `{data_root}/{dataset}/phase_c/subspaces/`; `phase_c_results.csv` (includes `significant_stratified`, `bootstrap_rel_std`, …); PNGs under `{workspace}/{dataset}/plots/phase_c/` (scree, bar charts, significance vs layer).

**Depends on:** Data generation.

---

## 6. Phase D — supervised LDA (correct vs wrong)

**Script:** `phase_d_lda.py`.

**Input:** Activations + answers; labels JSON for nuisance removal (same as Phase C).

**Methods:** `StandardScaler` + **LDA** (`solver='svd'`). **Stratified k-fold** accuracy (`phase_d.cv_folds`); **shuffle null**: empirical p-value comparing observed CV accuracy to label-shuffled repeats (`phase_d.shuffle_null_n`). Nuisance columns default to `phase_c.nuisance_columns`.

**Output:** `{data_root}/{dataset}/phase_d/{level_run_id}/layer_{L}/` — `lda_coef_scaled_space.npy`, `lda_projection.npy`, `metrics.json`; pooled `phase_d_summary.csv`. Logs under `{workspace}/{dataset}/logs/phase_d_lda.log`.

**Depends on:** Data generation; enough rows per class (`phase_d.min_per_class`).

---

## 7. Fourier screening — layer-axis spectrum

**Script:** `fourier_screening.py`.

**Input:** Activations for all `model.layers` per `level_run_id`.

**Methods:** Per sample, **L2 norm** of the activation vector at each layer → signal across depth; **rFFT** along layers; mean low-band power (excluding DC). **Null:** same random **permutation of layer indices** applied to every row (`fourier.n_layer_permutations`); z-score and empirical p-value vs that null. Histogram PNGs are written under `plots/fourier/` when plotting runs.

**Output:** `{data_root}/{dataset}/fourier/{level_run_id}_fourier.json`, `fourier_summary.csv`, plots under `{workspace}/{dataset}/plots/fourier/`.

**Depends on:** ≥3 layers in config; enough samples (`fourier.min_samples`).

---

## Order to run

```bash
python pipeline.py --config config.yaml
python analysis.py --config config.yaml
python phase_b_deconfounding.py --config config.yaml
python phase_a_embeddings.py --config config.yaml
python phase_a_analysis.py --config config.yaml
python phase_c_subspaces.py --config config.yaml
python phase_d_lda.py --config config.yaml
python fourier_screening.py --config config.yaml
```

**Level files:** Extra JSON files only add groups (`level_01`, …); no separate pipeline. Set `dataset.run_levels: [1, 3]` to restrict which level indices run; default `null` runs all keys in `dataset.levels`.

More run detail: [RUN.md](../RUN.md). Datasets: [datasets.md](datasets.md). Difficulty tiers: [test_levels.md](test_levels.md).

---

## Figure inventory

| Location | Files (typical) |
|----------|-----------------|
| `{workspace}/{dataset}/plots/` | `accuracy_by_level_run.png`, `problems_per_level_run.png`, `activation_norm_profile.png`, `activation_norm_correct_wrong.png` |
| + `analysis.py` | `error_categories.png`, `error_category_heatmap.png`, `num_test_lines_vs_correctness.png`, `outcome_distribution_pooled.png`, `error_distribution_by_level_run.png`, `accuracy_vs_num_test_lines.png`, `accuracy_vs_prompt_length.png` |
| `{workspace}/{dataset}/plots/phase_a/` | `*_umap.png`, `*_umap_by_category.png`, `*_tsne.png`, `*_tsne_by_category.png`, `divergence_heatmap_*.png`, `divergence_by_layer_*.png`, `norm_profile_correct_wrong.png`, `norm_profile_all_samples.png`, `cka_layer_similarity_{level_run_id}.png` |
| `{workspace}/{dataset}/plots/phase_c/` | `scree_*_correct.png`, `scree_*_wrong.png`, `top_eigenvalue_bar_*.png`, `significance_fraction_by_layer.png` |
| `{workspace}/{dataset}/plots/fourier/` | `*_fourier_null_hist.png` |

`phase_a_embeddings.py --skip-plots` writes Phase A CSVs/pickles without generating embedding PNGs.
