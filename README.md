# Code Geometry

**Manifold-style interpretability for code generation:** extract residual-stream activations from a code-capable LLM on coding tasks, evaluate correctness (e.g. tests), attach error categories, and analyze geometry with staged phases (embeddings, confound checks, subspaces; plus experimental LDA and Fourier summaries).

**This repository does not implement or replicate the REMA paper** (or any single external framework as a fixed spec). Phase definitions, statistics, and nulls are defined in the scripts and in `docs/pipeline_phases.md`. For **background reading** on manifolds, REMA, and related work, see [REFERENCES.md](REFERENCES.md)—citations there are **not** a claim that this codebase follows those methods end-to-end.

**Terminology:** *Level* = curriculum difficulty tier; *population* here means outcome groups (correct / wrong / error_category), not train/test splits. **`level_run_id`** names one saved batch (e.g. `level_01`); artifacts use **`level_run_<id>.json`** and **`level_run_<id>_layer{L}.npy`**. Set **`dataset.level_run_id`** to override the default batch id; **`dataset.hf_split`** selects the HuggingFace subset. Phase D uses stratified k-fold CV for LDA where applicable; avoid duplicating k-fold logic inside Phase C.

## Research question

When a code model gets a task wrong, how do **internal representations** differ from correct runs—and can we separate **genuine structure** from **confounds** (prompt length, number of test lines, etc.)? We study activation geometry for correct vs incorrect generations and error categories (syntax, logic, timeout, garbage).

## Pipeline status

| Stage | Status | Script | Output |
|-------|--------|--------|--------|
| Data generation | Active | `pipeline.py`, `analysis.py` | Activations (`level_run_<id>_layer{L}.npy`), labels, answers, error analysis |
| Phase B: Label confounds | Active | `phase_b_deconfounding.py` | Cramér’s V, point-biserial, splits/logistic, pairwise Pearson + residualized R, classified pairs, Spearman top-*K*; `deconfounding_plan.json`; Phase B is tables/JSON/Markdown (no Phase B PNGs) |
| Phase A: Visual reconnaissance | Active | `phase_a_embeddings.py`, `phase_a_analysis.py` | UMAP/t-SNE embeddings + interestingness; `phase_a_analysis.py` adds CKA and norm profiles (can run after pipeline even before embedding CSVs exist) |
| Phase C: Concept subspaces | Active | `phase_c_subspaces.py` | Concept subspaces (correct/wrong, error_category), significance, projections |
| Phase D: LDA | **Work in progress** | `phase_d_lda.py` | Supervised correct vs wrong; CV accuracy + shuffle null—**API, thresholds, and skip rules may change** |
| Fourier screening | **Work in progress** | `fourier_screening.py` | Layer-axis spectrum vs layer-order null—**experimental; validate on your config** |
| Correct vs wrong geometry | Active (A/C); WIP (D) | — | Divergence and subspaces in Phase A/C; LDA when Phase D is stable |

## Structure

- **Stage 1 – Pipeline** (`pipeline.py`): Load code tasks → extract activations → generate code → evaluate (tests) → save labels, activations, answers.
- **Stage 2 – Analysis** (`analysis.py`): Load answers + labels → error category breakdown → plots + JSON summary.
- **Phase B** (`phase_b_deconfounding.py`): Label-level confounds — tables, JSON, Markdown; `deconfounding_plan.json` for Phase A/C; correlation matrices under `phase_b/correlation_matrices/`.
- **Phase A**: UMAP/t-SNE by `level_run_id` / layer, colored by correct/error_category; interestingness; correct vs wrong divergence.
- **Phase C**: Concept subspaces (conditional covariance + SVD) for correct/wrong and error_category; residualized activations for downstream.
- **Phase D / Fourier**: See [docs/pipeline_phases.md](docs/pipeline_phases.md). Treat as **WIP** until release notes say otherwise.

## Dataset

Dataset loading is **generic**: only `dataset.source` and params for that source are used.

- **source: json** — Local JSON: `dataset.json_path`; further keys in [docs/datasets.md](docs/datasets.md).
- **source: huggingface** — HuggingFace: `dataset.hf_repo`, `dataset.hf_split`, and column mapping. Use for HumanEval (164 tasks) or MBPP (974 tasks); example configs in [docs/datasets.md](docs/datasets.md).

See **[docs/datasets.md](docs/datasets.md)** for full parameter lists and example YAML.

**Curriculum levels (easy → hard):** **[docs/test_levels.md](docs/test_levels.md)** describes the L0–L5 plan: what each tier is for, how to configure it, what to verify in logs and outputs, and when to advance. Committed JSON lives under `data/levels/` (`level1.json` … `level5.json`).

## Model

Config: `model.name` (HuggingFace id or local path), `model.layers` to extract, `model.hidden_dim`. Example: `meta-llama/CodeLlama-7b-hf` or `codellama/CodeLlama-7b-hf` (base; not Instruct). Gated models need a Hugging Face token.

## Config

**If you are not the original author:** Edit `config.yaml` before running. Set **`model.name`** for your machine: either an absolute path to a local model folder or a HuggingFace repo id. Set `paths.workspace` and `paths.data_root` when you want outputs outside the repo defaults. See [RUN.md](RUN.md) for full steps and Colab/Drive notes.

## Usage

```bash
# Install
pip install -r requirements.txt

# Data generation (GPU)
python pipeline.py --config config.yaml

# Error analysis (CPU)
python analysis.py --config config.yaml

# Phase B: label confounds (CPU; recommended before Phase A/C)
python phase_b_deconfounding.py --config config.yaml
# Or: bash run_phase_b.sh

# Phase A: UMAP/t-SNE embeddings (CPU; needs umap-learn)
python phase_a_embeddings.py --config config.yaml
python phase_a_analysis.py --config config.yaml
# Or: bash run_phase_a.sh

# Phase C: concept subspaces (CPU/GPU depending on config)
python phase_c_subspaces.py --config config.yaml
# Or: bash run_phase_c.sh

# Work in progress — expect rough edges, skips when classes are too small
python phase_d_lda.py --config config.yaml
# Or: bash run_phase_d.sh

python fourier_screening.py --config config.yaml
# Or: bash run_fourier.sh
```

**Google Colab:** see [COLAB.md](COLAB.md) and `colabsetup.ipynb` (Drive mount, `config_colab.yaml`, phase runners).

## Project structure

```
code-geometry/
├── pipeline.py
├── analysis.py
├── phase_b_deconfounding.py
├── phase_a_embeddings.py
├── phase_a_analysis.py
├── phase_c_subspaces.py
├── phase_d_lda.py              # WIP
├── fourier_screening.py        # WIP
├── geometry_common.py
├── path_utils.py
├── shape_geometry_hints.py
├── config.yaml
├── requirements.txt
├── colabsetup.ipynb
├── run_phase_a.sh
├── run_phase_b.sh
├── run_phase_c.sh
├── run_phase_d.sh
├── run_fourier.sh
├── README.md
├── COLAB.md
├── RUN.md
├── REFERENCES.md
├── docs/
│   ├── pipeline_phases.md
│   ├── datasets.md
│   └── test_levels.md
├── data/
│   └── levels/                 # level1.json … level5.json (+ README)
└── scripts/
    ├── generate_level_benchmarks.py
    ├── pad_level_json_to_count.py
    └── export_benchmarks_to_json.py
```

**Outputs:** Under `paths.workspace` and `paths.data_root` in `config.yaml`, in **dataset-named subfolders** (labels, logs, plots, activations, answers, phase_a, phase_b data as JSON/CSV, phase_c, phase_d, fourier when run). Default paths in sample configs may target Colab Drive; edit for local use. See [RUN.md](RUN.md).

## Roadmap

- **Phase D (LDA):** Stabilize CV + shuffle null reporting and class-balance skip rules.
- **Fourier screening:** Harden layer-axis statistics and null calibration across layer lists.
- **Divergence / deviation metrics:** Optional extensions (e.g. k-NN style deviation between correct and error clouds) remain **design space**—not claimed to match any external paper’s exact definition unless documented in `docs/`.

## Documentation

- **[docs/pipeline_phases.md](docs/pipeline_phases.md)** — Phase-by-phase guide, run order, outputs, figure inventory.
- **[docs/datasets.md](docs/datasets.md)** — Dataset config (`json`, level files, HuggingFace).
- **[docs/test_levels.md](docs/test_levels.md)** — Level / difficulty tiers for code tasks.
- **[RUN.md](RUN.md)** — Install, config, commands.
- **[COLAB.md](COLAB.md)** — Google Colab workflow.

## References

See [REFERENCES.md](REFERENCES.md) for related manifold and interpretability papers (background only).
