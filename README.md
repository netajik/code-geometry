# Code Geometry

Manifold-style interpretability for **code generation**: extract residual-stream activations from a code-capable LLM on code tasks (e.g. HumanEval), evaluate correctness, and analyze error patterns for downstream geometry (REMA-style deviation, UMAP, rSVD, LDA).

Aligned with manifold-style interpretability and REMA-style ideas (deviation from correct representations in activation space); see [REFERENCES.md](REFERENCES.md).

**Terminology:** *Level* = curriculum difficulty tier; *population* here means outcome groups (correct / wrong / error_category), not train/test splits. **`level_run_id`** names one saved batch (e.g. `level_01`); artifacts use **`level_run_<id>.json`** and **`level_run_<id>_layer{L}.npy`**. Set **`dataset.level_run_id`** to override the default batch id; **`dataset.hf_split`** selects the HuggingFace subset. **Phase D** already uses stratified k-fold CV for LDA; do not duplicate k-fold inside Phase C.

## Research question

When a code model gets a task wrong, where does the internal representation deviate? We study activation geometry for correct vs incorrect code generations and error categories (syntax, logic, timeout, garbage).

## Pipeline Status

| Stage | Status | Script | Output |
|-------|--------|--------|--------|
| Data Generation | Complete | `pipeline.py`, `analysis.py` | Activations (`level_run_<id>_layer{L}.npy`), labels, answers, error analysis |
| Phase B: Label confounds | Complete | `phase_b_deconfounding.py` | Cramér’s V, point-biserial, splits/logistic, pairwise Pearson + residualized R, classified pairs, Spearman top-*K*; `deconfounding_plan.json`; no Phase B PNGs |
| Phase A: Visual Reconnaissance | Complete | `phase_a_embeddings.py`, `phase_a_analysis.py` | UMAP/t-SNE embeddings + interestingness; `phase_a_analysis.py` adds CKA and norm profiles (runs after pipeline even before embeddings CSVs exist) |
| Phase C: Concept Subspaces | Complete | `phase_c_subspaces.py` | Concept subspaces (correct/wrong, error_category), significance, projections |
| Phase D: LDA | Complete | `phase_d_lda.py` | Supervised correct vs wrong; CV accuracy + shuffle null |
| Fourier Screening | Complete | `fourier_screening.py` | Layer-axis spectrum vs layer-order null |
| Correct vs. Wrong Geometric Comparison | Phase A/C/D | — | Divergence, subspaces, LDA (see phase outputs) |

## Structure

- **Stage 1 – Pipeline** (`pipeline.py`): Load code tasks → extract activations → generate code → evaluate (tests) → save labels, activations, answers.
- **Stage 2 – Analysis** (`analysis.py`): Load answers + labels → error category breakdown → plots + JSON summary.
- **Phase B** (`phase_b_deconfounding.py`): Label-level confounds — tables, JSON, Markdown; `deconfounding_plan.json` for Phase A/C; correlation matrices under `phase_b/correlation_matrices/`.
- **Phase A**: UMAP/t-SNE by level_run_id/layer, colored by correct/error_category; interestingness; correct vs wrong divergence.
- **Phase C**: Concept subspaces (conditional covariance + SVD) for correct/wrong and error_category; residualized activations for downstream.
- **Phase D / Fourier**: `phase_d_lda.py`, `fourier_screening.py` (see [docs/pipeline_phases.md](docs/pipeline_phases.md)).

## Dataset

Dataset loading is **generic**: only `dataset.source` and params for that source are used.

- **source: json** — Local JSON: `dataset.json_path`; further keys in docs/datasets.md.
- **source: huggingface** — HuggingFace: `dataset.hf_repo`, `dataset.hf_split`, and column mapping. Use for HumanEval (164 tasks) or MBPP (974 tasks); example configs in docs/datasets.md.

See **docs/datasets.md** for full parameter lists and example YAML.

**Test levels (easy → hard):** See **[docs/test_levels.md](docs/test_levels.md)** for a committed L0–L5 plan: what each tier is for, how to configure it, what to verify in logs and outputs, and when to advance.

## Model

Config: `model.name` (HuggingFace id or local path), `model.layers` to extract, `model.hidden_dim`. Example: `codellama/CodeLlama-7b-hf` (base; not Instruct).

## Config

**If you are not the original author:** Edit `config.yaml` before running. Set **`model.name`** for your machine: either an absolute path to a local model folder (e.g. after downloading CodeLlama-7b-hf manually) or a HuggingFace repo id (e.g. `codellama/CodeLlama-7b-hf`) to download on first run. Set `paths.workspace` and `paths.data_root` when you want outputs outside the repo defaults. See [RUN.md](RUN.md) for full steps and config options.

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

# Phase C: Concept subspaces (CPU)
python phase_c_subspaces.py --config config.yaml
# Or: bash run_phase_c.sh

# Phase D / Fourier (planned stubs)
python phase_d_lda.py --config config.yaml
python fourier_screening.py --config config.yaml
```

## Project structure

```
code-geometry/
├── pipeline.py              # Data generation (problems, activations, answers)
├── analysis.py              # Error analysis, plots, summary
├── phase_b_deconfounding.py # Label confounds, deconfounding_plan.json
├── phase_a_embeddings.py    # UMAP/t-SNE, coloring, divergence scores
├── phase_a_analysis.py     # Phase A summary
├── phase_c_subspaces.py    # Concept subspaces (correct/wrong), permutation null
├── phase_d_lda.py          # Planned: LDA for error directions
├── fourier_screening.py    # Planned: periodicity in centroids
├── config.yaml
├── run_phase_a.sh
├── run_phase_b.sh
├── run_phase_c.sh
├── requirements.txt
├── README.md
├── COLAB.md
└── REFERENCES.md
```

**Outputs**: Under `paths.workspace` and `paths.data_root` in `config.yaml`, in **dataset-named subfolders** (labels, logs, plots, activations, answers, phase_a, `phase_b` data JSON/CSV only, phase_c). Default config uses **Colab Drive** (`/content/drive/MyDrive/code-geometry-output`); edit for local Mac (`~/Google Drive/...`) or repo (`output` / `output/data`). See [RUN.md](RUN.md).

## What's Next: Beyond Linear Subspaces

- **Phase D — LDA for error directions**: Discriminative linear analysis (correct vs wrong, or by error_category) to capture low-variance directions Phase C may miss.
- **Fourier screening**: Periodicity in activation centroids (e.g. by task_id or error type) — do code concepts sit on low-dimensional periodic structure?
- **Correct vs. wrong geometric comparison**: k-NN deviation from correct manifold, divergence layer (REMA-style); currently embedded in Phase A (divergence score) and Phase C (principal angles).
- **Manifold interaction**: How do correct vs error sub-manifolds compose across layers? Causal validation via steering along discovered directions.

## Documentation

- **[docs/pipeline_phases.md](docs/pipeline_phases.md)** — Manifold-oriented analysis guide, phase-by-phase run order, outputs, figure inventory.
- **[docs/datasets.md](docs/datasets.md)** — Dataset config (`json`, Level files, HuggingFace).
- **[docs/test_levels.md](docs/test_levels.md)** — Level / difficulty tiers for code tasks.
- **[RUN.md](RUN.md)** — Install, config, commands.

## References

See [REFERENCES.md](REFERENCES.md) for manifold papers, REMA, and related work.
