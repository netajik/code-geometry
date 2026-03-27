# Code Geometry

We study **how a code LLM’s hidden states organize** around coding tasks when difficulty increases. The motivating lens is the same broad question as in interpretability more generally: are outcomes such as correctness, error type, or task structure reflected **linearly** or **subspace-wise** in activations—and what simple **confounds** (prompt length, number of test lines) might explain those patterns?

This repository runs a single pipeline: load tasks (JSON levels or HuggingFace benchmarks), generate answers with a frozen model, save **per-layer activations**, then run **Phase A–C** analyses (embeddings, deconfounding, concept subspaces). **Phase D (LDA)** and **Fourier screening** are included as scripts but are **work in progress** (interfaces, nulls, and reporting are still evolving—not production baselines yet).

**Terminology:** *Level* = curriculum difficulty tier; *population* here means outcome groups (correct / wrong / error_category), not train/test splits. **`level_run_id`** names one saved batch (e.g. `level_01`); artifacts use **`level_run_<id>.json`** and **`level_run_<id>_layer{L}.npy`**. Set **`dataset.level_run_id`** to override the default batch id; **`dataset.hf_split`** selects the HuggingFace subset. Phase D uses stratified k-fold CV for LDA where applicable; avoid duplicating k-fold logic inside Phase C.

---

## Research question

As problems get harder, does the model still separate **correct vs incorrect** representations in ways that survive obvious confounds? Where in the network do embedding geometry, subspace tests, and (eventually) discriminative or spectral summaries agree or disagree?

---

## Dataset

Problems can come from **local JSON** (`data/levels/level1.json` … `level5.json`) or from **HuggingFace** (`dataset.source`, `hf_repo`, `hf_split`, column mapping). See **[docs/datasets.md](docs/datasets.md)** for parameters and example YAML. **[docs/test_levels.md](docs/test_levels.md)** describes the L0–L5 curriculum plan.

| Level | Role |
|-------|------|
| 1 | Easiest; often near-saturated accuracy |
| 2–4 | Intermediate difficulty; useful for correct/wrong contrasts |
| 5 | Hardest; often error-rich when the model fails enough for subspace/LDA-style splits |

Exact counts and accuracies depend on the data and model; re-run `pipeline.py` + `analysis.py` for your run.

---

## Model

- **CodeLlama** (e.g. `meta-llama/CodeLlama-7b-hf`)—set `model.name` in `config.yaml` or `config_colab.yaml`
- Hidden size and layer list come from `config.yaml` (`model.layers`, etc.)
- Activations are stored as **NumPy** arrays per layer under the configured `data_root` (see `path_utils` / pipeline logs)
- Gated checkpoints need a Hugging Face token

---

## Labels and metadata

The pipeline attaches **execution outcomes** (correctness, error categories where applicable) and **pre-generation covariates** used in Phase B (e.g. prompt length, number of test lines). Phase B treats these as **observational** checks: strong associations with correctness warn you that geometric effects might be partly **confounded**, not causal.

---

## Pipeline status

| Stage | Status | Script | Typical outputs |
|-------|--------|--------|-----------------|
| Data generation + activations | Active | `pipeline.py` | Labels JSON, `answers/`, layer `.npy` activations, logs, plots |
| Aggregate analysis | Active | `analysis.py` | `analysis_summary.json`, summary plots |
| Phase B: deconfounding | Active | `phase_b_deconfounding.py` | `phase_b/summary.json`, `deconfounding_plan.json`, correlation tables |
| Phase A: embeddings + summaries | Active | `phase_a_embeddings.py`, `phase_a_analysis.py` | UMAP/t-SNE, interestingness CSV, CKA / norm profiles, Phase A plots |
| Phase C: concept subspaces | Active | `phase_c_subspaces.py` | `phase_c_results.csv`, permutation-style nulls, plots |
| Phase D: LDA | **Work in progress** | `phase_d_lda.py` | CV / shuffle-null summaries (API and thresholds may change) |
| Fourier screening | **Work in progress** | `fourier_screening.py` | Layer-axis spectrum summaries and plots (needs multiple layers in config) |

Treat **Phase D** and **Fourier** as experimental: validate on your config before relying on specific numbers.

---

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

Heavy artifacts (weights, large `.npy`, Drive mirrors) stay **outside** git; paths are set in YAML. Outputs live under `paths.workspace` and `paths.data_root` in **dataset-named subfolders**. See [RUN.md](RUN.md).

---

## Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is missing, install: `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`, `umap-learn`, `transformers`, `accelerate`, `datasets`, `huggingface_hub`, `tqdm`.

Download the model (token if gated), then set `config.yaml` `model.name`, `paths.workspace`, and `paths.data_root`.

---

## Usage

```bash
python pipeline.py --config config.yaml
python analysis.py --config config.yaml
python phase_b_deconfounding.py --config config.yaml
python phase_a_embeddings.py --config config.yaml
python phase_a_analysis.py --config config.yaml
python phase_c_subspaces.py --config config.yaml

# WIP — expect rough edges, skips when classes are too small
python phase_d_lda.py --config config.yaml
python fourier_screening.py --config config.yaml
```

Shell wrappers: `run_phase_a.sh`, `run_phase_b.sh`, `run_phase_c.sh`, `run_phase_d.sh`, `run_fourier.sh`.

**Colab:** [COLAB.md](COLAB.md) and `colabsetup.ipynb` (Drive, `config_colab.yaml`, phase cells).

---

## Documentation

- **[docs/pipeline_phases.md](docs/pipeline_phases.md)** — Phase-by-phase guide, run order, outputs
- **[docs/datasets.md](docs/datasets.md)** — Dataset config (JSON, levels, HuggingFace)
- **[docs/test_levels.md](docs/test_levels.md)** — Level / difficulty tiers
- **[RUN.md](RUN.md)** — Install, config, commands
- **[COLAB.md](COLAB.md)** — Google Colab workflow
