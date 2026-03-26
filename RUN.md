# Steps to Run Code-Geometry

Follow these steps to run the pipeline and analysis on your machine.

**Phase-by-phase explanation** (what each stage does and what it reads/writes): [docs/pipeline_phases.md](docs/pipeline_phases.md).

---

## 1. Prerequisites

- **Python**: 3.8+ (3.10 recommended)
- **GPU**: Recommended for `pipeline.py` (CodeLlama 7B in bfloat16 needs ~14–16 GB VRAM). Phase A and Phase C can run on CPU.
- **Disk**: Enough space for activations (e.g. levels mode: **500** tasks across 5 files × `model.layers` × hidden_dim × 4 bytes — often **tens of MB to ~1 GB** depending on layer count and dtype).

---

## 2. Clone and enter the repo

```bash
cd /path/to/code-geometry
# or: git clone <repo-url> && cd code-geometry
```

---

## 3. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\activate
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `torch`, `transformers`, `accelerate`, `datasets`, `numpy`, `matplotlib`, `pyyaml`, `pandas`, `scipy`, `scikit-learn`, `umap-learn`.

---

## 5. Set config for your machine (required)

Edit **`config.yaml`** before running. The checked-in config may contain machine-specific paths.

**Model (`model.name`)** — choose one:

- **Local model (recommended if you have slow/unreliable internet):**  
  Set to the **absolute path** to the folder that contains the model files (e.g. `config.json`, tokenizer files, and weight files like `.safetensors` or `pytorch_model.bin`).  
  Example: `name: /Users/you/models/CodeLlama-7b-hf`  
  Download the model once (browser or `huggingface-cli download codellama/CodeLlama-7b-hf --local-dir /path/to/CodeLlama-7b-hf`), then point `model.name` at that folder.

- **Download from HuggingFace:**  
  Set to the repo id: `name: codellama/CodeLlama-7b-hf`  
  The pipeline will download and cache the model on first run. For gated models you must log in (step 6).

**Output paths (`paths.workspace`, `paths.data_root`):**  
  Defaults are `output` and `output/data` (relative to the config file, so outputs go under the repo’s `output/`). You can leave these as-is or set absolute paths if you want results elsewhere.

---

## 6. Hugging Face login (gated models)

For gated models (e.g. CodeLlama), log in so the pipeline can download weights:

```bash
python -c "from huggingface_hub import login; login()"
# Enter your HF token when prompted (get one at https://huggingface.co/settings/tokens)
```

Alternatively, set the token in the environment before running:

```bash
export HF_TOKEN=your_token_here
```

---

## 7. Problem count (30+ per level for Phase A t-SNE)

- **`dataset.max_problems: null`** (repo default) uses **every** task in each `data/levels/level*.json`.
- For **≥30 samples per level** (default `phase_a.min_tsne_samples`), pad files once:

```bash
python scripts/pad_level_json_to_count.py --target 35
```

Set **`max_problems: N`** to cap load (e.g. quick tests).

---

## 8. Run the pipeline (GPU)

This step loads the model, downloads the dataset, extracts activations, and generates code. **Results are stored in the repo** under `paths.workspace` and `paths.data_root`. Defaults are **relative to the config file** (repo root): `output` and `output/data`, so outputs go to `output/` and `output/data/` in the repo. All outputs are grouped under a **subfolder** of workspace / data_root (e.g. HumanEval/MBPP from HuggingFace, **`custom`** for one JSON file, or **`levels`** when you use **Level files** (one JSON per level) — per-level outputs `level_01`, …). The `output/` directory is in `.gitignore` so results are not committed.

```bash
python pipeline.py --config config.yaml
```

- **First run**: Downloads the model from HuggingFace and the dataset (HumanEval or MBPP). This can take a while.
- **Outputs** (e.g. repo `output/` and `output/data/`, under dataset subdirs):
  - `output/{dataset}/labels/`, `output/{dataset}/logs/`
  - `output/data/{dataset}/activations/` (`.npy` as `level_run_<id>_layer{L}.npy`), `output/data/{dataset}/answers/`
  - `output/{dataset}/plots/` — pipeline + `analysis.py` diagnostics (accuracy, norms, error distributions, accuracy vs test lines / prompt length); see [docs/pipeline_phases.md](docs/pipeline_phases.md)

To use a custom config path:

```bash
python pipeline.py --config /path/to/my_config.yaml
```

---

## 9. Run error analysis (CPU)

```bash
python analysis.py --config config.yaml
```

Reads answers and labels, assigns error categories, writes plots under `{workspace}/{dataset}/plots/`, and writes `analysis_summary.json` under **both** `{workspace}/{dataset}/labels/` and `{data_root}/{dataset}/answers/` (same content).

---

## 10. Run Phase B – label confounds (CPU, recommended before Phase A/C)

```bash
python phase_b_deconfounding.py --config config.yaml
# Or: bash run_phase_b.sh
```

Reads `{workspace}/{dataset}/labels/` and `{data_root}/{dataset}/answers/`; writes **`{data_root}/{dataset}/phase_b/`** (CSVs + JSON + Markdown). Includes correct-vs-wrong factor notes (`*_correct_vs_wrong_factors.md`) and **pairwise label correlation**: `correlation_matrices/*_R_raw.csv`, `*_R_residualized_wrt_*.csv`, `*_classified_pairs.json`, `*_spearman_top_k.json`, `*_label_correlation_summary.md` (Pearson + residualization w.r.t. `prompt_len` by default; tune `phase_b` in `config.yaml`). No Phase B PNGs. No GPU.

Use this to document **definitional** links (e.g. `correct` vs `error_category`) and **empirical** confounds (prompt length, test-line count vs correctness) before interpreting embedding or subspace separation.

---

## 11. Run Phase A – embeddings (CPU)

```bash
python phase_a_embeddings.py --config config.yaml
python phase_a_analysis.py --config config.yaml
# Or: bash run_phase_a.sh
```

Produces UMAP/t-SNE embeddings and divergence scores under `{data_root}/{dataset}/phase_a/`.

---

## 12. Run Phase C – concept subspaces (CPU)

```bash
python phase_c_subspaces.py --config config.yaml
# Or: bash run_phase_c.sh
```

Produces concept subspaces (correct/wrong) and related outputs under `{data_root}/{dataset}/phase_c/` (including stratified-null and bootstrap columns in `phase_c_results.csv` when enabled in `config.yaml`).

---

## 13. Phase D — LDA discriminants (CPU)

```bash
python phase_d_lda.py --config config.yaml
# Or: bash run_phase_d.sh
```

Writes **`{data_root}/{dataset}/phase_d/`** — per `(level_run_id, layer)` LDA coefficients, 1D decision scores, `metrics.json` (CV accuracy, shuffle-null p-value), and `phase_d_summary.csv`.

---

## 14. Fourier screening (CPU)

```bash
python fourier_screening.py --config config.yaml
# Or: bash run_fourier.sh
```

Writes **`{data_root}/{dataset}/fourier/`** (`*_fourier.json`, `fourier_summary.csv`) and histograms under `{workspace}/{dataset}/plots/fourier/`.

---

## Summary order

| Step | Command | Needs GPU |
|------|----------|-----------|
| Install | `pip install -r requirements.txt` | No |
| 1. Data + activations | `python pipeline.py --config config.yaml` | Yes (recommended) |
| 2. Error analysis | `python analysis.py --config config.yaml` | No |
| 3. Phase B (recommended) | `python phase_b_deconfounding.py --config config.yaml` | No |
| 4. Phase A | `python phase_a_embeddings.py ...` then `phase_a_analysis.py ...` | No |
| 5. Phase C | `python phase_c_subspaces.py --config config.yaml` | No |
| 6. Phase D | `python phase_d_lda.py --config config.yaml` | No |
| 7. Fourier | `python fourier_screening.py --config config.yaml` | No |

---

## Where results are stored

- **Roots**: Everything goes under `paths.workspace` and `paths.data_root` in `config.yaml`. **`~` is expanded** on macOS/Linux. Default in repo config is **Colab Drive**: `/content/drive/MyDrive/code-geometry-output` (mount Drive in Colab first). For local Mac, use e.g. `~/Google Drive/...` or absolute paths.
- **By dataset**: Under those roots, outputs use a **dataset-named subfolder**:
  - HumanEval-style repo → e.g. `.../openai_humaneval/` (or your `output_name`)
  - MBPP → `.../mbpp/` (or repo tail)
  - `source: json` → `.../custom/` (unless `output_name` is set)
  - Level files (`source: json_levels`) → `.../levels/` with per-level files `level_01`, … ([data/levels/README.md](data/levels/README.md), [docs/datasets.md](docs/datasets.md))

Example layout (workspace = `/content/drive/MyDrive/code-geometry-output`, dataset / **level_run_id** `custom` for `source: json`):

```
/content/drive/MyDrive/code-geometry-output/
├── custom/
│   ├── labels/
│   ├── logs/
│   └── plots/
└── data/
    └── custom/
        ├── activations/
        ├── answers/
        ├── phase_a/
        ├── phase_b/
        ├── phase_c/
        ├── phase_d/
        └── fourier/
```

## Changing dataset or paths

- **Dataset**: In `config.yaml`, set `dataset.source` and the params for that source (see [docs/datasets.md](docs/datasets.md)).
- **Paths**: Set `paths.workspace` and `paths.data_root` to any folder. **Colab:** `/content/drive/MyDrive/...` after mounting Drive. **Mac:** `~/Google Drive/...` or `~/Library/CloudStorage/...`. **Repo-local:** `output` and `output/data`.
