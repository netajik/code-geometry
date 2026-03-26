# Run Code Geometry on Google Colab

Short setup. **Model** and **results** default to **Google Drive** paths that match `config.yaml` (adjust `paths` for your Drive layout).

## Before you start

- **Runtime:** GPU (Runtime → Change runtime type → GPU).
- **Colab secret:** Left sidebar → key icon → Add secret `HF_TOKEN` = your Hugging Face token (for gated models if needed).

## 1. Mount Drive (recommended)

```python
from google.colab import drive
drive.mount("/content/drive")
import os
DRIVE_BASE = "/content/drive/MyDrive"
DRIVE_RESULTS = f"{DRIVE_BASE}/code-geometry-results"
os.makedirs(DRIVE_RESULTS, exist_ok=True)
os.makedirs(os.path.join(DRIVE_RESULTS, "data"), exist_ok=True)
```

Default model cache: `f"{DRIVE_BASE}/models/CodeLlama-7b-hf"` (same as repo `config.yaml`).

## 2. Clone / unzip and install

If you **unzip** a release zip, the repo often ends up at `/content/code-geometry/code-geometry/` (nested). `cd` to the directory that contains `pipeline.py`.

```python
!git clone https://github.com/YOUR_ORG/code-geometry /content/code-geometry
# or: !unzip "/content/code-geometry.zip" -d "/content/code-geometry"
import os
from pathlib import Path
root = Path("/content/code-geometry/code-geometry")
if not (root / "pipeline.py").is_file():
    root = Path("/content/code-geometry")
os.chdir(root)
!pip install -q torch numpy matplotlib pyyaml transformers accelerate datasets umap-learn scipy scikit-learn pandas
```

## 3. Login, model on runtime, build config

Run **after** the clone cell so you're in the repo. Uses absolute path so it works even if cwd changed.

```python
import os
import yaml
from huggingface_hub import snapshot_download, login
from google.colab import userdata

from pathlib import Path
REPO_DIR = Path("/content/code-geometry/code-geometry") if (Path("/content/code-geometry/code-geometry") / "pipeline.py").is_file() else Path("/content/code-geometry")
os.chdir(REPO_DIR)

login(token=userdata.get("HF_TOKEN"))

DRIVE_BASE = "/content/drive/MyDrive"
RUNTIME_MODEL = f"{DRIVE_BASE}/models/CodeLlama-7b-hf"
HF_REPO_ID = "meta-llama/CodeLlama-7b-hf"
os.makedirs(RUNTIME_MODEL, exist_ok=True)
snapshot_download(HF_REPO_ID, local_dir=RUNTIME_MODEL, local_dir_use_symlinks=False)

with open(REPO_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["paths"]["workspace"] = DRIVE_RESULTS if "DRIVE_RESULTS" in globals() else f"{DRIVE_BASE}/code-geometry-results"
cfg["paths"]["data_root"] = os.path.join(cfg["paths"]["workspace"], "data")
cfg["model"]["name"] = RUNTIME_MODEL
# cfg["dataset"]["max_problems"] = 10  # cap tasks for a quick run

with open("config_colab.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
```

## 4. Run pipeline and analysis

```python
!python pipeline.py --config config_colab.yaml
!python analysis.py --config config_colab.yaml
```

## 4b. Phase B (label confounds)

Run after `analysis.py` and before Phase A/C. Label-level checks: correctness vs `error_category` (definitional in this pipeline), prompt length and test-line count vs correctness (JSON/CSV under `data/{dataset}/phase_b/`; no Phase B plots).

```python
!python phase_b_deconfounding.py --config config_colab.yaml
```

## 4c. Geometry phases (same order as [RUN.md](RUN.md))

```python
!python phase_a_embeddings.py --config config_colab.yaml
!python phase_a_analysis.py --config config_colab.yaml
!python phase_c_subspaces.py --config config_colab.yaml
!python phase_d_lda.py --config config_colab.yaml
!python fourier_screening.py --config config_colab.yaml
```

## 5. View plots

With default `dataset.source: json_levels`, outputs use folder name **`levels`**:  
`DRIVE_RESULTS/levels/plots/` (and `DRIVE_RESULTS/data/levels/` for activations / phase data).

```python
from IPython.display import Image, display
plots_dir = os.path.join(DRIVE_RESULTS, "levels", "plots")
for name in sorted(os.listdir(plots_dir)):
    if name.endswith(".png"):
        display(Image(os.path.join(plots_dir, name), width=500))
```
