# References: Code Geometry & Manifolds

This project combines (1) a **staged pipeline** (activations + labels + answers) for code generation, (2) **REMA-style** reasoning-manifold and deviation analysis, and (3) **manifold papers** on LLM representation geometry and code/debugging.

---

## 1. REMA – Reasoning manifold & failure geometry

- **REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Models** (arXiv 2509.22518).
  - **Reasoning manifold**: Correct-reasoning representations lie near a low-dimensional manifold; errors **deviate** from it.
  - **Deviation**: k-NN distance of each error representation to the correct-set point cloud; Welch t-test vs within-correct distances.
  - **Divergence point**: Per error sample, first layer where deviation > μ_correct + α·σ_correct.
  - **Separability**: Binary classifier (e.g. SVC RBF) on correct vs error representations per layer.
  - Tasks in paper: MATH, GSM8K, GPQA, VQAv2, SNLI-VE, AI2D, MathVista (no code); methodology applies to code (correct = tests pass, error = tests fail / syntax / timeout).

---

## 2. Manifold papers (from manifold papers notes)

- **The Origins of Representation Manifolds in Large Language Models** (2025): LLM internal states as continuous manifolds; cosine similarity ≈ geometry along manifold; relevance to interpretability and code/debugging.
- **Learning Stratified Manifold Structures in LLM Embedding Space** (2025): Multiple sub-manifolds (e.g. code vs natural language); task specialization.
- **Characterizing LLM Geometry Helps Solve Tasks** (2023): Intrinsic dimension, piecewise-linear structure; interpretability and control.
- **Emergence of Separable Manifolds in Deep Language Representations**: Transformers untangle manifolds across layers; early layers messy, later more separable.
- **Manifold-based debugging**: Valid code on structured manifolds; bugs/wrong code as off-manifold; distance-from-manifold and embedding anomalies for detection and clustering of failure modes.

---

## 3. Project flow (thesis / excalidraw)

- **Step 1 – Subspace identification of known concepts**: Map “concepts” to code labels (syntax, construct, correct/error); use activations + rSVD/LDA to identify subspaces.
- **Model + Dataset + Labels**: Code model, code benchmark (e.g. HumanEval), code labels (tests, syntax, structure).
- **Activation storage**: Per **level_run_id**, per-layer `.npy` files for reproducible geometry.

---

## 4. Downstream phases (code-geometry) — status

- **Phase A (done)**: `phase_a_embeddings.py`, `phase_a_analysis.py` — UMAP/t-SNE colored by correct/error_category; divergence score (correct vs wrong); interestingness CSV.
- **Phase C (done)**: `phase_c_subspaces.py` — Concept subspaces (correct/wrong) via conditional covariance + SVD; permutation null; basis and eigenvalues saved.
- **Phase D (planned)**: `phase_d_lda.py` stub — LDA for error directions.
- **Fourier screening (planned)**: `fourier_screening.py` stub — Periodicity in centroids.
- **Correct vs wrong geometric comparison**: Embedded in Phase A (divergence score) and Phase C (subspace comparison). Full REMA-style k-NN deviation and divergence layer can be added later.

---

## 5. Code benchmarks

- **HumanEval**: OpenAI; 164 Python function tasks; prompt + test string. Paper: Chen et al., *Evaluating Large Language Models Trained on Code*, arXiv:2107.03374.
- **MBPP**: Mostly Basic Python Problems; ~1k tasks (full) or 427 sanitized; loaded via HuggingFace config in `dataset.source: huggingface`. Paper: Austin et al., *Program Synthesis with Large Language Models*, arXiv:2108.07732.
- **Custom JSON**: User-defined tasks via `dataset.source: json` and `dataset.json_path`.

Details (standard vs custom, citations, other benchmarks like APPS/DS-1000): **docs/datasets.md**.

All references above are used to define the code-geometry pipeline, analysis, and planned manifold steps; no outputs are stored in the repo.
