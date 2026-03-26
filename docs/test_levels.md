# Test levels (committed plan)

This document is the **contract** for how you ramp difficulty with **CodeLlama-7b-hf base** (or any model in `config.yaml`). Each level has a **goal**, **what you configure**, **what you should see**, and **when to move on** — a **difficulty / scale gradient** for code-generation manifold experiments.

---

## Summary table

| Level | Name | Goal | Typical data | Rough runtime (local MPS) |
|-------|------|------|--------------|---------------------------|
| **L0** | Smoke | Prove stack: model loads, tokenizer works, one forward + one generate, files written | `json`, **1 task**, `max_problems: 1` | seconds–1 min |
| **L1** | Micro suite | End-to-end pipeline + analysis on tiny curated JSON | `data/quick_test_problems.json`, **10–20 tasks** | few minutes |
| **L2** | Benchmark slice | Real tasks, still small N; compare to “known hard” tasks | HumanEval or MBPP via `huggingface`, **10–30** `max_problems` | ~10–30 min |
| **L3** | Full standard benchmark | Paper-comparable scale on one benchmark | HumanEval **164** or MBPP **test** (full or sanitized) | long |
| **L4** | Multi-benchmark | Same model, different task distribution (geometry transfer) | Run **separate** configs: HumanEval then MBPP; outputs in different `output/` subfolders | long |
| **L5** | Curated hard | Hand-picked or screened task IDs in a JSON | `json` pointing at a **subset file** you maintain | as needed |

---

## L0 — Smoke

**Commitment:** You do not debug geometry until L0 passes.

**Configure**

- `dataset.source: json`
- `dataset.json_path`: any JSON list with **one** object `{ "task_id", "prompt", "test_cases" }` (or reuse `data/quick_test_problems.json` with `max_problems: 1`)
- `dataset.max_problems: 1`
- `generation.batch_size: 1`
- `model.layers`: one layer, e.g. `[16]`, to save time and disk

**You should see**

- Log: model device (`mps` / `cuda` / `cpu`) as intended
- `output/<name>/labels/`, `output/data/<name>/activations/`, `answers/` populated for **level_run_id** `custom` (or your `dataset.level_run_id`)
- No crash through `save_answers` (or end of `main`)

**If it fails:** fix model path, tokenizer, memory (`batch_size`, fewer `layers`, lower `max_new_tokens`) before L1.

---

## L1 — Micro suite

**Commitment:** L1 is the **default** for iterating on prompts, evaluation, and plots.

**Configure**

- `dataset.source: json`
- `dataset.json_path: data/quick_test_problems.json` (or your own small JSON)
- `dataset.max_problems: null` or cap at 15–20
- Restore a sensible `model.layers` (repo default five layers `[4, 8, 16, 24, 31]` for 32-layer stacks, or fewer for smoke tests)
- `generation.batch_size`: 4–8 if memory allows

**You should see**

- Logs: activation extraction + generation + “generation done” timing
- `analysis.py` runs; `analysis_summary.json` + plots under `output/<name>/plots/`
- Mix of correct/wrong is OK for base model; **category counts** should make sense (e.g. not 100% one bucket if you have diverse tasks)

**When to move on:** L1 runs reliably twice in a row (same config, reproducible greedy outputs).

---

## L2 — Benchmark slice

**Commitment:** First contact with **real** HumanEval/MBPP semantics (columns, tests, length).

**Configure**

- `dataset.source: huggingface`
- `dataset.hf_repo`, `dataset.hf_split`, column mapping per [datasets.md](datasets.md)
- `dataset.max_problems: 20` (or 30)

**You should see**

- Output folder name derived from repo (e.g. `openai_humaneval` → check `output/` layout in your run)
- Longer wall time; generation is the bottleneck
- Answers JSON with real `task_id` strings

**When to move on:** You trust that loading, padding, and generation behave on **multi-line** prompts (not only tiny JSON tasks).

---

## L3 — Full standard benchmark

**Commitment:** One full benchmark run for **paper-scale** numbers and geometry (Phase A/C need enough N).

**Configure**

- Same as L2 but `dataset.max_problems: null` (all HumanEval test, or full MBPP test subset)
- Consider **fewer layers** if disk/time constrained; document which layers you kept

**You should see**

- Large `.npy` activations; Phase A/C meaningful if you have both correct and wrong examples

**When to move on:** When you have a frozen config (model, layers, max_new_tokens) you want to cite in writeups.

---

## L4 — Multi-benchmark

**Commitment:** Same pipeline, **two** distributions (e.g. HumanEval vs MBPP). Do **not** merge runs in one folder without renaming; rely on separate `output_name` or different `hf_repo` so `output/<dataset>/` stays separate.

**Configure**

- Run 1: HumanEval (full or large `max_problems`)
- Run 2: MBPP (set `mbpp_sanitized: true` for the 427-task sanitized split)

**You should see**

- Two trees under `output/` (different dataset folder names)
- Comparable logs for timing and memory

---

## L5 — Curated hard

**Commitment:** You **choose** tasks (e.g. failed IDs from L3, or hand-listed hard tasks) in a **JSON** and point `dataset.source: json` at that file.

**Configure**

- Build `data/my_hard_subset.json` (list of `{ task_id, prompt, test_cases }`)
- `dataset.json_path: data/my_hard_subset.json`
- You may maintain a **screening** script/notebook outside the main pipeline; this repo consumes the JSON only.

**You should see**

- Focused analysis on hard region; often lower accuracy — useful for error geometry if N is still sufficient per category

---

## What to look at after each run (checklist)

1. **Log:** device line (GPU/MPS/CPU), generation batch message, “generation done” time.
2. **`output/.../answers/level_run_*.json`:** `n_problems`, `n_correct`, sample `raw_text` / `generated_code`.
3. **`analysis_summary.json`:** error_category histogram.
4. **Phase A:** Run `phase_a_analysis.py` after `pipeline.py` for CKA + norm plots. Run `phase_a_embeddings.py` before or after (order: embeddings first if you want UMAP/t-SNE, `interestingness_scores.csv`, and divergence line plots under `plots/phase_a/`).

---

## Single source of truth

- **Levels are defined here** (`docs/test_levels.md`).  
- **Mechanics** (JSON vs HuggingFace keys) stay in [datasets.md](datasets.md).  
- **Run commands** stay in [RUN.md](../RUN.md).

When you change what “L2” means in your project, **update this file** so future you (and collaborators) know exactly what was run.
