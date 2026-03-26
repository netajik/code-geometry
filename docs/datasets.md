# Datasets: Generic Config

The pipeline loads code problems **only from parameters** in `config.yaml` under `dataset`. Three ways to supply problems:

1. **`json`** — one local file of problems.
2. **`json_levels`** — **one file per Level** (fixed problems on disk per tier; one pipeline run over all levels). See [data/levels/README.md](../data/levels/README.md).
3. **`huggingface`** — load from a HuggingFace dataset.

Folder names under your output roots follow `source` / `hf_repo` / `output_name` when set (Level files default to a folder named **`levels`**).

---

## Source: `json`

Load from a local JSON file.

| Param | Required | Default | Description |
|-------|----------|---------|-------------|
| `source` | yes | — | `json` |
| `json_path` | yes | — | Path to JSON file (relative to config file or absolute). |
| `max_problems` | no | null | Cap number of tasks. Repo default **`null`** uses full level files (**35+** per level after `scripts/pad_level_json_to_count.py`). |
| `prompt_key` | no | `prompt` | Key in each item for the prompt text. |
| `test_cases_key` | no | `test_cases` | Key for test code string. |
| `task_id_key` | no | `task_id` | Key for task id. |
| `entry_point_key` | no | `entry_point` | Key for entry point function name. |
| `list_key` | no | null | If the JSON root is an object, key that holds the list (e.g. `problems` or `data`). If null, root must be a list. |
| `level_run_id` | no | `custom` | Batch id for this run (saved in JSON + filenames `level_run_<id>_…`). Distinct from outcome groups (correct / wrong). |
| `output_name` | no | `custom` | Folder name under workspace/data_root. |

**Example:**

```yaml
dataset:
  source: json
  json_path: data/quick_test_problems.json
  max_problems: null
```

---

## Source: `json_levels` (Level files)

**One JSON file per Level** — problems live in `level1.json`, `level2.json`, …; by default the pipeline processes **every** configured level in one run and writes **per-level** outputs. Internal grouping names are `level_01`, `level_02`, … (zero-padded from the keys in `levels`) so filenames sort by level.

| Param | Required | Default | Description |
|-------|----------|---------|-------------|
| `source` | yes | — | `json_levels` |
| `levels` | yes | — | Map **level index → path**, e.g. `1: data/levels/level1.json`. Indices are sorted numerically. |
| `run_levels` | no | null | **null** or omit → load **all** keys in `levels`. **List** of integers, e.g. `[2, 4]` → only those levels (each must exist in `levels`). Order in the list is ignored; runs follow numeric level order. |
| `max_problems` | no | null in repo `config.yaml` | Cap problems **per level** (first *N* in each file). **`null`** uses all rows (**35+** per level with padded `data/levels/*.json`). |
| `prompt_key` | no | `prompt` | Same as `json`. |
| `test_cases_key` | no | `test_cases` | Same as `json`. |
| `task_id_key` | no | `task_id` | Same as `json`. |
| `entry_point_key` | no | `entry_point` | Same as `json`. |
| `list_key` | no | null | Same as `json`. |
| `output_name` | no | — | Output subfolder under workspace / data_root; if unset, uses **`levels`**. |

**Example:**

```yaml
dataset:
  source: json_levels
  levels:
    1: data/levels/level1.json
    2: data/levels/level2.json
    3: data/levels/level3.json
  run_levels: null    # or e.g. [1, 3] to run only levels 1 and 3
  max_problems: null   # use all tasks per level (35+ after padding script)
```

Each file uses the **same problem object shape** as **`json`**. Which problems belong in which Level is up to you (difficulty gradient). Details: **[data/levels/README.md](../data/levels/README.md)**.

---

## Source: `huggingface`

Load from a HuggingFace dataset.

| Param | Required | Default | Description |
|-------|----------|---------|-------------|
| `source` | yes | — | `huggingface` |
| `hf_repo` | yes | — | HuggingFace dataset repo id (e.g. `openai_humaneval`, `google-research-datasets/mbpp`). |
| `hf_split` | no | `test` | HuggingFace subset key (e.g. `test`, `train`). |
| `hf_config` | no | null | Dataset config name if the repo has multiple (e.g. `openai_humaneval`). |
| `max_problems` | no | null | Cap number of tasks. |
| `prompt_column` | no | `prompt` | Column name for prompt text. |
| `test_cases_column` | no | — | Column name for test code as a **single string**. Use this **or** `test_list_column`, not both. |
| `test_list_column` | no | — | Column name for test code as a **list of lines** (joined with newline). |
| `test_setup_column` | no | — | If using `test_list_column`, column for setup code to prepend (leave unset if none). |
| `task_id_column` | no | `task_id` | Column for task id. |
| `entry_point_column` | no | `entry_point` | Column for entry point function name. |
| `task_id_prefix` | no | — | If task_id is not a string, prepend this (e.g. `mbpp/`). |
| `level_run_id` | no | value of `hf_split` | Output batch id (defaults to `hf_split` string). `hf_split` selects the HuggingFace subset. |
| `output_name` | no | last part of `hf_repo` | Folder name under workspace/data_root. |

**HumanEval example:**

```yaml
dataset:
  source: huggingface
  hf_repo: openai_humaneval
  hf_config: openai_humaneval
  hf_split: test
  prompt_column: prompt
  test_cases_column: test
  task_id_column: task_id
  entry_point_column: entry_point
  max_problems: null   # or 50 for quick run
```

**MBPP example:**

```yaml
dataset:
  source: huggingface
  hf_repo: google-research-datasets/mbpp
  hf_split: test
  prompt_column: text
  test_list_column: test_list
  test_setup_column: test_setup_code
  task_id_column: task_id
  task_id_prefix: "mbpp/"
  max_problems: null   # or 50 for quick run
```

---

## Standard benchmarks (papers)

These are established benchmarks you can load via the **huggingface** source with the params above.

### HumanEval

- **What**: 164 hand-written Python function-level problems. Each task has a docstring prompt and unit tests; the model completes the function body.
- **Paper**: Chen et al., **Evaluating Large Language Models Trained on Code**, 2021. arXiv: [2107.03374](https://arxiv.org/abs/2107.03374).

### MBPP (Mostly Basic Python Problems)

- **What**: ~1,000 crowd-sourced Python tasks (entry-level), each with a short description, reference solution, and 3 test assertions.
- **Paper**: Austin et al., **Program Synthesis with Large Language Models**, 2021. arXiv: [2108.07732](https://arxiv.org/abs/2108.07732).

---

## Other standard benchmarks (not yet in this repo)

These are well-known code benchmarks from research; they are **not** currently implemented in this pipeline but are commonly cited:

| Benchmark | Paper / source | Notes |
|-----------|----------------|--------|
| **APPS** | Hendrycks et al., *Measuring Coding Challenge Competence With APPS*, NeurIPS 2021. arXiv: [2105.09938](https://arxiv.org/abs/2105.09938) | ~10k problems, difficulty levels; larger and more varied than HumanEval/MBPP. |
| **DS-1000** | Lai et al., *DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation*, ICML 2023 | Data science code (Pandas, NumPy, etc.); different domain. |
| **HumanEval+ / MBPP+** | Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*, 2023; and extended test suites | Stronger test coverage; drop-in replacements for HumanEval/MBPP. |
| **HumanEval Pro / MBPP Pro** | *HumanEval Pro and MBPP Pro: Evaluating LLMs on Self-invoking Code Generation*, arXiv:2412.21199 | Self-invoking code; harder variants. |

Adding support for APPS, DS-1000, or Pro variants would require implementing the corresponding loaders and test harness in `pipeline.py` and extending this doc.

---

## Summary

| source | Description |
|--------|--------------|
| `json` | One local JSON file of problems; default output folder **`custom`** (or `output_name` / **`level_run_id`** when set). |
| `json_levels` | One JSON file per Level; per-level output groups `level_01`, `level_02`, …; default folder **`levels`**. |
| `huggingface` | HuggingFace dataset; repo, `hf_split`, and column mapping. |

HumanEval and MBPP are loaded by setting `source: huggingface` and the params in the examples above. For comparable results, cite the respective papers when reporting metrics.
