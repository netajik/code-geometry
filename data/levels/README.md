# Dataset (levels)

Problems are **fixed on disk**, grouped by **Level** (difficulty). Each level is one JSON file (`level1.json` … `level5.json`). The pipeline runs **all levels in one pass** and writes **per-level** labels, activations, and answers under your output paths (see [RUN.md](../RUN.md)).

## Level table

| Level | Type (difficulty) | What the problems look like |
|-------|-------------------|-------------------------------|
| 1 | Easiest | Tiny arithmetic / bool / string ops (100 tasks). |
| 2 | Easy | One-pass lists/strings; empty and edge cases (100). |
| 3 | Medium | Second-largest, XOR single, merge intervals, bitcount, etc. (100). |
| 4 | Hard | Kadane, LIS, substring window, islands, Fib variants (100). |
| 5 | Hardest | `typing` stubs; heap/spiral/partition-style specs (100). |

Each file has **100** problems; **500** total with **no duplicate prompts** across levels. Regenerate from `scripts/generate_level_benchmarks.py` (top of file: `TASKS_PER_LEVEL`; validates each task with a reference solution).

Higher levels should be **harder for the model** (expect accuracy to drop). If almost nothing passes at one level, move some problems to an easier file.

## File layout

- `level1.json` … `level5.json` — each file is a **JSON array of problems**.
- Each problem has at least: `task_id`, `prompt`, `test_cases` (same shape as a single-file JSON run in this repo).

## Config (`config.yaml`)

Set `source: json_levels` and a `levels` map from level index to path (see repo `config.yaml`).

```yaml
dataset:
  source: json_levels
  levels:
    1: data/levels/level1.json
    2: data/levels/level2.json
    # …
  run_levels: null     # null = all levels; or e.g. [2, 4] to run only those
  max_problems: null   # null = all tasks per level; or cap e.g. 20 for quick tests
```

Extra keys (`prompt_key`, `test_cases_key`, `list_key`, …) match single-file `json`; see [docs/datasets.md](../docs/datasets.md).

## Outputs

Under `paths.workspace` and `paths.data_root`, results use a subfolder (default **`levels`** unless `dataset.output_name` is set). Per-level artifacts use names like **`level_01`, `level_02`, …** so numeric order sorts correctly.

## Editing levels

Add or change JSON files and update the `levels` map. If the JSON root is an object, set `list_key` to the key that holds the problem list.

**Phase A (t-SNE):** 100 tasks per level exceeds the “30+ samples” guideline. To grow smaller files, pad:

```bash
python scripts/pad_level_json_to_count.py --target 100
```

**Regenerate curated benchmark (default 100×5):** `python scripts/generate_level_benchmarks.py` — edit `TASKS_PER_LEVEL` in that script to change size.
