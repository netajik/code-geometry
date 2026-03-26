#!/usr/bin/env python3
"""
Code Geometry Pipeline

Loads code-generation tasks (e.g. HumanEval), extracts residual-stream
activations from a code-capable LLM, runs generated code against tests,
and saves labels/activations/answers for manifold and error analysis.

Terminology:
  * **level_run_id** — ID for one saved batch of tasks (e.g. ``level_01``, ``custom``).
    Outcome **population** here means correct vs wrong / ``error_category``, not the batch id.
  * On-disk files use prefix ``level_run_`` (e.g. ``level_run_level_01_layer8.npy``).
  * Set ``dataset.level_run_id`` to override the default batch id; HuggingFace subset
    is selected with ``dataset.hf_split``.

Usage:
    python pipeline.py
    python pipeline.py --config path.yaml
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import yaml


# ====================================================================
# 1. CONFIG
# ====================================================================

def get_dataset_name(cfg):
    """Return folder name for outputs. Uses dataset.output_name if set; else derived from source."""
    d = cfg.get("dataset", {})
    if d.get("output_name"):
        return d["output_name"]
    source = d.get("source", "json")
    if source == "json":
        return "custom"
    if source == "huggingface":
        repo = d.get("hf_repo", "")
        return repo.split("/")[-1] if repo else "hf"
    if source == "json_levels":
        return "levels"  # output subfolder when using per-level JSON files
    return "custom"


def resolve_output_roots(workspace_str, data_root_str, config_base):
    """Expand ~ and resolve workspace / data_root. Absolute paths used as-is (after expanduser)."""
    base = Path(config_base)
    ws = Path(workspace_str).expanduser()
    dr = Path(data_root_str).expanduser()
    if not ws.is_absolute():
        ws = (base / ws).resolve()
    else:
        ws = ws.resolve()
    if not dr.is_absolute():
        dr = (base / dr).resolve()
    else:
        dr = dr.resolve()
    return ws, dr


def load_config(config_path=None):
    """Load config, validate, derive paths. All outputs go under dataset-named subdirs.
    Relative paths are resolved against the config file's directory (repo root).
    Absolute paths and ~ (home) are supported for persistent stores (e.g. Google Drive).
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    config_path = Path(config_path)
    base = config_path.resolve().parent
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_base"] = base

    num_layers = cfg["model"]["num_layers"]
    for layer in cfg["model"]["layers"]:
        assert 0 <= layer < num_layers, f"Layer {layer} out of range [0, {num_layers})"

    ws, dr = resolve_output_roots(
        cfg["paths"]["workspace"], cfg["paths"]["data_root"], base
    )
    dset = get_dataset_name(cfg)
    cfg["paths"]["dataset_name"] = dset
    cfg["paths"]["labels_dir"] = str(ws / dset / "labels")
    cfg["paths"]["logs_dir"] = str(ws / dset / "logs")
    cfg["paths"]["plots_dir"] = str(ws / dset / "plots")
    cfg["paths"]["activations_dir"] = str(dr / dset / "activations")
    cfg["paths"]["answers_dir"] = str(dr / dset / "answers")

    for key in ("labels_dir", "logs_dir", "plots_dir", "activations_dir", "answers_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    return cfg


# ====================================================================
# 2. LOGGING
# ====================================================================

def setup_logging(cfg):
    """Console + rotating file logger."""
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logger = logging.getLogger("code_geom")
    logger.setLevel(logging.DEBUG)
    fmt_console = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    fmt_file = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)
    fh = RotatingFileHandler(logs_dir / "pipeline.log", maxBytes=10 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)
    return logger


# ====================================================================
# 3. LOAD CODE PROBLEMS (generic: driven only by dataset params in config)
# ====================================================================

def _resolve_json_path(path_str, cfg):
    path = Path(path_str)
    if not path.is_absolute():
        base = Path(cfg.get("_config_base", Path.cwd()))
        path = (base / path).resolve()
    return path


def _parse_problem_rows(problems_raw, d):
    """Normalize list of JSON objects to internal problem dicts."""
    max_p = d.get("max_problems")
    if max_p is not None:
        problems_raw = problems_raw[:max_p]
    pk = d.get("prompt_key", "prompt")
    tk = d.get("test_cases_key", "test_cases")
    tidk = d.get("task_id_key", "task_id")
    epk = d.get("entry_point_key", "entry_point")
    out = []
    for i, p in enumerate(problems_raw):
        if not isinstance(p, dict):
            continue
        prompt = p.get(pk, p.get("problem", ""))
        test_cases = p.get(tk, p.get("test", ""))
        task_id = p.get(tidk, f"task_{i}")
        entry_point = p.get(epk, "check")
        out.append({"task_id": task_id, "prompt": prompt, "test_cases": test_cases, "entry_point": entry_point})
    return out


def _load_problems_from_json(cfg, logger):
    """Load problems from a JSON file. Params: json_path, max_problems; keys: prompt_key, test_cases_key, task_id_key, entry_point_key, list_key (defaults in config)."""
    d = cfg["dataset"]
    path = _resolve_json_path(d["json_path"], cfg)
    with open(path) as f:
        data = json.load(f)
    list_key = d.get("list_key")  # None = root is list; else e.g. "problems" or "data"
    problems = data if list_key is None else data.get(list_key, [])
    if not isinstance(problems, list):
        raise ValueError(f"dataset.list_key must point to a list; got {type(problems)}")
    out = _parse_problem_rows(problems, d)
    level_run_id = d.get("level_run_id", "custom")
    logger.info(f"Loaded JSON {path}: level_run_id={level_run_id} n={len(out)}")
    return {level_run_id: {"problems": out, "n_problems": len(out)}}


def _resolve_run_level_keys(levels, run_levels, logger):
    """Which keys from `levels` to load. run_levels None -> all keys sorted numerically.
    run_levels: list of ints (e.g. [1, 3]) -> only those levels; each must exist in levels."""
    all_sorted = sorted(levels.keys(), key=lambda x: int(str(x)))
    if run_levels is None:
        return all_sorted
    if not isinstance(run_levels, (list, tuple)):
        raise ValueError("dataset.run_levels must be null/omitted (all levels) or a list of level numbers, e.g. [1, 3]")
    if len(run_levels) == 0:
        raise ValueError("dataset.run_levels is empty; use null or omit to run all levels")
    defined = {int(str(k)): k for k in levels.keys()}
    want_nums = [int(str(x)) for x in run_levels]
    for w in want_nums:
        if w not in defined:
            raise ValueError(
                f"dataset.run_levels includes {w} but dataset.levels has no such level; "
                f"available: {sorted(defined.keys())}"
            )
    uniq_sorted = sorted(set(want_nums))
    chosen = [defined[w] for w in uniq_sorted]
    logger.info(
        "run_levels=%s -> loading %s (all defined levels: %s)",
        want_nums,
        [int(str(k)) for k in chosen],
        [int(str(k)) for k in all_sorted],
    )
    return chosen


def _load_problems_from_json_levels(cfg, logger):
    """Load one JSON file per Level (fixed problems on disk per tier).
    Params: levels: { 1: path, 2: path, ... }. run_levels: [1, 2] restricts which levels; null = all.
    Per-level output tags are level_01, level_02, ... (zero-padded).
    Same keys as json: prompt_key, test_cases_key, list_key, max_problems (applied per level)."""
    d = cfg["dataset"]
    levels = d.get("levels")
    if not isinstance(levels, dict) or not levels:
        raise ValueError("dataset.source is 'json_levels' but dataset.levels must be a non-empty map of level -> json path")
    run_levels = d.get("run_levels")
    level_keys = _resolve_run_level_keys(levels, run_levels, logger)
    all_level_runs = {}
    for lvl in level_keys:
        rel = levels[lvl]
        path = _resolve_json_path(rel, cfg)
        with open(path) as f:
            data = json.load(f)
        list_key = d.get("list_key")
        problems = data if list_key is None else data.get(list_key, [])
        if not isinstance(problems, list):
            raise ValueError(f"Level {lvl}: list_key must point to a list; got {type(problems)}")
        out = _parse_problem_rows(problems, d)
        level_run_id = f"level_{int(lvl):02d}"
        all_level_runs[level_run_id] = {"problems": out, "n_problems": len(out)}
        logger.info("Level %s -> level_run_id %s: %s problems from %s", lvl, level_run_id, len(out), path)
    return all_level_runs


def _load_problems_from_huggingface(cfg, logger):
    """Load problems from HuggingFace datasets. Params: hf_repo, hf_split; hf_config; prompt_column, task_id_column, entry_point_column; test_cases_column (string) OR test_list_column (+ test_setup_column when used); task_id_prefix, level_run_id."""
    from datasets import load_dataset
    d = cfg["dataset"]
    repo = d["hf_repo"]
    config = d.get("hf_config")
    hf_subset = d.get("hf_split", "test")
    ds = load_dataset(repo, config) if config else load_dataset(repo)
    rows = ds[hf_subset]
    n = len(rows)
    max_p = d.get("max_problems")
    if max_p is not None:
        n = min(n, max_p)
    prompt_col = d.get("prompt_column", "prompt")
    task_id_col = d.get("task_id_column", "task_id")
    entry_point_col = d.get("entry_point_column", "entry_point")
    task_id_prefix = d.get("task_id_prefix", "")
    # Test cases: either single string column or list column (+ setup column when configured)
    test_cases_col = d.get("test_cases_column")
    test_list_col = d.get("test_list_column")
    test_setup_col = d.get("test_setup_column")
    if test_cases_col and test_list_col:
        raise ValueError("Use either test_cases_column or test_list_column, not both")
    if not test_cases_col and not test_list_col:
        test_cases_col = "test"
    problems = []
    for i in range(n):
        row = rows[i]
        prompt = row.get(prompt_col, row.get("prompt", row.get("text", "")))
        task_id = row.get(task_id_col, i)
        if not isinstance(task_id, str) and task_id_prefix:
            task_id = f"{task_id_prefix}{task_id}"
        elif not isinstance(task_id, str):
            task_id = str(task_id)
        if test_cases_col:
            test_cases = row.get(test_cases_col, "")
        else:
            test_list = row.get(test_list_col, [])
            test_cases = "\n".join(test_list) if isinstance(test_list, list) else (test_list or "")
            if test_setup_col:
                setup = row.get(test_setup_col, "") or ""
                if str(setup).strip():
                    test_cases = str(setup).strip() + "\n" + test_cases
        entry_point = row.get(entry_point_col, "check")
        problems.append({"task_id": task_id, "prompt": prompt, "test_cases": test_cases, "entry_point": entry_point})
    level_run_id = d.get("level_run_id", hf_subset)
    logger.info(f"Loaded HuggingFace {repo} (hf_split={hf_subset}): {len(problems)} problems -> level_run_id={level_run_id}")
    return {level_run_id: {"problems": problems, "n_problems": len(problems)}}


def load_code_problems(cfg, logger):
    """Load code problems from the dataset specified by config. Only uses params under cfg['dataset'].
    source: 'json' | 'json_levels' | 'huggingface'. See config.yaml and docs/datasets.md.
    Returns dict[level_run_id] -> {problems, n_problems} (json_levels uses level_01, level_02, …)."""
    source = cfg["dataset"].get("source", "json")
    if source == "json":
        if "json_path" not in cfg["dataset"]:
            raise ValueError("dataset.source is 'json' but dataset.json_path is missing")
        return _load_problems_from_json(cfg, logger)
    if source == "json_levels":
        return _load_problems_from_json_levels(cfg, logger)
    if source == "huggingface":
        if "hf_repo" not in cfg["dataset"]:
            raise ValueError("dataset.source is 'huggingface' but dataset.hf_repo is missing")
        return _load_problems_from_huggingface(cfg, logger)
    raise ValueError(f"Unknown dataset.source: {source}. Use 'json', 'json_levels', or 'huggingface'.")


# ====================================================================
# 4. LABELS (per problem, algorithm-agnostic metadata)
# ====================================================================

def compute_labels_for_problem(prob):
    """Labels for one code problem (no execution yet)."""
    return {
        "task_id": prob["task_id"],
        "prompt": prob["prompt"],
        "num_test_lines": len([x for x in (prob.get("test_cases") or "").strip().split("\n") if x.strip()]),
        "entry_point": prob.get("entry_point", "check"),
    }


def compute_all_labels(all_problems, logger):
    """Labels for every problem in every level run."""
    all_labels = {}
    for level_run_id, data in all_problems.items():
        all_labels[level_run_id] = [compute_labels_for_problem(p) for p in data["problems"]]
        logger.info(f"Level run {level_run_id}: computed {len(all_labels[level_run_id])} label sets")
    return all_labels


# ====================================================================
# 5. PROMPT FORMATTING
# ====================================================================

def format_prompts(all_problems, cfg):
    """Build prompt strings per level_run_id. Instruction prefix comes from config when set."""
    prefix = cfg["dataset"].get("prompt_prefix", "")
    return {
        level_run_id: [prefix + p["prompt"] for p in data["problems"]]
        for level_run_id, data in all_problems.items()
    }


# ====================================================================
# 6. SAVE DATASETS
# ====================================================================

def save_datasets(all_problems, all_labels, all_prompts, cfg, logger):
    """Write per level_run_id JSON to workspace/labels/ (``level_run_<id>.json``)."""
    out_dir = Path(cfg["paths"]["labels_dir"])
    for level_run_id in sorted(all_problems):
        data = all_problems[level_run_id]
        problems = data["problems"]
        dataset = {
            "level_run_id": level_run_id,
            "n_problems": len(problems),
            "problems": [
                {"index": i, "prompt": all_prompts[level_run_id][i], "labels": all_labels[level_run_id][i]}
                for i in range(len(problems))
            ],
        }
        path = out_dir / f"level_run_{level_run_id}.json"
        with open(path, "w") as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Saved {path.name}")


# ====================================================================
# 7. TOKENIZER / MODEL
# ====================================================================

def load_tokenizer(cfg, logger):
    """Load tokenizer, set pad token.
    Prefer LlamaTokenizer (SentencePiece from tokenizer.model). If tokenizer.json
    causes 'Merges text file invalid' we temporarily hide it and load from tokenizer.model.
    """
    from pathlib import Path
    name = cfg["model"]["name"]
    logger.info(f"Loading tokenizer: {name}")
    model_dir = Path(name)
    tokenizer_json = model_dir / "tokenizer.json"
    json_hidden = False
    try:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(name)
    except Exception as e:
        err_msg = str(e)
        if "Merges" in err_msg and tokenizer_json.exists():
            logger.warning("Hiding tokenizer.json and loading from tokenizer.model only")
            tokenizer_json.rename(tokenizer_json.with_suffix(".json.bak"))
            json_hidden = True
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(name)
        else:
            raise
    finally:
        if json_hidden and tokenizer_json.with_suffix(".json.bak").exists():
            tokenizer_json.with_suffix(".json.bak").rename(tokenizer_json)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def log_prompt_and_tokens(tokenizer, all_prompts, logger, sample_per_level_run=None, max_tokens_to_show=80, max_chars=500):
    """Log prompt(s) per level run: prompt text, token count, token ids, and decoded tokens.
    sample_per_level_run: how many prompts to log per run (None = all, up to 50).
    """
    max_to_log = 50 if sample_per_level_run is None else sample_per_level_run
    for level_run_id in sorted(all_prompts):
        prompts = all_prompts[level_run_id]
        n_log = min(len(prompts), max_to_log)
        logger.info(f"[{level_run_id}] logging prompt/tokens for {n_log} of {len(prompts)} prompts")
        for idx in range(n_log):
            prompt = prompts[idx]
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            ids = enc["input_ids"][0].tolist()
            n = len(ids)
            tokens_preview = ids[:max_tokens_to_show] if n > max_tokens_to_show else ids
            decoded_tokens = tokenizer.convert_ids_to_tokens(ids)
            decoded_preview = decoded_tokens[:max_tokens_to_show] if n > max_tokens_to_show else decoded_tokens
            logger.info(f"[{level_run_id}] prompt index={idx} | num_tokens={n}")
            logger.info(f"[{level_run_id}] prompt: %s", repr(prompt[:max_chars] + ("..." if len(prompt) > max_chars else "")))
            logger.info(f"[{level_run_id}] token_ids: %s", tokens_preview)
            logger.info(f"[{level_run_id}] decoded_tokens: %s", decoded_preview)


def load_model(cfg, logger):
    """Load model. Returns (model, device)."""
    import torch
    from transformers import AutoModelForCausalLM
    name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])
    logger.info(f"Loading model: {name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    dev_str = str(device)
    using_gpu = "cuda" in dev_str
    using_mps = "mps" in dev_str
    logger.info(f"Model loaded in {time.time()-t0:.1f}s on {device} (GPU/cuda: {using_gpu}, MPS: {using_mps})")
    return model, device


# ====================================================================
# 8. ACTIVATION EXTRACTION
# ====================================================================

def make_hook(storage, layer_idx):
    import torch
    def hook_fn(module, input, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        storage[layer_idx] = hidden[:, -1, :].detach().float().cpu()
    return hook_fn


def extract_activations(model, tokenizer, all_prompts, cfg, logger):
    """Extract residual-stream activations at last prompt token per level run."""
    import torch
    layers = cfg["model"]["layers"]
    batch_size = cfg["generation"]["batch_size"]
    act_dir = Path(cfg["paths"]["activations_dir"])
    hidden_dim = cfg["model"]["hidden_dim"]

    for level_run_id in sorted(all_prompts):
        prompts = all_prompts[level_run_id]
        n = len(prompts)
        captured = {}
        hooks = []
        for layer in layers:
            h = model.model.layers[layer].register_forward_hook(make_hook(captured, layer))
            hooks.append(h)
        level_acts = {layer: [] for layer in layers}
        logger.info(f"Level run {level_run_id}: extracting {n} prompts x {len(layers)} layers")
        t0 = time.time()
        with torch.no_grad():
            for bs in range(0, n, batch_size):
                be = min(bs + batch_size, n)
                batch = prompts[bs:be]
                inputs = tokenizer(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
                model(**inputs)
                for layer in layers:
                    level_acts[layer].append(captured[layer].numpy().astype(np.float32))
                captured.clear()
        for h in hooks:
            h.remove()
        for layer in layers:
            arr = np.concatenate(level_acts[layer], axis=0)
            np.save(act_dir / f"level_run_{level_run_id}_layer{layer}.npy", arr)
        logger.info(f"Level run {level_run_id}: extraction done in {time.time()-t0:.1f}s")
    logger.info("Activation extraction complete")


# ====================================================================
# 9. CODE EXTRACTION FROM GENERATION
# ====================================================================

def _prompt_is_code_stub(prompt: str) -> bool:
    """True if the prompt is incomplete source that should be prepended before generated body.

    HumanEval-style tasks often start with ``from typing import ...`` then ``def foo(...):``.
    Evaluation must merge ``prompt + body + tests``. Natural-language-only prompts use
    ``generated + tests`` only.
    """
    p = prompt.strip()
    if not p:
        return False
    if p.startswith("class "):
        return True
    if p.startswith("def "):
        return True
    if re.search(r"^\s*(?:async\s+)?def\s+\w+\s*\(", p, re.MULTILINE):
        return True
    return False


def _extract_body_after_first_def(block: str):
    """Lines inside the first top-level ``def`` (not ``def check``), excluding the ``def`` line.

    Used when the prompt already contains the signature (stub merge): model may return imports +
    full ``def`` + body; we must not treat the first ``def`` as "stop before" (which would leave
    only import lines — see ``_take_first_function_body``).
    """
    lines = block.split("\n")
    def_indent = None
    def_line_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("def ") and not s.startswith("def check("):
            def_indent = len(line) - len(line.lstrip(" \t"))
            def_line_idx = i
            break
        if s.startswith("async def ") and "def check(" not in s:
            def_indent = len(line) - len(line.lstrip(" \t"))
            def_line_idx = i
            break
    if def_line_idx is None:
        return None
    body = []
    for line in lines[def_line_idx + 1 :]:
        if not line.strip():
            body.append(line)
            continue
        li = len(line) - len(line.lstrip(" \t"))
        if li <= def_indent and line.strip():
            break
        body.append(line)
    text = "\n".join(body).rstrip()
    return text if text.strip() else None


def _block_has_import_before_first_def(block: str) -> bool:
    """True if the first logical line is ``from``/``import`` before any top-level ``def``.

    Used to gate ``_extract_body_after_first_def``: that helper fixes HumanEval-style blocks
    (imports + ``def``) where ``_take_first_function_body`` would keep only imports. For
    prompts that already start with ``def`` (e.g. level 1), the model usually emits only a
    body or a single ``def``; always preferring *first* ``def`` in the block can steal the
    wrong function when a helper ``def`` appears first — so we only force first-def body
    extraction when imports precede the first ``def`` in the model output, or when the
    prompt itself is not ``def``-first (e.g. ``from typing ...`` then ``def``).
    """
    for line in block.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("def ") and not s.startswith("def check("):
            return False
        if s.startswith("async def ") and "def check(" not in s:
            return False
        if s.startswith("from ") or s.startswith("import "):
            return True
    return False


def _use_extract_body_after_first_def_for_stub(prompt: str, block: str) -> bool:
    """Whether stub-mode extraction should peel body from the first ``def`` in ``block``."""
    p = (prompt or "").strip()
    if not p:
        return False
    # Level 5-style stub: not led by ``def`` → always try (fixes import-only collapse).
    if not p.startswith("def "):
        return True
    # Level 1-style ``def``-first stub: only when model echoed imports before its ``def``.
    return _block_has_import_before_first_def(block)


def extract_code_from_generation(raw_text, problem=None):
    """Extract code block from model output (e.g. ```python ... ```).

    For code-stub prompts (see ``_prompt_is_code_stub``), prefer the indented body under the
    first ``def`` in the model block when present, so HumanEval-style ``import`` + ``def`` output
    does not collapse to import-only text. For natural-language prompts, keep prior behavior
    (full function when the block starts with ``def``, etc.).
    """
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    stub = problem is not None and _prompt_is_code_stub((problem.get("prompt") or ""))
    # Try markdown code block
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        block = m.group(1).rstrip()
        if stub and _use_extract_body_after_first_def_for_stub(
            (problem.get("prompt") or "") if problem else "", block
        ):
            inner = _extract_body_after_first_def(block)
            if inner:
                return inner.rstrip()
        return _take_first_function_body(block)
    # Otherwise take lines until assert, def check, or next top-level def
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("assert ") or s.startswith("def check"):
            break
        if s.startswith("def ") and not s.startswith("def check("):
            break
        code_lines.append(line)
    joined = "\n".join(code_lines).rstrip() if code_lines else ""
    if stub and joined and _use_extract_body_after_first_def_for_stub(
        (problem.get("prompt") or "") if problem else "", text
    ):
        inner = _extract_body_after_first_def(text)
        if inner:
            return inner.rstrip()
    return joined if code_lines else (text.strip() or None)


def _take_first_function_body(block):
    """From a code block, keep only the first function body (stop at next top-level def)."""
    lines = block.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("def ") and not s.startswith("def check("):
            break
        if s.startswith("assert "):
            break
        out.append(line)
    return "\n".join(out).rstrip() if out else block.rstrip()


def _leading_indent(line):
    """Length of leading spaces/tabs (tabs counted as single chars)."""
    return len(line) - len(line.lstrip(" \t"))


def _ensure_indented_body(generated_code):
    """Normalize model output so it is a valid body under ``def ...:`` + docstring.

    Models often emit a *mixed* block: first line at column 0 (``total = 0``) while
    the rest is already indented as if inside a function (``    for ...``). The naive
    fix of prefixing four spaces to *every* line double-indents those lines and raises
    ``IndentationError`` in subprocess eval.

    Rules:
    - All non-empty lines flush left → indent each line by 4 spaces.
    - Min indent 0 but some lines deeper → add 4 spaces only to lines at min indent.
    - Uniform min indent > 0 → shift so smallest line becomes 4 spaces (embed under def).
    """
    if not generated_code or not generated_code.strip():
        return generated_code
    lines = generated_code.rstrip().split("\n")
    if not lines:
        return generated_code

    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return generated_code

    leads = [_leading_indent(ln) for ln in nonempty]
    m, M = min(leads), max(leads)

    # Every statement flush left — indent whole block as function body.
    if m == 0 and M == 0:
        return "\n".join("    " + ln if ln.strip() else ln for ln in lines)

    # First line(s) forgot indent; deeper lines already correct — fix only min-indent lines.
    if m == 0 and M > 0:
        out = []
        for ln in lines:
            if not ln.strip():
                out.append(ln)
                continue
            if _leading_indent(ln) == m:
                out.append("    " + ln.lstrip(" \t"))
            else:
                out.append(ln)
        return "\n".join(out)

    # Body already uses a positive baseline indent — rebase to 4 spaces for ``def`` body.
    if m > 0:
        out = []
        for ln in lines:
            if not ln.strip():
                out.append("")
                continue
            li = _leading_indent(ln)
            rest = ln.lstrip(" \t")
            out.append(" " * (li - m + 4) + rest)
        return "\n".join(out)

    return generated_code


# ====================================================================
# 10. CODE EVALUATION (run in subprocess with timeout)
# ====================================================================

def run_code_evaluation(generated_code, problem, cfg, logger):
    """
    Run generated code against test_cases in a subprocess. Returns (correct: bool, error_category: str).
    """
    import subprocess
    if generated_code is None or not generated_code.strip():
        return False, "garbage"
    timeout = cfg["evaluation"].get("timeout_seconds", 5)
    test_cases = problem.get("test_cases") or ""
    if not test_cases.strip():
        return False, "no_tests"
    prompt = problem.get("prompt", "")
    prompt_stripped = prompt.strip()
    # If prompt is a code stub (e.g. "def foo(x):" or "from typing import List\\n\\ndef foo(...):"),
    # prepend it and use generated code as body. If prompt is plain natural language, run generated + tests only.
    if _prompt_is_code_stub(prompt_stripped):
        body = _ensure_indented_body(generated_code.rstrip())
        full_code = prompt.rstrip() + "\n" + body + "\n" + test_cases.strip()
    else:
        full_code = generated_code.rstrip() + "\n" + test_cases.strip()
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None,
        )
        if result.returncode == 0:
            return True, "correct"
        stderr = (result.stderr or "").strip()
        if stderr:
            logger.info("evaluation stderr: %s", stderr[:800])
        if "SyntaxError" in stderr or "IndentationError" in stderr:
            return False, "syntax_error"
        if "AssertionError" in stderr or result.returncode != 0:
            return False, "logic_error"
        return False, "runtime_error"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        logger.debug("run_code_evaluation exception: %s", e)
        return False, "runtime_error"


def _evaluation_self_check(cfg, logger):
    """Run a known-passing snippet to confirm subprocess evaluation is active."""
    fake_prob = {
        "prompt": "def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
        "test_cases": "assert add_one(0) == 1\nassert add_one(2) == 3",
    }
    generated = "    return x + 1"
    full_code = fake_prob["prompt"].rstrip() + "\n" + generated.rstrip() + "\n" + fake_prob["test_cases"].strip()
    ok, cat = run_code_evaluation(generated, fake_prob, cfg, logger)
    if ok and cat == "correct":
        logger.info("Code evaluation self-check: subprocess execution is ACTIVE (1/1 passed)")
    else:
        logger.warning("Code evaluation self-check FAILED (got %s). full_code passed to Python:", cat)
        logger.warning("---\n%s\n---", full_code[:600])


def evaluate_all_answers(all_raw_answers, all_problems, all_labels, cfg, logger):
    """For each level run, extract code and evaluate; return all_answers dict."""
    logger.info("Evaluating generated code in subprocess (timeout=%s s)", cfg["evaluation"].get("timeout_seconds", 5))
    _evaluation_self_check(cfg, logger)

    all_answers = {}
    for level_run_id in sorted(all_raw_answers):
        raw_list = all_raw_answers[level_run_id]
        problems = all_problems[level_run_id]["problems"]
        labels = all_labels[level_run_id]
        results = []
        for i, (raw, prob, lab) in enumerate(zip(raw_list, problems, labels)):
            code = extract_code_from_generation(raw, prob)
            correct, error_cat = run_code_evaluation(code, prob, cfg, logger)
            code_preview = (code or "(none)")[:500] + ("..." if code and len(code) > 500 else "")
            logger.info(f"[{level_run_id}] index={i} task_id={lab['task_id']} | {error_cat} | generated_code: %s", code_preview)
            results.append({
                "index": i,
                "raw_text": raw,
                "generated_code": code,
                "correct": correct,
                "error_category": error_cat,
                "task_id": lab["task_id"],
            })
        all_answers[level_run_id] = results
        n_correct = sum(1 for r in results if r["correct"])
        counts = Counter(r["error_category"] for r in results)
        logger.info(f"Level run {level_run_id}: {n_correct}/{len(results)} correct — %s", dict(counts))
    return all_answers


# ====================================================================
# 11. ANSWER GENERATION (greedy decode)
# ====================================================================

def generate_raw(model, tokenizer, all_prompts, cfg, logger):
    """Greedy decode all prompts. Returns dict[level_run_id] -> list of raw generated strings."""
    import torch
    batch_size = cfg["generation"]["batch_size"]
    max_new = cfg["generation"]["max_new_tokens"]
    device = next(model.parameters()).device
    all_raw = {}
    for level_run_id in sorted(all_prompts):
        prompts = all_prompts[level_run_id]
        n = len(prompts)
        results = []
        logger.info(f"Level run {level_run_id}: generating for {n} problems")
        logger.info("Running model.generate() in batches (batch_size=%s, max_new_tokens=%s) — wait for 'generation done' log.", batch_size, max_new)
        t0 = time.time()
        with torch.no_grad():
            for bs in range(0, n, batch_size):
                be = min(bs + batch_size, n)
                batch = prompts[bs:be]
                inputs = tokenizer(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for i in range(len(batch)):
                    gen_ids = outputs[i][input_len:]
                    results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        logger.info(f"Level run {level_run_id}: generation done in {time.time()-t0:.1f}s")
        all_raw[level_run_id] = results
    return all_raw


# ====================================================================
# 12. SAVE ANSWERS
# ====================================================================

def save_answers(all_answers, all_labels, cfg, logger):
    """Write per level_run_id answer JSON (``level_run_<id>.json``)."""
    ans_dir = Path(cfg["paths"]["answers_dir"])
    accuracies = {}
    for level_run_id in sorted(all_answers):
        answers = all_answers[level_run_id]
        labels = all_labels[level_run_id]
        n_correct = sum(1 for a in answers if a["correct"])
        n_total = len(answers)
        acc = n_correct / n_total if n_total else 0.0
        accuracies[level_run_id] = acc
        output = {
            "level_run_id": level_run_id,
            "n_problems": n_total,
            "n_correct": n_correct,
            "accuracy": acc,
            "results": answers,
        }
        path = ans_dir / f"level_run_{level_run_id}.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Level run {level_run_id}: accuracy {acc:.1%} ({n_correct}/{n_total})")
    return accuracies


# ====================================================================
# 13. DIAGNOSTIC PLOTS
# ====================================================================

def generate_plots(all_prompts, all_problems, accuracies, cfg, logger, all_answers=None):
    """Save diagnostic plots (accuracy, activation norms, per level_run correct vs wrong norms, problem counts)."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    layers = cfg["model"]["layers"]
    act_dir = Path(cfg["paths"]["activations_dir"])
    level_run_ids = sorted(all_prompts)

    # Bar + line accuracy (level runs often ordered by curriculum level)
    fig, ax = plt.subplots(figsize=(9, 5))
    accs = [accuracies.get(s, 0) for s in level_run_ids]
    x = np.arange(len(level_run_ids))
    ax.bar(x, accs, color="steelblue", alpha=0.85, label="accuracy")
    ax.plot(x, accs, color="darkblue", marker="o", linewidth=2, markersize=6, label="trend")
    ax.set_xticks(x)
    ax.set_xticklabels(level_run_ids, rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Code generation accuracy by level run (curriculum tier)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_by_level_run.png", dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy_by_level_run.png")

    # Problems per level run
    figp, axp = plt.subplots(figsize=(8, 4))
    counts = [len(all_prompts[s]) for s in level_run_ids]
    axp.bar(x, counts, color="teal", alpha=0.85)
    axp.set_xticks(x)
    axp.set_xticklabels(level_run_ids, rotation=25, ha="right")
    axp.set_ylabel("Number of problems")
    axp.set_title("Problems per level run")
    figp.tight_layout()
    figp.savefig(plots_dir / "problems_per_level_run.png", dpi=150)
    plt.close(figp)
    logger.info("Saved problems_per_level_run.png")

    if act_dir.exists():
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for rid in level_run_ids:
            means = []
            for layer in layers:
                fpath = act_dir / f"level_run_{rid}_layer{layer}.npy"
                if fpath.exists():
                    arr = np.load(fpath, mmap_mode="r")
                    means.append(np.linalg.norm(arr, axis=1).mean())
                else:
                    means.append(np.nan)
            ax2.plot(layers, means, marker="o", label=rid)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Mean L2 Norm")
        ax2.set_title("Activation Norm Profile (all samples)")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(plots_dir / "activation_norm_profile.png", dpi=150)
        plt.close(fig2)
        logger.info("Saved activation_norm_profile.png")

        # Per level_run_id: correct vs wrong mean activation norm vs layer (populations: correct / wrong)
        if all_answers:
            n_runs = len(level_run_ids)
            ncols = min(3, max(1, n_runs))
            nrows = math.ceil(n_runs / ncols)
            fig3, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
            for idx, rid in enumerate(level_run_ids):
                r, c = divmod(idx, ncols)
                ax3 = axes[r][c]
                results = all_answers.get(rid) or []
                if not results:
                    ax3.set_title(f"{rid} (no answers)")
                    continue
                correct_mask = np.array([bool(r.get("correct")) for r in results], dtype=bool)
                for flag, name, color in [(True, "correct", "green"), (False, "wrong", "red")]:
                    m = correct_mask if flag else ~correct_mask
                    if m.sum() == 0:
                        continue
                    means = []
                    for layer in layers:
                        fpath = act_dir / f"level_run_{rid}_layer{layer}.npy"
                        if not fpath.exists():
                            means.append(np.nan)
                            continue
                        arr = np.load(fpath, mmap_mode="r")
                        if arr.shape[0] != len(results):
                            means.append(np.nan)
                            continue
                        means.append(float(np.linalg.norm(arr[m], axis=1).mean()))
                    ax3.plot(layers, means, marker="o", label=name, color=color)
                ax3.set_xlabel("Layer")
                ax3.set_ylabel("Mean L2 norm")
                ax3.set_title(rid)
                ax3.legend(fontsize=8)
            for j in range(len(level_run_ids), nrows * ncols):
                r, c = divmod(j, ncols)
                axes[r][c].set_visible(False)
            fig3.suptitle("Activation norm: correct vs wrong (populations)", fontsize=12, y=1.02)
            fig3.tight_layout()
            fig3.savefig(plots_dir / "activation_norm_correct_wrong.png", dpi=150, bbox_inches="tight")
            plt.close(fig3)
            logger.info("Saved activation_norm_correct_wrong.png")


# ====================================================================
# 14. MAIN
# ====================================================================

def main(config_path=None):
    t_start = time.time()
    cfg = load_config(config_path)
    logger = setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("Code Geometry Pipeline")
    logger.info("=" * 60)

    logger.info("--- Loading code problems ---")
    all_problems = load_code_problems(cfg, logger)
    logger.info("--- Computing labels ---")
    all_labels = compute_all_labels(all_problems, logger)
    all_prompts = format_prompts(all_problems, cfg)
    logger.info("--- Saving datasets ---")
    save_datasets(all_problems, all_labels, all_prompts, cfg, logger)

    logger.info("--- Loading tokenizer ---")
    tokenizer = load_tokenizer(cfg, logger)
    log_prompt_and_tokens(tokenizer, all_prompts, logger)
    logger.info("--- Loading model ---")
    model, _ = load_model(cfg, logger)
    logger.info("--- Extracting activations ---")
    extract_activations(model, tokenizer, all_prompts, cfg, logger)
    logger.info("--- Generating answers ---")
    all_raw = generate_raw(model, tokenizer, all_prompts, cfg, logger)
    logger.info("--- Evaluating code ---")
    all_answers = evaluate_all_answers(all_raw, all_problems, all_labels, cfg, logger)
    logger.info("--- Saving answers ---")
    accuracies = save_answers(all_answers, all_labels, cfg, logger)
    logger.info("--- Generating plots ---")
    generate_plots(all_prompts, all_problems, accuracies, cfg, logger, all_answers=all_answers)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {(time.time()-t_start)/60:.1f} min")
    for level_run_id in sorted(accuracies):
        logger.info(f"  {level_run_id} accuracy: {accuracies[level_run_id]:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Geometry Pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
