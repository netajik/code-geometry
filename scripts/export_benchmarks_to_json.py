#!/usr/bin/env python3
"""
Export HumanEval and MBPP to custom JSON format for code-geometry.

Run from repo root:
    pip install datasets
    python scripts/export_benchmarks_to_json.py

Outputs (in data/ by default):
    data/humaneval_custom.json   - HumanEval test subset (164 problems)
    data/mbpp_custom.json        - MBPP full test (974 problems)
    data/mbpp_sanitized_custom.json - MBPP sanitized test (427 problems)
"""

import json
import argparse
from pathlib import Path


def export_humaneval(out_path: Path) -> int:
    """Load HumanEval from HuggingFace, save as custom JSON. Returns count."""
    from datasets import load_dataset

    dataset = load_dataset("openai_humaneval", "openai_humaneval")
    rows = dataset["test"]
    problems = []
    for i in range(len(rows)):
        row = rows[i]
        problems.append({
            "task_id": row.get("task_id", f"humaneval/{i}"),
            "prompt": row.get("prompt", "") or row.get("problem", ""),
            "test_cases": row.get("test", ""),
            "entry_point": row.get("entry_point", "check"),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(problems, f, indent=2)
    return len(problems)


def export_mbpp(out_path: Path, sanitized: bool = False) -> int:
    """Load MBPP from HuggingFace, save as custom JSON. Returns count."""
    from datasets import load_dataset

    if sanitized:
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
        text_key = "prompt"
    else:
        dataset = load_dataset("google-research-datasets/mbpp")
        text_key = "text"

    rows = dataset["test"]
    problems = []
    for i in range(len(rows)):
        row = rows[i]
        prompt = row.get(text_key, "") or row.get("prompt", "") or row.get("text", "")
        task_id = row.get("task_id", i)
        if not isinstance(task_id, str):
            task_id = f"mbpp/{task_id}"
        test_list = row.get("test_list", [])
        test_cases = "\n".join(test_list) if isinstance(test_list, list) else (test_list or "")
        setup = row.get("test_setup_code", "") or ""
        if sanitized and row.get("test_imports"):
            setup = "\n".join(row["test_imports"]) + "\n" + setup
        if setup.strip():
            test_cases = setup.strip() + "\n" + test_cases
        problems.append({
            "task_id": task_id,
            "prompt": prompt,
            "test_cases": test_cases,
            "entry_point": "check",
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(problems, f, indent=2)
    return len(problems)


def main():
    parser = argparse.ArgumentParser(description="Export HumanEval and MBPP to custom JSON.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for JSON files (default: repo/data)",
    )
    parser.add_argument(
        "--humaneval-only",
        action="store_true",
        help="Export only HumanEval",
    )
    parser.add_argument(
        "--mbpp-only",
        action="store_true",
        help="Export only MBPP (full + sanitized unless --mbpp-full-only or --mbpp-sanitized-only)",
    )
    parser.add_argument(
        "--mbpp-full-only",
        action="store_true",
        help="Export only MBPP full test (974)",
    )
    parser.add_argument(
        "--mbpp-sanitized-only",
        action="store_true",
        help="Export only MBPP sanitized test (427)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir = out_dir.resolve()

    if not args.mbpp_only and not args.mbpp_full_only and not args.mbpp_sanitized_only:
        n = export_humaneval(out_dir / "humaneval_custom.json")
        print(f"Saved HumanEval: {out_dir / 'humaneval_custom.json'} ({n} problems)")
    if not args.humaneval_only:
        if not args.mbpp_sanitized_only:
            n = export_mbpp(out_dir / "mbpp_custom.json", sanitized=False)
            print(f"Saved MBPP full: {out_dir / 'mbpp_custom.json'} ({n} problems)")
        if not args.mbpp_full_only:
            n = export_mbpp(out_dir / "mbpp_sanitized_custom.json", sanitized=True)
            print(f"Saved MBPP sanitized: {out_dir / 'mbpp_sanitized_custom.json'} ({n} problems)")

    print("Done. Use in config.yaml: dataset.source: json, dataset.json_path: data/<file>.json")


if __name__ == "__main__":
    main()
