#!/usr/bin/env python3
"""Pad data/levels/level*.json to at least TARGET problems (default 35) for Phase A / t-SNE.

Preserves existing problems (content + order). Appends synthetic micro-tasks as needed, then
rewrites every ``task_id`` to ``L{level}_001``, ``L{level}_002``, … in file order so ids stay
sorted and aligned with pipeline row indices.

Run from repo root:
  python scripts/pad_level_json_to_count.py
  python scripts/pad_level_json_to_count.py --target 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEVELS_DIR = REPO / "data" / "levels"
LEVEL_FILES = [LEVELS_DIR / f"level{i}.json" for i in range(1, 6)]


def pad_problem(level_index: int, seq: int) -> dict:
    """One synthetic problem; difficulty hint by level_index 1..5.

    ``task_id`` is a placeholder; :func:`assign_sequential_task_ids` overwrites
    all rows after padding so ids match file order (L1_001, L1_002, …).
    """
    tid = f"_pad_build_{level_index}_{seq}"
    # Level 1: constants
    if level_index == 1:
        v = 11 * seq + level_index
        return {
            "task_id": tid,
            "prompt": f"def const_{seq}(x):\n    \"\"\"Return {v} plus x.\"\"\"\n",
            "test_cases": f"assert const_{seq}(0) == {v}\nassert const_{seq}(1) == {v + 1}",
        }
    # Level 2: simple list / int
    if level_index == 2:
        return {
            "task_id": tid,
            "prompt": f"def scale_sum_{seq}(nums, k):\n    \"\"\"Return sum(n*k for n in nums).\"\"\"\n",
            "test_cases": f"assert scale_sum_{seq}([1, 2, 3], {seq % 3 + 1}) == sum(n*({seq % 3 + 1}) for n in [1,2,3])\nassert scale_sum_{seq}([], 5) == 0",
        }
    # Level 3: small logic
    if level_index == 3:
        th = seq % 5
        return {
            "task_id": tid,
            "prompt": f"def count_gt_{seq}(nums, t):\n    \"\"\"Count elements strictly greater than t.\"\"\"\n",
            "test_cases": f"assert count_gt_{seq}([{th}, {th+1}, {th+3}], {th+1}) == 1\nassert count_gt_{seq}([], 0) == 0",
        }
    # Level 4: string / structure
    if level_index == 4:
        return {
            "task_id": tid,
            "prompt": f"def dup_prefix_{seq}(s, n):\n    \"\"\"Return first min(n,len(s)) chars of s repeated twice.\"\"\"\n",
            "test_cases": f"assert dup_prefix_{seq}('abcde', 2) == 'abab'\nassert dup_prefix_{seq}('x', 5) == 'xx'",
        }
    # Level 5: slightly richer signature
    return {
        "task_id": tid,
        "prompt": f"from typing import List\n\ndef dot_step_{seq}(a: List[int], b: List[int]) -> int:\n    \"\"\"Return sum(a[i]*b[i]); assume same non-empty length.\"\"\"\n",
        "test_cases": f"assert dot_step_{seq}([1, 2], [3, 4]) == 11\nassert dot_step_{seq}([{seq % 7}], [2]) == ({seq % 7}) * 2",
    }


def assign_sequential_task_ids(problems: list, level_index: int) -> None:
    """Set ``task_id`` to L{level}_{001..} in list order (stable with pipeline row index)."""
    for i, p in enumerate(problems):
        p["task_id"] = f"L{level_index}_{i + 1:03d}"


def pad_file(path: Path, level_index: int, target: int) -> tuple[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected JSON array")
    out = [p for p in raw if isinstance(p, dict)]
    seq = 0
    while len(out) < target:
        out.append(pad_problem(level_index, seq))
        seq += 1
    assign_sequential_task_ids(out, level_index)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(raw), len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=35, help="Minimum problems per file (default 35)")
    args = ap.parse_args()
    target = max(args.target, 1)
    for i, path in enumerate(LEVEL_FILES, start=1):
        if not path.exists():
            raise SystemExit(f"missing {path}")
        before, after = pad_file(path, i, target)
        print(f"{path.name}: {before} -> {after} (target {target})")


if __name__ == "__main__":
    main()
