#!/usr/bin/env python3
"""
Generate data/levels/level[1-5].json — ``TASKS_PER_LEVEL`` tasks per level (default 100).

Design goals (aligned with common LLM code-gen failure reports: edge cases, composition,
longer specs, typing + structure, off-by-one, empty/singleton inputs):

  - No duplicate *prompt* strings across all tasks (global uniqueness check).
  - Distinct function names per task (lv{level}_t{nn}).
  - Every task validated via reference body + subprocess python -c.

Larger ``TASKS_PER_LEVEL`` improves Phase C/D stability (more correct/wrong rows per level).

Run from repo root:
  python scripts/generate_level_benchmarks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "levels"

# Per-level task count (tune here; 100 recommended for subspace / LDA sample size).
TASKS_PER_LEVEL = 100


def run_check(prompt: str, body: str, tests: str) -> None:
    code = prompt.rstrip() + "\n" + body.rstrip() + "\n" + tests.strip()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Validation failed:\nstderr={r.stderr!r}\n---\n{code[:1200]}")


def build_all_tasks() -> dict[int, list[dict]]:
    """Return level -> list of {task_id, prompt, test_cases}."""
    levels: dict[int, list[dict]] = {i: [] for i in range(1, 6)}
    seen_prompts: set[str] = set()

    def add(level: int, tid: str, prompt: str, tests: str, body: str) -> None:
        if prompt in seen_prompts:
            raise ValueError(f"Duplicate prompt for {tid}")
        seen_prompts.add(prompt)
        run_check(prompt, body, tests)
        levels[level].append({"task_id": tid, "prompt": prompt, "test_cases": tests.strip()})

    # --- Level 1: tiny arithmetic / bool / str (baseline “easy”) ---
    for n in range(1, TASKS_PER_LEVEL + 1):
        name = f"lv1_t{n:02d}"
        if n <= 15:
            specs = [
                (lambda: (f"def {name}(x):\n    \"\"\"Return x plus {n + 3}.\"\"\"\n", f"    return x + {n + 3}", f"assert {name}(0) == {n+3}\nassert {name}(-2) == {n+1}")),
                (lambda: (f"def {name}(x):\n    \"\"\"Return {2*n} times x.\"\"\"\n", f"    return {2*n} * x", f"assert {name}(0) == 0\nassert {name}(3) == {6*n}")),
                (lambda: (f"def {name}(a, b):\n    \"\"\"Return True iff a equals b.\"\"\"\n", "    return a == b", f"assert {name}(1, 1) is True\nassert {name}(1, 2) is False")),
                (lambda: (f"def {name}(s):\n    \"\"\"Return length of s.\"\"\"\n", "    return len(s)", f"assert {name}('') == 0\nassert {name}('abc') == 3")),
                (lambda: (f"def {name}(x):\n    \"\"\"Return absolute value of x.\"\"\"\n", "    return abs(x)", f"assert {name}(-{n}) == {n}\nassert {name}({n}) == {n}")),
            ]
            idx = (n - 1) % len(specs)
            pr, bd, ts = specs[idx]()
            add(1, f"L1_{n:03d}", pr, ts, bd)
        else:
            a, b = n, n + 7
            pr = f"def {name}(x):\n    \"\"\"Return ({a} * x) + {b}.\"\"\"\n"
            bd = f"    return ({a} * x) + {b}"
            ts = f"assert {name}(0) == {b}\nassert {name}(1) == {a + b}\nassert {name}(-1) == {-a + b}"
            add(1, f"L1_{n:03d}", pr, ts, bd)

    # --- Level 2: one-pass list/str, common slip: empty / negatives ---
    l2_cases = [
        ("L2_001", "def lv2_t01(nums):\n    \"\"\"Return sum of positive ints in nums; empty -> 0.\"\"\"\n", "assert lv2_t01([1, -2, 3]) == 4\nassert lv2_t01([]) == 0\nassert lv2_t01([-1, -2]) == 0", "    return sum(x for x in nums if x > 0)"),
        ("L2_002", "def lv2_t02(s):\n    \"\"\"Return number of vowels a,e,i,o,u (case insensitive) in s.\"\"\"\n", "assert lv2_t02('AeIoU') == 5\nassert lv2_t02('xyz') == 0\nassert lv2_t02('') == 0", "    return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("L2_003", "def lv2_t03(s):\n    \"\"\"Return s reversed (empty stays empty).\"\"\"\n", "assert lv2_t03('ab') == 'ba'\nassert lv2_t03('') == ''\nassert lv2_t03('x') == 'x'", "    return s[::-1]"),
        ("L2_004", "def lv2_t04(nums):\n    \"\"\"Return product of all elements; empty list -> 1.\"\"\"\n", "assert lv2_t04([2, 3]) == 6\nassert lv2_t04([]) == 1\nassert lv2_t04([5]) == 5", "    p = 1\n    for x in nums:\n        p *= x\n    return p"),
        ("L2_005", "def lv2_t05(nums):\n    \"\"\"Return index of first negative element, or -1 if none.\"\"\"\n", "assert lv2_t05([1, -3, 2]) == 1\nassert lv2_t05([1, 2]) == -1\nassert lv2_t05([-1]) == 0", "    for i, x in enumerate(nums):\n        if x < 0:\n            return i\n    return -1"),
        ("L2_006", "def lv2_t06(s, ch):\n    \"\"\"Return count of ch in s.\"\"\"\n", "assert lv2_t06('aba', 'a') == 2\nassert lv2_t06('', 'x') == 0", "    return s.count(ch)"),
        ("L2_007", "def lv2_t07(nums):\n    \"\"\"Return True if nums is strictly increasing (each < next). len<2 -> True.\"\"\"\n", "assert lv2_t07([1,2,3]) is True\nassert lv2_t07([1,1]) is False\nassert lv2_t07([]) is True", "    return all(nums[i] < nums[i+1] for i in range(len(nums)-1))"),
        ("L2_008", "def lv2_t08(a, b, c):\n    \"\"\"Return median of three ints (middle value when sorted).\"\"\"\n", "assert lv2_t08(1,2,3) == 2\nassert lv2_t08(5,5,1) == 5\nassert lv2_t08(9,1,7) == 7", "    return sorted([a,b,c])[1]"),
        ("L2_009", "def lv2_t09(n):\n    \"\"\"Return factorial n for n>=0; 0! = 1.\"\"\"\n", "assert lv2_t09(0) == 1\nassert lv2_t09(5) == 120\nassert lv2_t09(1) == 1", "    r = 1\n    for i in range(2, n+1):\n        r *= i\n    return r"),
        ("L2_010", "def lv2_t10(s):\n    \"\"\"Return True if s is palindrome (case sensitive).\"\"\"\n", "assert lv2_t10('aba') is True\nassert lv2_t10('ab') is False\nassert lv2_t10('') is True", "    return s == s[::-1]"),
        ("L2_011", "def lv2_t11(nums):\n    \"\"\"Return min value; assume nums non-empty.\"\"\"\n", "assert lv2_t11([3,1,2]) == 1\nassert lv2_t11([-5]) == -5", "    return min(nums)"),
        ("L2_012", "def lv2_t12(nums, k):\n    \"\"\"Return True if k appears in nums.\"\"\"\n", "assert lv2_t12([1,2], 2) is True\nassert lv2_t12([], 0) is False", "    return k in nums"),
        ("L2_013", "def lv2_t13(s):\n    \"\"\"Return first character or '' if empty.\"\"\"\n", "assert lv2_t13('hi') == 'h'\nassert lv2_t13('') == ''", "    return s[:1]"),
        ("L2_014", "def lv2_t14(nums):\n    \"\"\"Return last element; assume nums non-empty.\"\"\"\n", "assert lv2_t14([1,2,3]) == 3\nassert lv2_t14([9]) == 9", "    return nums[-1]"),
        ("L2_015", "def lv2_t15(x, lo, hi):\n    \"\"\"Clamp x into [lo, hi] (inclusive).\"\"\"\n", "assert lv2_t15(5, 0, 10) == 5\nassert lv2_t15(-1, 0, 10) == 0\nassert lv2_t15(99, 0, 10) == 10", "    return max(lo, min(hi, x))"),
        ("L2_016", "def lv2_t16(s):\n    \"\"\"Return s without spaces.\"\"\"\n", "assert lv2_t16('a b c') == 'abc'\nassert lv2_t16('') == ''", "    return s.replace(' ', '')"),
        ("L2_017", "def lv2_t17(nums):\n    \"\"\"Return number of even ints in nums.\"\"\"\n", "assert lv2_t17([1,2,4]) == 2\nassert lv2_t17([]) == 0", "    return sum(1 for x in nums if x % 2 == 0)"),
        ("L2_018", "def lv2_t18(a, b):\n    \"\"\"Return integer floor of a/b for b!=0.\"\"\"\n", "assert lv2_t18(7, 2) == 3\nassert lv2_t18(-7, 2) == -4", "    return a // b"),
        ("L2_019", "def lv2_t19(s):\n    \"\"\"Return True if all chars are digits.\"\"\"\n", "assert lv2_t19('123') is True\nassert lv2_t19('12a') is False\nassert lv2_t19('') is True", "    return all(c.isdigit() for c in s) if s else True"),
        ("L2_020", "def lv2_t20(nums):\n    \"\"\"Return sum of squares of nums.\"\"\"\n", "assert lv2_t20([1,2,3]) == 14\nassert lv2_t20([]) == 0", "    return sum(x*x for x in nums)"),
    ]
    for tid, pr, ts, bd in l2_cases:
        add(2, tid, pr, ts, bd)

    # Fill L2 021+ with parameterized list tasks
    for k in range(21, TASKS_PER_LEVEL + 1):
        name = f"lv2_t{k:02d}"
        m = k + 3
        pr = f"def {name}(nums):\n    \"\"\"Return sum of nums[i] for even indices i only.\"\"\"\n"
        bd = "    return sum(nums[i] for i in range(0, len(nums), 2))"
        ts = f"assert {name}([{m},1,2]) == {m + 2}\nassert {name}([]) == 0\nassert {name}([{m}]) == {m}"
        add(2, f"L2_{k:03d}", pr, ts, bd)

    # --- Level 3: edges + small algorithms (models often slip) ---
    l3 = [
        ("L3_001", "def lv3_t01(nums):\n    \"\"\"Return second largest *distinct* value, or None if fewer than 2 distinct.\"\"\"\n", "assert lv3_t01([5,5,3,9,9]) == 5\nassert lv3_t01([1]) is None\nassert lv3_t01([2,2]) is None", "    u = sorted(set(nums))\n    return u[-2] if len(u) >= 2 else None"),
        ("L3_002", "def lv3_t02(s):\n    \"\"\"Return longest run length of same character (empty -> 0).\"\"\"\n", "assert lv3_t02('aaabb') == 3\nassert lv3_t02('') == 0\nassert lv3_t02('x') == 1", "    if not s:\n        return 0\n    best = cur = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            cur += 1\n            best = max(best, cur)\n        else:\n            cur = 1\n    return best"),
        ("L3_003", "def lv3_t03(nums):\n    \"\"\"Return element that appears exactly once; others appear twice; assume one such exists.\"\"\"\n", "assert lv3_t03([2,1,2]) == 1\nassert lv3_t03([4,1,2,1,2]) == 4", "    r = 0\n    for x in nums:\n        r ^= x\n    return r"),
        ("L3_004", "def lv3_t04(intervals):\n    \"\"\"intervals is list of [a,b] with a<=b, sorted by a. Merge overlaps; return merged.\"\"\"\n", "assert lv3_t04([[1,3],[2,6]]) == [[1,6]]\nassert lv3_t04([[1,4],[4,5]]) == [[1,5]]\nassert lv3_t04([]) == []", "    if not intervals:\n        return []\n    out = [list(intervals[0])]\n    for a, b in intervals[1:]:\n        if a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out"),
        ("L3_005", "def lv3_t05(s):\n    \"\"\"Return number of balanced '(' ')' pairs as a prefix scan; unbalanced extra parens ignored at end.\"\"\"\n", "assert lv3_t05('(())') == 2\nassert lv3_t05('(()') == 1\nassert lv3_t05('') == 0", "    d = m = 0\n    for c in s:\n        if c == '(':\n            d += 1\n        elif c == ')':\n            if d:\n                d -= 1\n                m += 1\n    return m"),
        ("L3_006", "def lv3_t06(nums, k):\n    \"\"\"Rotate nums right by k steps; return new list (do not mutate input).\"\"\"\n", "assert lv3_t06([1,2,3,4], 1) == [4,1,2,3]\nassert lv3_t06([1,2], 0) == [1,2]\nassert lv3_t06([], 5) == []", "    if not nums:\n        return []\n    k %= len(nums)\n    return nums[-k:] + nums[:-k] if k else nums[:]"),
        ("L3_007", "def lv3_t07(words):\n    \"\"\"Return words joined by single space; skip empty strings.\"\"\"\n", "assert lv3_t07(['a','','b']) == 'a b'\nassert lv3_t07([]) == ''", "    return ' '.join(w for w in words if w)"),
        ("L3_008", "def lv3_t08(n):\n    \"\"\"Return binary string of n without '0b' prefix for n>=0.\"\"\"\n", "assert lv3_t08(5) == '101'\nassert lv3_t08(0) == '0'", "    return bin(n)[2:]"),
        ("L3_009", "def lv3_t09(nums):\n    \"\"\"Return list of prefix sums (length len(nums)+1, first 0).\"\"\"\n", "assert lv3_t09([1,2,3]) == [0,1,3,6]\nassert lv3_t09([]) == [0]", "    out = [0]\n    s = 0\n    for x in nums:\n        s += x\n        out.append(s)\n    return out"),
        ("L3_010", "def lv3_t10(s, t):\n    \"\"\"Return True if t is an anagram of s (same multiset of chars).\"\"\"\n", "assert lv3_t10('listen', 'silent') is True\nassert lv3_t10('a', 'ab') is False", "    return sorted(s) == sorted(t)"),
        ("L3_011", "def lv3_t11(grid):\n    \"\"\"grid is list of equal-length str rows; return number of 'X' chars.\"\"\"\n", "assert lv3_t11(['X.','.X']) == 2\nassert lv3_t11([]) == 0", "    return sum(row.count('X') for row in grid)"),
        ("L3_012", "def lv3_t12(nums):\n    \"\"\"Return length of longest contiguous subarray with sum 0; if none, 0.\"\"\"\n", "assert lv3_t12([1,-1,2]) == 2\nassert lv3_t12([1,2]) == 0\nassert lv3_t12([0]) == 1", "    seen = {0: -1}\n    s = 0\n    best = 0\n    for i, x in enumerate(nums):\n        s += x\n        if s in seen:\n            best = max(best, i - seen[s])\n        else:\n            seen[s] = i\n    return best"),
        ("L3_013", "def lv3_t13(s):\n    \"\"\"Return first non-repeating char or None if none.\"\"\"\n", "assert lv3_t13('aabbc') == 'c'\nassert lv3_t13('aabb') is None", "    from collections import Counter\n    c = Counter(s)\n    for ch in s:\n        if c[ch] == 1:\n            return ch\n    return None"),
        ("L3_014", "def lv3_t14(nums):\n    \"\"\"Return True if nums can become non-decreasing by changing at most one element.\"\"\"\n", "assert lv3_t14([4,2,3]) is True\nassert lv3_t14([3,4,2,3]) is False\nassert lv3_t14([1]) is True", "    a = list(nums)\n    c = 0\n    for i in range(1, len(a)):\n        if a[i - 1] > a[i]:\n            c += 1\n            if c > 1:\n                return False\n            if i >= 2 and a[i - 2] > a[i]:\n                a[i] = a[i - 1]\n            else:\n                a[i - 1] = a[i]\n    return True"),
        ("L3_015", "def lv3_t15(s):\n    \"\"\"Compress runs: 'aaab' -> [('a',3),('b',1)] as list of [char,count].\"\"\"\n", "assert lv3_t15('aaab') == [['a',3],['b',1]]\nassert lv3_t15('') == []", "    if not s:\n        return []\n    out = [[s[0], 1]]\n    for c in s[1:]:\n        if c == out[-1][0]:\n            out[-1][1] += 1\n        else:\n            out.append([c, 1])\n    return out"),
    ]
    for tid, pr, ts, bd in l3:
        add(3, tid, pr, ts, bd)

    for k in range(16, TASKS_PER_LEVEL + 1):
        name = f"lv3_t{k:02d}"
        base = k * 7
        pr = f"def {name}(n):\n    \"\"\"Return number of bits set in binary representation of n (n>=0).\"\"\"\n"
        bd = "    return bin(n).count('1')"
        ts = f"assert {name}(0) == 0\nassert {name}({base}) == {bin(base).count('1')}\nassert {name}(7) == 3"
        add(3, f"L3_{k:03d}", pr, ts, bd)

    # --- Level 4: harder composition / traps ---
    l4 = [
        ("L4_001", "def lv4_t01(nums):\n    \"\"\"Return max subarray sum (Kadane); empty -> 0.\"\"\"\n", "assert lv4_t01([-2,1,-3,4,-1,2,1,-5,4]) == 6\nassert lv4_t01([]) == 0\nassert lv4_t01([-1]) == 0", "    best = cur = 0\n    for x in nums:\n        cur = max(0, cur + x)\n        best = max(best, cur)\n    return best"),
        ("L4_002", "def lv4_t02(nums):\n    \"\"\"Return product of all nums except self without division; len>=1.\"\"\"\n", "assert lv4_t02([1,2,3,4]) == [24,12,8,6]\nassert lv4_t02([2]) == [1]", "    n = len(nums)\n    left = [1]*n\n    for i in range(1,n):\n        left[i] = left[i-1]*nums[i-1]\n    right = 1\n    out = [0]*n\n    for i in range(n-1,-1,-1):\n        out[i] = left[i]*right\n        right *= nums[i]\n    return out"),
        ("L4_003", "def lv4_t03(s):\n    \"\"\"Longest substring without repeating characters (length).\"\"\"\n", "assert lv4_t03('abcabcbb') == 3\nassert lv4_t03('') == 0\nassert lv4_t03('bbbb') == 1", "    last = {}\n    start = best = 0\n    for i, c in enumerate(s):\n        if c in last and last[c] >= start:\n            start = last[c] + 1\n        last[c] = i\n        best = max(best, i - start + 1)\n    return best"),
        ("L4_004", "def lv4_t04(nums, target):\n    \"\"\"Two-sum: return 0-based indices i<j with nums[i]+nums[j]==target; assume unique solution.\"\"\"\n", "assert lv4_t04([2,7,11,15], 9) in ([0,1], (0,1))\nassert lv4_t04([3,3], 6) in ([0,1], (0,1))", "    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target-x], i]\n        seen[x] = i\n    return [-1,-1]"),
        ("L4_005", "def lv4_t05(s):\n    \"\"\"Valid parentheses only '(' ')'; return True if balanced.\"\"\"\n", "assert lv4_t05('(())') is True\nassert lv4_t05('(()') is False\nassert lv4_t05('') is True", "    d = 0\n    for c in s:\n        d += 1 if c == '(' else -1\n        if d < 0:\n            return False\n    return d == 0"),
        ("L4_006", "def lv4_t06(nums):\n    \"\"\"Next permutation lex order as list; if last permutation, return sorted ascending.\"\"\"\n", "assert lv4_t06([1,2,3]) == [1,3,2]\nassert lv4_t06([3,2,1]) == [1,2,3]", "    a = list(nums)\n    i = len(a)-2\n    while i >= 0 and a[i] >= a[i+1]:\n        i -= 1\n    if i < 0:\n        return sorted(a)\n    j = len(a)-1\n    while a[j] <= a[i]:\n        j -= 1\n    a[i], a[j] = a[j], a[i]\n    a[i+1:] = reversed(a[i+1:])\n    return a"),
        ("L4_007", "def lv4_t07(n):\n    \"\"\"Integer sqrt floor for n>=0.\"\"\"\n", "assert lv4_t07(8) == 2\nassert lv4_t07(9) == 3\nassert lv4_t07(0) == 0", "    lo, hi = 0, n\n    while lo <= hi:\n        m = (lo+hi)//2\n        if m*m <= n:\n            lo = m + 1\n            ans = m\n        else:\n            hi = m - 1\n    return ans"),
        ("L4_008", "def lv4_t08(nums):\n    \"\"\"Return length of LIS (strictly increasing subsequence).\"\"\"\n", "assert lv4_t08([10,9,2,5,3,7,101,18]) == 4\nassert lv4_t08([]) == 0", "    tails = []\n    for x in nums:\n        lo, hi = 0, len(tails)\n        while lo < hi:\n            m = (lo+hi)//2\n            if tails[m] < x:\n                lo = m+1\n            else:\n                hi = m\n        if lo == len(tails):\n            tails.append(x)\n        else:\n            tails[lo] = x\n    return len(tails)"),
        ("L4_009", "def lv4_t09(s, words):\n    \"\"\"words same length; return start indices where s is concatenation of each word in words exactly once (in some order).\"\"\"\n", "assert sorted(lv4_t09('barfoothefoobarman', ['foo','bar'])) == [0,9]\nassert lv4_t09('a', ['a']) == [0]\nassert lv4_t09('ab', ['a','c']) == []", "    from collections import Counter\n    if not words:\n        return []\n    L, n = len(words[0]), len(words)\n    need = Counter(words)\n    res = []\n    for i in range(len(s) - L * n + 1):\n        seen = Counter()\n        ok = True\n        for j in range(n):\n            w = s[i + j * L : i + (j + 1) * L]\n            if w not in need or seen[w] >= need[w]:\n                ok = False\n                break\n            seen[w] += 1\n        if ok and seen == need:\n            res.append(i)\n    return res"),
        ("L4_010", "def lv4_t10(grid):\n    \"\"\"grid rows are str of 0/1; return number of islands (4-connected '1').\"\"\"\n", "assert lv4_t10(['111','010','111']) == 1\nassert lv4_t10(['10','01']) == 2", "    if not grid:\n        return 0\n    R, C = len(grid), len(grid[0])\n    vis = set()\n    def dfs(r,c):\n        if r<0 or c<0 or r>=R or c>=C or grid[r][c]!='1' or (r,c) in vis:\n            return\n        vis.add((r,c))\n        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):\n            dfs(r+dr,c+dc)\n    cnt = 0\n    for r in range(R):\n        for c in range(C):\n            if grid[r][c]=='1' and (r,c) not in vis:\n                dfs(r,c)\n                cnt += 1\n    return cnt"),
    ]
    for tid, pr, ts, bd in l4:
        add(4, tid, pr, ts, bd)

    # L4_011+ Fibonacci variants (still validated)
    for k in range(11, TASKS_PER_LEVEL + 1):
        name = f"lv4_t{k:02d}"
        pr = f"def {name}(n):\n    \"\"\"Return nth Fibonacci number F(0)=0,F(1)=1.\"\"\"\n"
        bd = "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a"
        # F(0)=0 F(1)=1 F(2)=1 F(3)=2 F(4)=3
        ts = f"assert {name}(0) == 0\nassert {name}(1) == 1\nassert {name}(10) == 55"
        add(4, f"L4_{k:03d}", pr, ts, bd)

    # --- Level 5: typing + longer specs (HumanEval-style failure modes) ---
    l5 = [
        ("L5_001", "from typing import List\n\ndef lv5_t01(nums: List[int], k: int) -> List[int]:\n    \"\"\"Return k largest distinct values sorted ascending; if fewer than k distinct, return all distinct sorted.\"\"\"\n", "assert lv5_t01([3,1,2,2,4], 2) == [3,4]\nassert lv5_t01([5,5,5], 2) == [5]", "    u = sorted(set(nums))\n    return u[-k:] if len(u) >= k else u"),
        ("L5_002", "from typing import List\n\ndef lv5_t02(nums: List[int]) -> float:\n    \"\"\"Return arithmetic mean; empty list -> 0.0\"\"\"\n", "assert lv5_t02([2,4]) == 3.0\nassert lv5_t02([]) == 0.0\nassert lv5_t02([1]) == 1.0", "    return sum(nums)/len(nums) if nums else 0.0"),
        ("L5_003", "from typing import List, Tuple\n\ndef lv5_t03(points: List[Tuple[int,int]]) -> float:\n    \"\"\"Return Euclidean distance from first point to last; len<2 -> 0.0\"\"\"\n", "assert lv5_t03([(0,0),(3,4)]) == 5.0\nassert lv5_t03([]) == 0.0", "    import math\n    if len(points) < 2:\n        return 0.0\n    x1,y1 = points[0]\n    x2,y2 = points[-1]\n    return math.hypot(x2-x1, y2-y1)"),
        ("L5_004", "from typing import List\n\ndef lv5_t04(nums: List[int]) -> List[int]:\n    \"\"\"Move all zeros to end preserving order of non-zeros.\"\"\"\n", "assert lv5_t04([0,1,0,3,12]) == [1,3,12,0,0]\nassert lv5_t04([]) == []", "    return [x for x in nums if x != 0] + [0]*nums.count(0)"),
        ("L5_005", "from typing import List\n\ndef lv5_t05(intervals: List[List[int]]) -> int:\n    \"\"\"Minimum rooms for meeting intervals [start,end]; assume sorted input not required — sort inside.\"\"\"\n", "assert lv5_t05([[0,30],[5,10],[15,20]]) == 2\nassert lv5_t05([[7,10],[2,4]]) == 1", "    import heapq\n    intervals.sort()\n    h = []\n    for s,e in intervals:\n        while h and h[0] <= s:\n            heapq.heappop(h)\n        heapq.heappush(h, e)\n    return len(h)"),
        ("L5_006", "from typing import List\n\ndef lv5_t06(nums: List[int]) -> int:\n    \"\"\"Maximum product of any contiguous subarray.\"\"\"\n", "assert lv5_t06([2,3,-2,4]) == 6\nassert lv5_t06([-2,0,-1]) == 0", "    best = mn = mx = nums[0]\n    for x in nums[1:]:\n        choices = (x, x*mn, x*mx)\n        mn, mx = min(choices), max(choices)\n        best = max(best, mx)\n    return best"),
        ("L5_007", "from typing import List\n\ndef lv5_t07(nums: List[int]) -> bool:\n    \"\"\"Can partition into two subsets with equal sum?\"\"\"\n", "assert lv5_t07([1,5,11,5]) is True\nassert lv5_t07([1,2,3,5]) is False", "    s = sum(nums)\n    if s % 2:\n        return False\n    t = s//2\n    dp = {0}\n    for x in nums:\n        dp |= {v+x for v in list(dp) if v+x <= t}\n    return t in dp"),
        ("L5_008", "from typing import List\n\ndef lv5_t08(matrix: List[List[int]]) -> List[int]:\n    \"\"\"Return spiral order traversal of matrix (may be empty).\"\"\"\n", "assert lv5_t08([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\nassert lv5_t08([]) == []", "    if not matrix:\n        return []\n    r0, r1 = 0, len(matrix)-1\n    c0, c1 = 0, len(matrix[0])-1\n    out = []\n    while r0 <= r1 and c0 <= c1:\n        for c in range(c0, c1+1):\n            out.append(matrix[r0][c])\n        r0 += 1\n        for r in range(r0, r1+1):\n            out.append(matrix[r][c1])\n        c1 -= 1\n        if r0 <= r1:\n            for c in range(c1, c0-1, -1):\n                out.append(matrix[r1][c])\n            r1 -= 1\n        if c0 <= c1:\n            for r in range(r1, r0-1, -1):\n                out.append(matrix[r][c0])\n            c0 += 1\n    return out"),
        ("L5_009", "from typing import List\n\ndef lv5_t09(nums: List[int]) -> int:\n    \"\"\"Minimum jumps to reach last index; each nums[i] is max jump length; assume reachable.\"\"\"\n", "assert lv5_t09([2,3,1,1,4]) == 2\nassert lv5_t09([0]) == 0", "    n = len(nums)\n    if n <= 1:\n        return 0\n    jumps = far = end = 0\n    for i in range(n-1):\n        far = max(far, i + nums[i])\n        if i == end:\n            jumps += 1\n            end = far\n    return jumps"),
        ("L5_010", "from typing import List\n\ndef lv5_t10(citations: List[int]) -> int:\n    \"\"\"H-index: max h with at least h papers having >=h citations.\"\"\"\n", "assert lv5_t10([3,0,6,1,5]) == 3\nassert lv5_t10([1,3,1]) == 1", "    c = sorted(citations, reverse=True)\n    h = 0\n    for i, x in enumerate(c, 1):\n        if x >= i:\n            h = i\n    return h"),
    ]
    for tid, pr, ts, bd in l5:
        add(5, tid, pr, ts, bd)

    # L5_011+: typing + parameterized distinct tasks
    for k in range(11, TASKS_PER_LEVEL + 1):
        name = f"lv5_t{k:02d}"
        shift = k % 5
        pr = (
            "from typing import List\n\n"
            f"def {name}(nums: List[int]) -> List[int]:\n"
            f"    \"\"\"Return nums rotated left by {shift} (wrap); empty unchanged.\"\"\"\n"
        )
        bd = f"    if not nums:\n        return []\n    s = {shift} % len(nums)\n    return nums[s:] + nums[:s]"
        ts = (
            f"assert {name}([]) == []\n"
            f"assert {name}([1,2,3,4]) == {[1,2,3,4][shift%4:] + [1,2,3,4][:shift%4]}\n"
            f"assert {name}([10]) == [10]"
        )
        add(5, f"L5_{k:03d}", pr, ts, bd)

    # Size checks
    for lv in range(1, 6):
        if len(levels[lv]) != TASKS_PER_LEVEL:
            raise RuntimeError(
                f"Level {lv} has {len(levels[lv])} tasks, expected {TASKS_PER_LEVEL}"
            )
    return levels


def main() -> None:
    levels = build_all_tasks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lv in range(1, 6):
        path = OUT_DIR / f"level{lv}.json"
        with open(path, "w") as f:
            json.dump(levels[lv], f, indent=2)
        print(f"Wrote {path} ({len(levels[lv])} tasks)")
    print("All tasks validated with reference solutions.")


if __name__ == "__main__":
    main()
