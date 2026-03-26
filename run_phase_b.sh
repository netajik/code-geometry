#!/usr/bin/env bash
# Phase B: Label-level confounds (correctness, error_category, prompt/tests) — CPU only
set -e
cd "$(dirname "$0")"
CONFIG="${1:-config.yaml}"
echo "Phase B: code geometry label confounds (config=$CONFIG)"
python phase_b_deconfounding.py --config "$CONFIG"
echo "Phase B done."
