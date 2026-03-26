#!/usr/bin/env bash
# Phase C: Concept subspaces (correct/wrong, error_category) — CPU only
set -e
cd "$(dirname "$0")"
CONFIG="${1:-config.yaml}"
echo "Phase C: code geometry subspaces (config=$CONFIG)"
python phase_c_subspaces.py --config "$CONFIG"
echo "Phase C done."
