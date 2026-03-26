#!/usr/bin/env bash
# Phase A: UMAP/t-SNE embeddings for code geometry (CPU; optional GPU via cuML)
set -e
cd "$(dirname "$0")"
CONFIG="${1:-config.yaml}"
echo "Phase A: code geometry embeddings (config=$CONFIG)"
python phase_a_embeddings.py --config "$CONFIG"
python phase_a_analysis.py --config "$CONFIG"
echo "Phase A done."
