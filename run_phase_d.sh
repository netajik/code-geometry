#!/usr/bin/env bash
# Phase D: LDA discriminants (correct vs wrong) — CPU only
set -e
cd "$(dirname "$0")"
CONFIG="${1:-config.yaml}"
echo "Phase D: LDA / supervised directions (config=$CONFIG)"
python phase_d_lda.py --config "$CONFIG"
echo "Phase D done."
