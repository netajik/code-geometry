#!/usr/bin/env bash
# Fourier screening: layer-axis spectrum of activation norms — CPU only
set -e
cd "$(dirname "$0")"
CONFIG="${1:-config.yaml}"
echo "Fourier screening (config=$CONFIG)"
python fourier_screening.py --config "$CONFIG"
echo "Fourier done."
