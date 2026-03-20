#!/usr/bin/env bash
set -e

python src/run_noise_edit_full.py \
  --config configs/noise_edit_full.yaml \
  --input_dir data/faces \
  --output_dir results/noise_edit_full

echo "Done. Check results/noise_edit_full/"
