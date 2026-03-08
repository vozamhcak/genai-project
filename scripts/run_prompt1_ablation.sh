#!/usr/bin/env bash
set -e

python src/run_prompt1_ablation.py \
  --config configs/baseline.yaml \
  --input_dir data/faces \
  --output_dir results/prompt1_ablation \
  --strengths 0.3,0.5,0.6,0.7 \
  --guidance_scales 5.0,7.5,10.0

echo "Done. Check results/prompt1_ablation/summary_prompt1.json"
