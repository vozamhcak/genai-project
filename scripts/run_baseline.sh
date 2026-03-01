#!/usr/bin/env bash
set -e

python src/run_batch_baseline.py
python src/make_grid.py
python src/eval_clip.py
python src/run_recon_sanity.py
python src/make_grid_recon.py

echo "Done. Check results/grids and results/metrics_clip.json"