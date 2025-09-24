#!/usr/bin/env bash
set -euo pipefail

# Runs grid sweeps for buckets: top, middle, bottom
# Usage:
#   bash scripts/run_grid_sweeps.sh              # run all three buckets
#   bash scripts/run_grid_sweeps.sh top          # run only top
#   bash scripts/run_grid_sweeps.sh middle       # run only middle
#   bash scripts/run_grid_sweeps.sh bottom       # run only bottom
#   bash scripts/run_grid_sweeps.sh top bottom   # run top and bottom
#
# Notes:
# - Expects YAML files at configs/grid_sweep_<bucket>.yaml
# - Writes to results/grid_sweep_<bucket>/{parquet,parallel_results.jsonl}
# - Uses base YAML at configs/base_sweep.yaml

BASE_YAML="configs/base_sweep.yaml"
BUCKETS=("top" "middle" "bottom")

# If args provided, override default buckets
if [[ $# -gt 0 ]]; then
  BUCKETS=("$@")
fi

for bucket in "${BUCKETS[@]}"; do
  GRID_YAML="configs/grid_sweep_${bucket}.yaml"
  OUT_DIR="results/grid_sweep_${bucket}"
  PARQUET_DIR="${OUT_DIR}/parquet"
  RESULTS_JSONL="${OUT_DIR}/parallel_results.jsonl"
  LABEL_PREFIX="grid_sweep_${bucket}"

  echo "=== Running grid sweep for bucket: ${bucket} ==="
  echo "YAML: ${GRID_YAML}"
  echo "Out:  ${OUT_DIR}"

  # Ensure output directory exists (parquet directory will be created by the job if needed)
  mkdir -p "${OUT_DIR}" || true

  make docker-run ARGS="python -u pipeline/parallel_sweep.py \
    --base-yaml ${BASE_YAML} \
    --grid-yaml ${GRID_YAML} \
    --parquet-dir ${PARQUET_DIR} \
    --results-jsonl ${RESULTS_JSONL} \
    --label-prefix ${LABEL_PREFIX}"

  echo "=== Completed: ${bucket} ==="
  echo

done

echo "All requested sweeps finished."