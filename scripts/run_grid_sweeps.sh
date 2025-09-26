#!/usr/bin/env bash
set -euo pipefail

# Runs a sequence of grid sweeps.
#
# Usage:
#   bash scripts/run_grid_sweeps.sh <grid1.yaml> [<grid2.yaml> ...]
#
# Behavior:
#   - For each provided grid YAML path, runs the parallel sweeper sequentially.
#   - Output directory derived from the grid filename stem.
#     For configs/grid_sweep_1.yaml -> OUT_DIR=results/grid_sweep_1
#   - Labels inside each sweep variant are formed as <base_label>-i<index> where base_label is:
#       (1) the variant's existing label if provided in base YAML after overrides, else
#       (2) the base YAML's label, else
#       (3) the grid filename stem.
#   - Base config is fixed at configs/base_sweep.yaml
#
# Outputs per grid:
#   results/<stem>/{parquet, parallel_results*.jsonl}

BASE_YAML="configs/base_sweep.yaml"

# Ensure base config exists
if [[ ! -f "$BASE_YAML" ]]; then
  echo "[ERROR] Base YAML not found: $BASE_YAML" >&2
  exit 1
fi

# Require at least one grid YAML path
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <grid1.yaml> [<grid2.yaml> ...]" >&2
  exit 2
fi

for GRID_YAML in "$@"; do
  if [[ ! -f "$GRID_YAML" ]]; then
    echo "[ERROR] Grid YAML not found: $GRID_YAML" >&2
    exit 1
  fi
  BASENAME="$(basename -- "$GRID_YAML")"
  STEM="${BASENAME%.*}"
  OUT_DIR="results/${STEM}"

  echo "=== Running grid sweep ==="
  echo "YAML: ${GRID_YAML}"
  echo "Out:  ${OUT_DIR}"

  mkdir -p "${OUT_DIR}" || true

  make docker-run ARGS="python -u -m pipeline.parallel_sweep \
    --base-yaml ${BASE_YAML} \
    --grid-yaml ${GRID_YAML} \
    --output-dir ${OUT_DIR}"

  echo "=== Completed: ${STEM} ==="
  echo
done

echo "All requested sweeps finished."