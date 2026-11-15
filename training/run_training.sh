#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SRC_ROOT="${SRC_ROOT:-$SCRIPT_DIR/source}"
: "${DATA_ROOT:?set DATA_ROOT to the audited endpoint dataset package}"
: "${OUT_ROOT:?set OUT_ROOT to a new output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing to reuse existing run directory: $OUT_ROOT" >&2
  exit 73
fi
mkdir -p "$OUT_ROOT"
export PYTHONPATH="$SRC_ROOT/map_flow_matching_p1:$SRC_ROOT/map_flow_matching:${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -u "$SRC_ROOT/map_flow_matching_p1/scripts/train_p37_ordered_field.py" --source-project "$SRC_ROOT/map_flow_matching" --data-path "$DATA_ROOT/train_data.npz" --route-labels "$DATA_ROOT/p37_routes.npz" --split-path "$DATA_ROOT/split.npz" --trusted-safety-audit "$DATA_ROOT/safety_audit.npz" --save-dir "$OUT_ROOT/prior" --epochs 100 --batch-size 128 --lr 2e-4 --hidden-dim 64 --latent-dim 8 --precompute-inputs --seed 20261003 --device cuda

"$PYTHON_BIN" -u "$SRC_ROOT/map_flow_matching_p1/scripts/cache_refinement_rollouts.py" --checkpoint "$OUT_ROOT/prior/best_p37.pth" --data-path "$DATA_ROOT/train_data.npz" --save-dir "$OUT_ROOT/cache" --train-maps 128 --validation-maps 0 --tasks-per-map 2 --candidates 4 --batch-size 16 --flow-steps 100 --latent-scale 1.0 --noise-scale 1.0 --seed 20261004 --partition-seed 20260930 --workers 6

"$PYTHON_BIN" -u "$SRC_ROOT/map_flow_matching_p1/scripts/build_refinement_teacher.py" --cache-dir "$OUT_ROOT/cache" --save-dir "$OUT_ROOT/teacher" --steps 200 --batch-size 64 --workers 6

"$PYTHON_BIN" -u "$SRC_ROOT/map_flow_matching_p1/scripts/train_dense_path_refiner.py" --cache-dir "$OUT_ROOT/cache" --teacher-path "$OUT_ROOT/teacher/teacher.npz" --save-dir "$OUT_ROOT/refiner_warm" --warm-updates 1000 --rl-updates 0 --batch-size 32 --draws 4 --lr 3e-4 --auxiliary-weight 0.05 --validation-interval 100 --workers 6 --seed 20261004 --hidden 64 --max-offset-px 3.0 --feature-ablation none
