#!/usr/bin/env bash
# 2-GPU fan-out: one dataset per GPU, both running concurrently.
#
# Usage: bash benchmarks/run_all.sh [preset]   (default: moderate)
#
# Each script reads CUDA_VISIBLE_DEVICES so JAX inside the child process
# sees only one GPU. Per-basis timing stays single-GPU and Julia-comparable.
set -euo pipefail

PRESET=${1:-moderate}
TS=$(date +%Y%m%d-%H%M%S)
PYTHON=${PYTHON:-$(command -v python || command -v python3)}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESULTS_BASE="$ROOT/benchmarks/results"
mkdir -p "$RESULTS_BASE"

QD_OUT="$RESULTS_BASE/quickdraw_${PRESET}_${TS}"
DV_OUT="$RESULTS_BASE/div2k_8q_${PRESET}_${TS}"

echo "== launching quickdraw on GPU 0 → $QD_OUT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/benchmarks/run_quickdraw.py" "$PRESET" --out "$QD_OUT" &
PID_QD=$!

echo "== launching div2k_8q  on GPU 1 → $DV_OUT"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$ROOT/benchmarks/run_div2k_8q.py"  "$PRESET" --out "$DV_OUT" &
PID_DV=$!

RC_QD=0; RC_DV=0
wait "$PID_QD" || RC_QD=$?
wait "$PID_DV" || RC_DV=$?
echo "quickdraw exit=$RC_QD; div2k_8q exit=$RC_DV"
exit $(( RC_QD + RC_DV ))
