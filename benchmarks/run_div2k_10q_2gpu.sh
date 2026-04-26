#!/usr/bin/env bash
# 2-GPU fan-out for DIV2K-10q. Splits bases across two GPUs to halve
# wall-clock vs sequential single-GPU runs.
#
# Per-basis steady-state at m=n=10, bs=4 on RTX 3090 (from profile_training):
#   qft           ~798 ms/step   ≈ 1.66 h for `generalized` (7500 steps)
#   entangled_qft ~806 ms/step   ≈ 1.68 h
#   tebd          ~708 ms/step   ≈ 1.48 h
#   mera           skipped (m+n=20 is not a power of 2)
#
# Best balance: gpu 0 handles entangled_qft alone, gpu 1 handles qft+tebd.
# Total wall ≈ max(1.68, 1.66+1.48) = 3.14 h on `generalized`.
#
# Usage: bash benchmarks/run_div2k_10q_2gpu.sh [preset]   (default: moderate)
set -euo pipefail

PRESET=${1:-moderate}
TS=$(date +%Y%m%d-%H%M%S)
PYTHON=${PYTHON:-$(command -v python || command -v python3)}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESULTS_BASE="$ROOT/benchmarks/results"
mkdir -p "$RESULTS_BASE"

OUT0="$RESULTS_BASE/div2k_10q_${PRESET}_${TS}_gpu0"
OUT1="$RESULTS_BASE/div2k_10q_${PRESET}_${TS}_gpu1"

echo "== launching div2k_10q on GPU 0 (entangled_qft) → $OUT0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/benchmarks/run_div2k_10q.py" "$PRESET" \
    --bases entangled_qft --out "$OUT0" --log-file &
PID0=$!

echo "== launching div2k_10q on GPU 1 (qft, tebd)        → $OUT1"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$ROOT/benchmarks/run_div2k_10q.py" "$PRESET" \
    --bases qft,tebd --out "$OUT1" --log-file &
PID1=$!

RC0=0; RC1=0
wait "$PID0" || RC0=$?
wait "$PID1" || RC1=$?
echo "gpu0 exit=$RC0; gpu1 exit=$RC1"
exit $(( RC0 + RC1 ))
