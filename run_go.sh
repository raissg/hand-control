#!/usr/bin/env bash
# Run go.py with the most recent calibrated session checkpoint and a tuned
# smoothing pipeline. Pass any extra args after `--` (or just append).
#
# Usage:
#   ./run_go.sh                       # newest emg5bit_session_*.pt
#   ./run_go.sh path/to/other.pt      # explicit checkpoint
#   ./run_go.sh -- --no-flash         # forward extra flags to go.py

set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${1:-}" ] && [ "${1}" != "--" ]; then
    MODEL="$1"; shift
else
    MODEL="$(ls -1t models/emg5bit_session_*.pt 2>/dev/null | head -n 1)"
    if [ -z "$MODEL" ]; then
        echo "No models/emg5bit_session_*.pt found. Run ./calibrate.sh first," >&2
        echo "or pass an explicit checkpoint: $0 path/to/ckpt.pt" >&2
        exit 1
    fi
fi
[ "${1:-}" = "--" ] && shift

if [ ! -f "$MODEL" ]; then
    echo "Checkpoint not found: $MODEL" >&2
    exit 1
fi
echo "Model: $MODEL"

python go.py \
    --model "$MODEL" \
    --no-flash \
    --lock-s 0.3 \
    --median-k 5 \
    --ema-alpha 0.35 \
    --min-consecutive 3 \
    --min-dwell-s 0.15 \
    --on-thresh 0.6 \
    --off-thresh 0.4 \
    "$@"
