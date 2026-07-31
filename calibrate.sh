#!/usr/bin/env bash
# Per-session calibration: record ~50s, then head-only fine-tune from the
# best base checkpoint to adapt to the current band placement.
#
# Usage: ./calibrate.sh [base_ckpt]
#   base_ckpt defaults to models/emg5bit_selflabel_best.pt

set -euo pipefail

cd "$(dirname "$0")"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <base_ckpt>" >&2
    echo "    e.g. $0 models/model_20260420_191234.pt" >&2
    exit 1
fi
BASE_CKPT="$1"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_CKPT="emg5bit_session_${STAMP}.pt"

if [ ! -f "$BASE_CKPT" ]; then
    echo "Base checkpoint not found: $BASE_CKPT" >&2
    exit 1
fi
echo "Base checkpoint: $BASE_CKPT"

echo "=== 1/2  Recording calibration session ==="
python -m src.record_bits --calibrate

# Pick the calibration dir we just wrote (newest calib_* under recordings/).
CALIB_DIR="$(ls -1dt recordings/calib_*/ 2>/dev/null | head -n 1)"
if [ -z "$CALIB_DIR" ]; then
    echo "No calib_* directory found under recordings/. Aborting." >&2
    exit 1
fi
CALIB_CSV="${CALIB_DIR%/}/emg.csv"
echo "Recorded: $CALIB_CSV"

echo "=== 2/2  Head-only fine-tune from $BASE_CKPT ==="
python -m src.train_bits \
    --resume "$BASE_CKPT" \
    --sessions "$CALIB_CSV" \
    --freeze-backbone \
    --epochs 10 \
    --save-name "$SESSION_CKPT"

echo
echo "Done. Run live with:"
echo "    python -m src.live_binary --model models/${SESSION_CKPT}"
