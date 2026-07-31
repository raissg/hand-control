#!/usr/bin/env bash
# Sync `src/` plus `emg2pose/` (required by src/train.py) to a remote host and install deps.
#
# Usage (from repo root, on a machine that can reach the server, e.g. on VPN):
#   ./scripts/sync_handcontrol_gpu.sh user@gpu.example.edu:/data1/user/hand-control
#
# Optional env:
#   TORCH_INDEX_URL   — PyTorch wheel index (default: cu124)
#   SKIP_PIP          — if set to 1, only rsync, no remote pip
#   SRUN_ARGS         — srun args used to build the venv on a compute node
#                       (default: "--mcs-label=users -p gpu2 --gres=gpu:1 -t 0:25:00").
#                       The venv MUST be created on the node it will run on —
#                       login and compute nodes here have different Python
#                       versions, and a venv built on one won't work on the other.
#   BUILD_ON_LOGIN=1  — skip srun and build the venv on the login node instead
#                       (only safe if login and compute share a Python version).
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 user@host:/absolute/path/to/hand-control" >&2
  exit 1
fi

REMOTE="$1"
HOST="${REMOTE%%:*}"
DIR="${REMOTE#"${HOST}:"}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
SRUN_ARGS="${SRUN_ARGS:---mcs-label=users -p gpu2 --gres=gpu:1 -t 0:25:00}"
BUILD_ON_LOGIN="${BUILD_ON_LOGIN:-0}"

echo "==> rsync from: $ROOT"
echo "    destination: $HOST:$DIR"

ssh -o BatchMode=yes "$HOST" "mkdir -p \"$DIR\""

rsync -avz \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  "$ROOT/src/" "$REMOTE/src/"

rsync -avz \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'logs/' \
  --exclude 'notebooks/' \
  --exclude 'apple_jobs_html/' \
  "$ROOT/emg2pose/" "$REMOTE/emg2pose/"

rsync -avz "$ROOT/requirements-src-remote.txt" "$REMOTE/requirements-src-remote.txt"

if [[ "${SKIP_PIP:-0}" == "1" ]]; then
  echo "SKIP_PIP=1 — skipping remote pip install."
  exit 0
fi

echo "==> Remote: install torch + requirements + editable emg2pose"

# Build the venv inside an srun allocation so it uses the compute node's
# Python (login and compute nodes may have different python3 versions —
# a venv from the login node silently falls back to system python on compute
# and loses its site-packages). Override with BUILD_ON_LOGIN=1 if login+compute
# are known to match.
if [[ "$BUILD_ON_LOGIN" == "1" ]]; then
  WRAP=""
else
  WRAP="srun $SRUN_ARGS"
fi

ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
cd "$DIR"
$WRAP bash -lc '
  set -euo pipefail
  cd "$DIR"
  if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip wheel
  pip install torch --index-url "$TORCH_INDEX_URL"
  pip install -r requirements-src-remote.txt
  pip install -e emg2pose
  python3 - <<PY
import torch
from emg2pose.networks import NeuroPose
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("NeuroPose import OK")
PY
'
echo "Remote check finished. Run training from the project root, e.g.:"
echo "  srun --mcs-label=users -p gpu2 --gres=gpu:1 -t 1:00:00 --pty bash"
echo "  source .venv/bin/activate && PYTHONPATH=. python src/train.py --dataset data/dataset.hdf5 --epochs 50"
EOF
