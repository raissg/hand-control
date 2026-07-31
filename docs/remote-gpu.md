# Remote GPU: CMU Qatar (`gpujobs`) for this repo

Account **`graissov`**, login **`gpujobs.qatar.cmu.edu`**, project root on the cluster **`/data1/graissov/hand-control`**. This repository on the Mac is **`/Users/gr/Desktop/hand control`**.

---

## 1. Prerequisites

1. **Network:** CMU Qatar campus network or VPN so `gpujobs.qatar.cmu.edu` is reachable.
2. **SSH:** Key-based login as **`graissov`** using private key **`/Users/gr/.ssh/id_ed25519`** on the Mac; the matching public key is in **`~/.ssh/authorized_keys`** on **`gpujobs`** for user **`graissov`**.
3. **Slurm:** GPU work uses partition **`gpu2`**; compute runs on cluster GPU nodes (the same workflow references **`deepnet2`** and H200 MIG slices in `running_remote_gpu.md`).

`gpujobs` is the **login / staging** host. **`torch.cuda.is_available()` is false there** in normal SSH sessions; use **`srun` / `sbatch`** on **`gpu2`** for CUDA.

> **Gotchas verified by smoke test:**
> - **Every `srun` / `sbatch` must include `--mcs-label=users`.** Omitting it fails with `error: Please include --mcs-label in your job`.
> - **Login ≠ compute Python version.** Login node runs Python 3.10; compute node `deepnet2` runs 3.12. A venv built on the login node silently falls back to system python on compute and loses its site-packages — so the venv must be created inside an `srun` allocation on the compute node (the sync script does this).
> - **`src/` is not an importable package.** Run training with `PYTHONPATH=.` (or `python -m src.train`) from the project root.

---

## 2. Paths

| What | Path |
|------|------|
| SSH target | `graissov@gpujobs.qatar.cmu.edu` |
| Project directory on the cluster | `/data1/graissov/hand-control` |
| Virtualenv | `/data1/graissov/hand-control/.venv` |
| Training code | `/data1/graissov/hand-control/src/train.py` |
| NeuroPose YAML used by training | `/data1/graissov/hand-control/emg2pose/config/network/neuropose_mindrove.yaml` |
| Default dataset path in `train.py` | `/data1/graissov/hand-control/data/dataset.hdf5` |

---

## 3. Sync from the Mac (repo root)

Working directory on the Mac:

```bash
cd "/Users/gr/Desktop/hand control"
```

**Full sync** (rsync `src/` + `emg2pose/` + `requirements-src-remote.txt`, then on the server: create `.venv` if needed, install PyTorch from `https://download.pytorch.org/whl/cu124`, `pip install -r requirements-src-remote.txt`, `pip install -e emg2pose`, import check):

```bash
./scripts/sync_handcontrol_gpu.sh "graissov@gpujobs.qatar.cmu.edu:/data1/graissov/hand-control"
```

**Code only** (no pip; use after the venv already exists):

```bash
SKIP_PIP=1 ./scripts/sync_handcontrol_gpu.sh "graissov@gpujobs.qatar.cmu.edu:/data1/graissov/hand-control"
```

**PyTorch cu126 wheels** instead of cu124:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
  ./scripts/sync_handcontrol_gpu.sh "graissov@gpujobs.qatar.cmu.edu:/data1/graissov/hand-control"
```

---

## 4. Login shell on `gpujobs`

```bash
ssh graissov@gpujobs.qatar.cmu.edu
cd /data1/graissov/hand-control
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 5. Copy `dataset.hdf5` to the cluster

On the Mac:

```bash
mkdir -p "/Users/gr/Desktop/hand control/data"
```

Put **`dataset.hdf5`** at **`/Users/gr/Desktop/hand control/data/dataset.hdf5`**, then:

```bash
ssh graissov@gpujobs.qatar.cmu.edu "mkdir -p /data1/graissov/hand-control/data"
rsync -avP "/Users/gr/Desktop/hand control/data/dataset.hdf5" \
  "graissov@gpujobs.qatar.cmu.edu:/data1/graissov/hand-control/data/dataset.hdf5"
```

---

## 6. Training

**Working directory must be** `/data1/graissov/hand-control` (not `…/src`), so `emg2pose/config/...` resolves.

Do **not** try to run training on the login node — the venv's Python won't
resolve there (see gotcha above). Always go through `srun` / `sbatch`.

### Interactive GPU (Slurm)

```bash
ssh graissov@gpujobs.qatar.cmu.edu
cd /data1/graissov/hand-control
srun --mcs-label=users -p gpu2 --gres=gpu:1 -t 0:30:00 --pty bash
```

Inside that allocation:

```bash
source /data1/graissov/hand-control/.venv/bin/activate
cd /data1/graissov/hand-control
python -c "import torch; print('cuda', torch.cuda.is_available())"
PYTHONPATH=. python src/train.py --dataset data/dataset.hdf5 --epochs 50 --batch_size 32
```

### Batch file on the cluster

Create `/data1/graissov/hand-control/job-train.sh`:

```bash
#!/bin/bash
#SBATCH -J emg-train
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH --mcs-label=users
#SBATCH -t 2:00:00
#SBATCH -o /data1/graissov/hand-control/slurm-%j.out

set -euo pipefail
cd /data1/graissov/hand-control
source .venv/bin/activate
PYTHONPATH=. python src/train.py --dataset data/dataset.hdf5 --epochs 50 --batch_size 32
```

Submit from `gpujobs`:

```bash
ssh graissov@gpujobs.qatar.cmu.edu
cd /data1/graissov/hand-control
sbatch job-train.sh
```

---

## 7. Cursor / VS Code SFTP

Project file **`.vscode/sftp.json`** is set for:

- **Profile:** `gpujobs` (`defaultProfile` is `gpujobs`)
- **Host:** `gpujobs.qatar.cmu.edu`
- **User:** `graissov`
- **Remote path:** `/data1/graissov/hand-control`
- **Key:** `/Users/gr/.ssh/id_ed25519`

Open the folder **`/Users/gr/Desktop/hand control`** as the workspace root, then **SFTP: Set Profile** → **`gpujobs`**.

---

## 8. Troubleshooting

| Symptom | Check |
|--------|--------|
| `ssh: Could not resolve host gpujobs.qatar.cmu.edu` | VPN or campus network. |
| Rsync cannot create `/data1/graissov/hand-control` | Permissions or quota under `/data1/graissov` for `graissov`. |
| `ModuleNotFoundError: emg2pose` | On the cluster: `cd /data1/graissov/hand-control && source .venv/bin/activate && pip install -e emg2pose` or rerun sync **without** `SKIP_PIP=1`. |
| YAML `FileNotFoundError` | CWD is not `/data1/graissov/hand-control`. |
| `cuda` always false | Session is on the login node without a Slurm GPU allocation; use `srun -p gpu2 ...` as above. |
| SFTP log shows `[localhost]` | Workspace root must be `/Users/gr/Desktop/hand control`; profile must be **`gpujobs`**. |

---

## 9. Other repo pointers

- `scripts/sync_handcontrol_gpu.sh` — rsync to `/data1/graissov/hand-control` and, unless `SKIP_PIP=1`, remote venv + `pip install`.
- `requirements-src-remote.txt` — Linux pip deps for `src/`.
- `running_remote_gpu.md` — older short notes (WiLoR, `WILOR_DEVICE`, `WILOR_VIDEO` on `/data1/...`).
