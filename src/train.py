import os
import sys
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

# Add emg2pose to python path to load networks
sys.path.append(os.path.join(os.path.dirname(__file__), '../emg2pose'))
# Alternatively, assuming module `emg2pose` is successfully set up:
from emg2pose.networks import NeuroPose, EncoderBlock, ResidualBlock, DecoderBlock

from src.dataset import MindroveEMGDataset


def infer_hdf5_input_channels(dataset_path: str) -> int:
    """How many columns are stored in each ``emg`` row (8 = EMG-only, 14 = EMG+IMU)."""
    with h5py.File(dataset_path, "r") as f:
        if "input_channels" in f.attrs:
            return int(f.attrs["input_channels"])
        tg = f["train"]
        first = next(iter(tg.keys()))
        return int(f["train"][first]["emg"].shape[1])


def infer_hdf5_target(dataset_path: str) -> tuple[str, int]:
    """Return (target_type, target_dim) for the HDF5 file.

    Falls back to inspecting the first group's datasets for pre-`target_type`
    files (which only had `landmarks`).
    """
    with h5py.File(dataset_path, "r") as f:
        if "target_type" in f.attrs:
            ttype = str(f.attrs["target_type"])
            tdim = int(f.attrs.get("target_dim",
                                    20 if ttype == "angles" else 60))
            return ttype, tdim
        first = next(iter(f["train"].keys()))
        grp = f["train"][first]
        if "angles" in grp:
            return "angles", int(grp["angles"].shape[1])
        return "landmarks", int(grp["landmarks"].shape[1])


def get_model(config_path):
    """
    Manually instantiate the NeuroPose model based on the dict-like config
    from Hydra since we aren't using Hydra decorators here.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    network_cfg = cfg['network']
    dropout_rate = float(cfg['dropout_rate'])
    
    encoders = nn.ModuleList([
         EncoderBlock(
             in_channels=b['in_channels'],
             out_channels=b['out_channels'],
             kernel_size=tuple(b['kernel_size']),
             max_pool_size=tuple(b['max_pool_size']),
             dropout_rate=dropout_rate
         ) for b in network_cfg['encoder_blocks']
    ])
    
    residuals = nn.ModuleList([
         ResidualBlock(
             in_channels=b['in_channels'],
             out_channels=b['out_channels'],
             kernel_size=tuple(b['kernel_size']),
             num_convs=b['num_convs'],
             dropout_rate=dropout_rate
         ) for b in network_cfg['residual_blocks']
    ])
    
    decoders = nn.ModuleList([
         DecoderBlock(
             in_channels=b['in_channels'],
             out_channels=b['out_channels'],
             kernel_size=tuple(b['kernel_size']),
             upsampling=tuple(b['upsampling']),
             dropout_rate=dropout_rate
         ) for b in network_cfg['decoder_blocks']
    ])

    model = NeuroPose(
        encoder_blocks=encoders,
        residual_blocks=residuals,
        decoder_blocks=decoders,
        linear_in_channels=network_cfg['linear_in_channels'],
        out_channels=network_cfg['out_channels']
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/dataset.hdf5', help='Path to dataset.hdf5')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience (epochs with no val improvement)')
    parser.add_argument('--window', type=int, default=1000, help='Window length (time)')
    parser.add_argument('--step', type=int, default=250, help='Stride for train window slicing')
    parser.add_argument('--emg-only', action='store_true',
                        help='Drop IMU from input (8ch EMG only). Requires --config pointing at an 8ch network yaml.')
    parser.add_argument('--config', type=str, default=None,
                        help='Network config yaml. Default: emg2pose/config/network/neuropose_mindrove.yaml '
                             '(or neuropose_emg_only.yaml when --emg-only is set).')
    parser.add_argument('--whitened-loss', action='store_true',
                        help='Weight MSE per landmark coord by 1/variance (computed on train split). '
                             'Counters the rank-collapse where low-variance coords get ignored.')
    parser.add_argument('--save-name', type=str, default='neuropose_mindrove_best.pt',
                        help='Checkpoint filename in models/.')
    parser.add_argument('--angle-indices', type=str, default=None,
                        help='Comma-sep subset of 0..19 to keep from the 20-angle target. '
                             'e.g. "0,5,9,13,17" for 5 palm-to-finger flexion angles. '
                             'Model out_channels in the yaml must match len(indices).')
    args = parser.parse_args()

    keep_indices = None
    if args.angle_indices:
        keep_indices = [int(x) for x in args.angle_indices.split(',') if x.strip()]
        print(f"Subsetting angle targets to indices: {keep_indices}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    stored_ch = infer_hdf5_input_channels(args.dataset)
    if stored_ch == 8:
        if not args.emg_only:
            print(
                "Dataset stores 8 channels (filtered EMG, no IMU). "
                "Enabling --emg-only and 8-channel network config.",
            )
        args.emg_only = True
    elif stored_ch != 14:
        print(f"Warning: HDF5 input has {stored_ch} channels (expected 8 or 14).")

    target_type, target_dim = infer_hdf5_target(args.dataset)
    if keep_indices is not None:
        target_dim = len(keep_indices)
    print(f"Target: {target_type} ({target_dim}-D)")
    if target_type == "angles" and args.whitened_loss:
        print("WARNING: --whitened-loss is a landmark-era flag; disabling for angles.")
        args.whitened_loss = False

    # Datasets and Loaders
    print("Loading datasets...")
    try:
        train_ds = MindroveEMGDataset(args.dataset, split='train', window_length=args.window, step_size=args.step,
                                       emg_only=args.emg_only, keep_indices=keep_indices)
        val_ds = MindroveEMGDataset(args.dataset, split='val', window_length=args.window, step_size=args.window//2,
                                     emg_only=args.emg_only, keep_indices=keep_indices)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")

    # Model
    # Resolve relative to project root, not cwd
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.config:
        model_config_path = args.config if os.path.isabs(args.config) else os.path.join(_project_root, args.config)
    elif target_type == "angles":
        # Angles target implies EMG-only 8ch input + 20 output channels.
        model_config_path = os.path.join(_project_root, "emg2pose/config/network/neuropose_angles.yaml")
    elif args.emg_only:
        model_config_path = os.path.join(_project_root, "emg2pose/config/network/neuropose_emg_only.yaml")
    else:
        model_config_path = os.path.join(_project_root, "emg2pose/config/network/neuropose_mindrove.yaml")
    print(f"Network config: {model_config_path}")
    model = get_model(model_config_path).to(device)

    # Hard-check model output dim matches dataset target dim.
    model_out = int(model.linear.out_features)
    if model_out != target_dim:
        raise RuntimeError(
            f"Model output dim ({model_out}) does not match dataset target dim "
            f"({target_dim}). For target='{target_type}' use a config with "
            f"out_channels={target_dim} (e.g. neuropose_angles.yaml for angles, "
            f"neuropose_mindrove.yaml for landmarks)."
        )

    # Loss: plain MSE, or MSE whitened by per-coord variance (counters rank-collapse).
    coord_weights = None
    if args.whitened_loss:
        # Only hit for landmark targets (disabled upstream for angles).
        print("Computing per-coord landmark variance on train split for whitened loss...")
        with h5py.File(args.dataset, 'r') as h5f:
            target_key = 'angles' if target_type == 'angles' else 'landmarks'
            chunks = []
            for sname in h5f['train'].keys():
                chunks.append(h5f['train'][sname][target_key][:])
            all_lm = np.concatenate(chunks, axis=0)  # (T_total, target_dim)
        var = all_lm.var(axis=0)
        var = np.maximum(var, 1e-4)
        w = 1.0 / var
        w = w / w.mean()
        coord_weights = torch.from_numpy(w.astype(np.float32)).to(device)
        print(f"  coord weights: min={w.min():.3f} median={np.median(w):.3f} max={w.max():.3f}")

    def compute_loss(preds, target):
        # preds, target: (B, 60, T)
        if coord_weights is None:
            return nn.functional.mse_loss(preds, target)
        err2 = (preds - target) ** 2          # (B, 60, T)
        err2 = err2 * coord_weights[None, :, None]
        return err2.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW lr={args.lr} weight_decay={args.weight_decay} | dropout from config | "
          f"early-stop patience={args.patience} | emg_only={args.emg_only} | whitened_loss={args.whitened_loss}")

    best_val_loss = float('inf')
    epochs_since_best = 0
    models_dir = os.path.join(_project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for emg_x, target_y in loop:
            emg_x = emg_x.to(device)
            target_y = target_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Expected shape in: (B, C, T) -> emg_x is (B, 8, 1000)
            preds = model(emg_x)  # Out shape: (B, 60, 1000)

            loss = compute_loss(preds, target_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emg_x, target_y in val_loader:
                emg_x = emg_x.to(device)
                target_y = target_y.to(device)
                
                preds = model(emg_x)
                loss = compute_loss(preds, target_y)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"-> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
            ckpt_path = os.path.join(models_dir, args.save_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"-> Best model saved to {ckpt_path}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"-> Early stopping: no val improvement for {args.patience} epochs. Best val loss: {best_val_loss:.6f}")
                break

if __name__ == '__main__':
    main()