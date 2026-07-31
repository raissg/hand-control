import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml

class MindroveEMGDataset(Dataset):
    def __init__(self, hdf5_path, split='train', window_length=1000, step_size=250,
                 emg_only=False, num_emg=8, keep_indices=None):
        """
        Dataset for creating sliding windows from continuous session data.
        :param hdf5_path: Path to dataset.hdf5
        :param split: 'train', 'val', or 'test'
        :param window_length: Number of time samples per window
        :param step_size: Sliding window stride
        :param emg_only: If True, drop IMU channels — model sees only the first
                         `num_emg` channels (EMG). Use True for HDF5 built with
                         ``convert_to_hdf5 --no-imu`` (8-channel file) or to ablate
                         IMU on a 14-channel file.
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.window_length = window_length
        self.step_size = step_size
        self.emg_only = emg_only
        self.num_emg = num_emg
        self.keep_indices = (None if keep_indices is None
                             else np.asarray(keep_indices, dtype=np.int64))
        
        self.samples = []

        # Pre-calculate all valid window start indices for fast __getitem__
        with h5py.File(self.hdf5_path, 'r') as h5f:
            if split not in h5f:
                raise ValueError(f"Split '{split}' not found in HDF5 file.")

            # Per-channel normalization stats (computed from train split at build time)
            if 'input_mean' in h5f.attrs and 'input_std' in h5f.attrs:
                self.input_mean = np.asarray(h5f.attrs['input_mean'], dtype=np.float32)
                self.input_std = np.asarray(h5f.attrs['input_std'], dtype=np.float32)
            else:
                self.input_mean = None
                self.input_std = None

            # Target type: 'landmarks' (60-D, legacy) or 'angles' (20-D). If the
            # attr is missing, fall back to the dataset key present in the first
            # group for backward compat with older HDF5 files.
            if 'target_type' in h5f.attrs:
                self.target_type = str(h5f.attrs['target_type'])
            else:
                first_grp = h5f[split][next(iter(h5f[split].keys()))]
                self.target_type = 'angles' if 'angles' in first_grp else 'landmarks'
            self.target_key = 'angles' if self.target_type == 'angles' else 'landmarks'

            split_group = h5f[split]
            for session_name in split_group.keys():
                session_grp = split_group[session_name]
                total_length = session_grp['emg'].shape[0]
                
                # Sliding window
                start_indices = range(0, total_length - self.window_length + 1, self.step_size)
                for start_idx in start_indices:
                    self.samples.append({
                        'session': session_name,
                        'start_idx': start_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Open HDF5 file in worker thread when __getitem__ is called
        # to avoid multiprocessing issues with HDF5
        sample_info = self.samples[idx]
        session_name = sample_info['session']
        start_idx = sample_info['start_idx']
        end_idx = start_idx + self.window_length
        
        with h5py.File(self.hdf5_path, 'r') as h5f:
            grp = h5f[self.split][session_name]
            emg_window = grp['emg'][start_idx:end_idx].astype(np.float32)
            target_window = grp[self.target_key][start_idx:end_idx].astype(np.float32)
        if self.keep_indices is not None:
            target_window = target_window[:, self.keep_indices]

        # Per-channel z-score normalization (EMG + IMU have very different scales)
        if self.input_mean is not None:
            emg_window = (emg_window - self.input_mean) / self.input_std

        # EMG-only ablation: drop IMU channels entirely.
        if self.emg_only:
            emg_window = emg_window[:, :self.num_emg]

        # Input: (T, C) -> (C, T) for Conv1d-style layout
        emg_tensor = torch.from_numpy(emg_window).transpose(0, 1)

        # Target: (T, N) -> (N, T). N=60 for landmarks, N=20 for angles.
        target_tensor = torch.from_numpy(target_window).transpose(0, 1)

        return emg_tensor, target_tensor
