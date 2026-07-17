from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging


class IMUDataset(Dataset):
    def __init__(self, imu_dataset_file, window_size, input_size, window_shift=None):
        super(IMUDataset, self).__init__()
        if window_shift is None:
            window_shift = window_size
            
        df = pd.read_csv(imu_dataset_file)
        if df.shape[1] == 1:
            df = pd.read_csv(imu_dataset_file, delimiter='\t')

        # CRITICAL FIX: Force float32
        self.imu = df.iloc[:, :input_size].values.astype(np.float32)
        self.labels = df.iloc[:, input_size:].values.astype(np.int64)

        n = self.labels.shape[0]
        self.start_indices = list(range(0, n - window_size + 1, window_shift))
        self.window_size = window_size
        self.window_shift = window_shift

        logging.info(
            f"Number of windows: {len(self.start_indices)} "
            f"(generated from {n} samples, window_size={window_size}, shift={window_shift})"
        )

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, start_index + self.window_size))
        
        imu = self.imu[window_indices, :]
        window_labels = self.labels[window_indices, :]

        label = np.bincount(window_labels.flatten()).argmax()

        sample = {'imu': imu, 'label': label}
        return sample