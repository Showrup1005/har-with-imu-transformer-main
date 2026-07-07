from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging


class IMUDataset(Dataset):
    """
    IMU Dataset for Federated Learning
    Handles subject filtering and removes non-feature columns.
    """
    def __init__(self, imu_dataset_file, window_size, input_size,
                 window_shift=None, subject_ids=None):
        """
        :param subject_ids: list of subjects to include (None = use all)
        """
        super(IMUDataset, self).__init__()
        if window_shift is None:
            window_shift = window_size

        # Load data
        df = pd.read_csv(imu_dataset_file)
        if df.shape[1] == 1:  # fallback for tab-separated
            df = pd.read_csv(imu_dataset_file, delimiter='\t')

        # === Explicit column handling ===
        expected_cols = ['subject', 'label', 'timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
        
        # Keep only relevant columns in correct order
        available_cols = [col for col in expected_cols if col in df.columns]
        df = df[available_cols]

        # Filter by subjects if specified
        if subject_ids is not None and 'subject' in df.columns:
            df = df[df['subject'].isin(subject_ids)]

        # Drop non-feature columns
        drop_cols = ['subject', 'timestamp']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Now df should have: label + 6 IMU features (or just 6 features if label is last)
        if 'label' in df.columns:
            self.imu = df.drop(columns=['label']).values.astype(np.float32)
            self.labels = df['label'].values.reshape(-1, 1).astype(np.float64)
        else:
            # Fallback if label column is missing
            self.imu = df.values.astype(np.float32)
            self.labels = np.zeros((len(df), 1), dtype=np.float64)

        # Create sliding windows
        n = len(df)
        self.start_indices = list(range(0, n - window_size + 1, window_shift))
        self.window_size = window_size

        logging.info(
            f"Loaded {n} samples → {len(self.start_indices)} windows | "
            f"Features: {self.imu.shape[1]} | Subjects filtered: {subject_ids is not None}"
        )

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, start_index + self.window_size))

        imu = self.imu[window_indices, :]
        window_labels = self.labels[window_indices, :].astype(np.int64).flatten()

        # Majority vote for window label
        label = np.bincount(window_labels).argmax()

        sample = {'imu': imu, 'label': int(label)}
        return sample