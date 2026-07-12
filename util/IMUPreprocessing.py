import torch

class IMUPreprocessor:
    def __init__(self, fs=50.0):
        self.fs = fs

    def __call__(self, batch):
        """Light preprocessing: only normalization"""
        imu = batch["imu"].float()                    # (batch, seq_len, channels)
        
        # Per-window per-channel normalization
        mean = imu.mean(dim=1, keepdim=True)
        std = imu.std(dim=1, keepdim=True) + 1e-8
        imu_norm = (imu - mean) / std
        
        batch["imu"] = imu_norm
        print(batch)
        return batch