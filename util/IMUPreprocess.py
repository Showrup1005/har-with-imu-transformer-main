# preprocess.py - Light version
import torch

class IMUPreprocessor:
    def __init__(self, fs=50.0):
        self.fs = fs

    def __call__(self, batch):
        imu = batch["imu"].float()
        
        # Only light per-channel normalization (no gravity removal for now)
        mean = imu.mean(dim=1, keepdim=True)   # mean across time
        std = imu.std(dim=1, keepdim=True) + 1e-8
        imu = (imu - mean) / std
        
        batch["imu"] = imu
        return batch