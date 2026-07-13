# util/IMUPreprocess.py
import torch
import numpy as np
from scipy.signal import butter, filtfilt

class IMUPreprocessor:
    def __init__(self, fs=100.0, cutoff=0.3):
        self.fs = fs
        self.cutoff = cutoff  

    def butter_highpass(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(5, normal_cutoff, btype='high', analog=False)
        return b, a

    def remove_gravity(self, accel):
        """Remove gravity from accelerometer (first 3 channels)"""
        b, a = self.butter_highpass()
        accel_np = accel.cpu().numpy()
        filtered = np.zeros_like(accel_np)
        
        for i in range(accel_np.shape[0]):  # batch
            for j in range(3):  # accel axes
                filtered[i, :, j] = filtfilt(b, a, accel_np[i, :, j])
        
        return torch.from_numpy(filtered).float().to(accel.device)

    def normalize(self, x):
        """Z-score normalization per window per channel"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std

    def __call__(self, batch):
        imu = batch["imu"].float()  # (batch, window_size, channels)

        
        if imu.shape[-1] >= 6:
            gyro = imu[..., :3]
            accel = imu[..., 3:]

    
            accel_clean = self.remove_gravity(accel)

            # Normalize
            accel_norm = self.normalize(accel_clean)
            gyro_norm = self.normalize(gyro)

            batch["imu"] = torch.cat([accel_norm, gyro_norm], dim=-1)
        else:
            batch["imu"] = self.normalize(imu)

        return batch