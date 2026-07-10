import torch
import numpy as np
from scipy.signal import butter, filtfilt

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def remove_gravity(accel, fs=50.0, cutoff=0.3):
    """Remove gravity from accelerometer data using high-pass filter"""
    b, a = butter_highpass(cutoff, fs)
    accel_np = accel.cpu().numpy()
    filtered = np.zeros_like(accel_np)
    for i in range(accel_np.shape[1]):  # for each axis
        filtered[:, i] = filtfilt(b, a, accel_np[:, i])
    return torch.from_numpy(filtered).float().to(accel.device)

def normalize_window(window):
    """Normalize each channel (mean=0, std=1)"""
    mean = window.mean(dim=0, keepdim=True)
    std = window.std(dim=0, keepdim=True) + 1e-8
    return (window - mean) / std

class IMUPreprocessor:
    def __init__(self, fs=50.0):
        self.fs = fs

    def __call__(self, batch):
        imu = batch["imu"].float()  # shape: (batch, window_size, channels)
        

        if imu.shape[-1] >= 6:
            gyro = imu[..., :3]
            accel = imu[..., 3:]
            
            # Remove gravity from accel
            accel_detrend = torch.stack([remove_gravity(accel[i], self.fs) for i in range(len(accel))])
            
            # Normalize
            accel_norm = normalize_window(accel_detrend)
            gyro_norm = normalize_window(gyro)
            
            batch["imu"] = torch.cat([accel_norm, gyro_norm], dim=-1)
        else:
            batch["imu"] = normalize_window(imu)
        
        return batch