import torch
import numpy as np
import pywt  # PyWavelets

class IMUPreprocessor:
    def __init__(self, fs=100.0, cutoff=0.3):
        self.fs = fs
        self.cutoff = cutoff

    def remove_gravity(self, accel):
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(5, normal_cutoff, btype='high', analog=False)
        
        accel_np = accel.cpu().numpy()
        filtered = np.zeros_like(accel_np)
        for i in range(accel_np.shape[0]):
            for j in range(3):
                filtered[i, :, j] = filtfilt(b, a, accel_np[i, :, j])
        return torch.from_numpy(filtered).float().to(accel.device)

    def wavelet_denoise(self, x, wavelet='db4', level=2):  
        x_np = x.cpu().numpy()
        denoised = np.zeros_like(x_np)
        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[2]):
                coeffs = pywt.wavedec(x_np[i, :, j], wavelet, level=level)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                thresh = sigma * np.sqrt(2 * np.log(len(x_np[i, :, j])))
                coeffs[1:] = (pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:])
                denoised[i, :, j] = pywt.waverec(coeffs, wavelet)
        return torch.from_numpy(denoised).float().to(x.device)

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std

    def __call__(self, batch):
        imu = batch["imu"].float()

        if imu.shape[-1] >= 6:
            gyro = imu[..., :3]
            accel = imu[..., 3:]

            accel_clean = self.remove_gravity(accel)
            accel_denoised = self.wavelet_denoise(accel_clean)

            accel_norm = self.normalize(accel_denoised)
            gyro_norm = self.normalize(gyro)

            batch["imu"] = torch.cat([accel_norm, gyro_norm], dim=-1)
        else:
            imu_denoised = self.wavelet_denoise(imu)
            batch["imu"] = self.normalize(imu_denoised)

        return batch