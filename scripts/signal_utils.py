# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, sosfilt


def butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low_n = max(1e-6, low / nyq)
    high_n = min(0.9999, high / nyq)
    return butter(order, [low_n, high_n], btype="band")


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4, method: str = "sosfilt"):
    x0 = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    if method == "filtfilt":
        b, a = butter_bandpass(low, high, fs, order=order)
        return filtfilt(b, a, x0)
    if method == "sosfilt":
        nyq = 0.5 * fs
        low_n = max(1e-6, low / nyq)
        high_n = min(0.9999, high / nyq)
        sos = butter(order, [low_n, high_n], btype="band", output="sos")
        return sosfilt(sos, x0)
    raise ValueError(f"Unsupported filter method: {method}")


def bandpass_filter_2d(X: np.ndarray, fs: float, low: float, high: float, order: int = 4, method: str = "sosfilt"):
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros_like(X, dtype=np.float64)
    if method == "filtfilt":
        b, a = butter_bandpass(low, high, fs, order=order)
        for c in range(X.shape[1]):
            out[:, c] = filtfilt(b, a, X[:, c] - np.mean(X[:, c]))
        return out
    if method == "sosfilt":
        nyq = 0.5 * fs
        low_n = max(1e-6, low / nyq)
        high_n = min(0.9999, high / nyq)
        sos = butter(order, [low_n, high_n], btype="band", output="sos")
        for c in range(X.shape[1]):
            out[:, c] = sosfilt(sos, X[:, c] - np.mean(X[:, c]))
        return out
    raise ValueError(f"Unsupported filter method: {method}")


def moving_average(x: np.ndarray, win: int):
    if int(win) <= 1:
        return x
    kernel = np.ones(int(win), dtype=np.float64) / float(win)
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


def short_time_rms(x: np.ndarray, win: int):
    x2 = np.asarray(x, dtype=np.float64) ** 2
    kernel = np.ones(max(1, int(win)), dtype=np.float64) / float(max(1, int(win)))
    return np.sqrt(np.convolve(x2, kernel, mode="same") + 1e-12)


def robust_zscore(x: np.ndarray, eps: float = 1e-9):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad + eps)


def robust_normalize_map(x: np.ndarray, clip: float = 6.0):
    z = robust_zscore(x)
    z = np.clip(z, -clip, clip)
    return (z / clip).astype(np.float32)


def resample_audio(y: np.ndarray, src_sr: int, dst_sr: int):
    if int(src_sr) == int(dst_sr):
        return np.asarray(y, dtype=np.float32)
    from math import gcd

    g = gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // g
    down = int(src_sr) // g
    return resample_poly(np.asarray(y, dtype=np.float32), up, down).astype(np.float32)
