# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import DASBandConfig


def build_signal_prior(primary_energy: np.ndarray):
    log_e = np.log1p(np.asarray(primary_energy, dtype=np.float64))
    mu = np.mean(log_e, axis=1, keepdims=True)
    std = np.std(log_e, axis=1, keepdims=True) + 1e-12
    z = (log_e - mu) / std
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def interpolate_centerline(clean_points_df: pd.DataFrame, frame_times: np.ndarray, num_channels: int):
    centerline = np.full(len(frame_times), np.nan, dtype=np.float32)
    if clean_points_df.empty:
        return centerline

    for seg_id, seg_df in clean_points_df.groupby("segment_id"):
        seg_df = seg_df.sort_values("time")
        t = seg_df["time"].to_numpy(dtype=np.float64)
        c = seg_df["channel"].to_numpy(dtype=np.float64)
        if len(seg_df) == 1:
            idx = int(np.argmin(np.abs(frame_times - t[0])))
            centerline[idx] = float(np.clip(c[0], 0, num_channels - 1))
            continue
        valid = (frame_times >= t[0]) & (frame_times <= t[-1])
        if np.any(valid):
            centerline[valid] = np.interp(frame_times[valid], t, c).astype(np.float32)
            
    # 替换全局高斯平滑为 Savitzky-Golay 滤波器（保边、保极值平滑）
    # 这样既能消除由于插值引发的高频“锯齿”，又能完美保留目标真实的低频“转身（极值）”轨迹
    valid_mask = np.isfinite(centerline)
    if np.sum(valid_mask) > 15:
        from scipy.signal import savgol_filter
        
        # 提取有效的一段连续波段进行平滑
        valid_idx = np.where(valid_mask)[0]
        c_valid = centerline[valid_idx]
        
        # --- 增大了平滑尺度的核心修改 ---
        # 为什么之前没有明显变化？
        # 如果 DAS 的特征帧率(Frame Rate)较高（例如10-50Hz），之前的 window_length=31 可能仅代表 0.5~3 秒。
        # 走路带来的候选点横跳误差（比如左右脚交替、或标签抖动）可能就要持续 1-2 秒，小窗口会直接“顺应”这些噪声。
        # 
        # 现在我们把窗口扩大到一个相当大的范围（最大允许 201 帧，或者总长度的三分之一），
        # 并且降低多项式阶数（polyorder=2，即只允许拟合抛物线/匀加速运动），杜绝更高次的三次、四次曲线扭曲。
        # 这样滤波器会被迫拉展直线，并在转身处画一个完美的二次抛物线。
        window_length = min(201, (len(c_valid) // 3 * 2) - 1)
        if window_length % 2 == 0:
            window_length += 1
            
        # 确保窗口长度大于多项式阶数，且有足够的平滑空间
        if window_length > 7:
            c_smooth = savgol_filter(c_valid, window_length=window_length, polyorder=2)
            centerline[valid_idx] = c_smooth.astype(np.float32)
        
    return centerline


def build_band_label(centerline: np.ndarray, num_channels: int, config: DASBandConfig):
    grid_c = np.arange(num_channels, dtype=np.float32)[None, :]
    center = centerline[:, None].astype(np.float32)
    valid = np.isfinite(center).astype(np.float32)
    if config.label_mode == "hard":
        mask = (np.abs(grid_c - center) <= float(config.hard_band_radius_ch)).astype(np.float32)
    else:
        sigma = max(1e-3, float(config.gaussian_sigma_ch))
        mask = np.exp(-0.5 * ((grid_c - center) / sigma) ** 2).astype(np.float32)
    return mask * valid


def build_pseudo_label(
    clean_points_df: pd.DataFrame,
    frame_times: np.ndarray,
    primary_energy: np.ndarray,
    config: DASBandConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_channels = int(primary_energy.shape[1])
    centerline = interpolate_centerline(clean_points_df, frame_times, num_channels)
    base_label = build_band_label(centerline, num_channels, config)
    prior = build_signal_prior(primary_energy)
    label = base_label * prior if config.use_signal_prior else base_label
    label = np.clip(label, 0.0, 1.0).astype(np.float32)
    return label, prior, centerline
