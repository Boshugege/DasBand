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
        
        # 自适应窗口大小：假设真实人的转身动作不可能是瞬态的，而是秒级的
        # polyorder=3 可以完美拟合局部的抛物线（转身时的加减速特征）而不将其抹平
        window_length = min(31, (len(c_valid) // 2 * 2) - 1)  # 必须是奇数
        if window_length > 3:
            c_smooth = savgol_filter(c_valid, window_length=window_length, polyorder=3)
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
