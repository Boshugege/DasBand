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
