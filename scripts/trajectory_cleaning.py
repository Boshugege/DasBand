# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression

from .config import DASBandConfig


@dataclass
class FittedLine:
    slope: float
    intercept: float

    def predict(self, t: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(t, dtype=np.float64) + self.intercept


def _fit_line(times: np.ndarray, channels: np.ndarray) -> FittedLine:
    x = np.asarray(times, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(channels, dtype=np.float64)
    if len(y) < 2:
        return FittedLine(0.0, float(y[0]) if len(y) else 0.0)
    try:
        model = HuberRegressor()
        model.fit(x, y)
        return FittedLine(float(model.coef_[0]), float(model.intercept_))
    except Exception:
        fallback = LinearRegression()
        fallback.fit(x, y)
        return FittedLine(float(fallback.coef_[0]), float(fallback.intercept_))


def _huber_cost(residuals: np.ndarray, delta: float = 2.0):
    r = np.abs(np.asarray(residuals, dtype=np.float64))
    quad = np.minimum(r, delta)
    linear = r - quad
    return float(np.sum(0.5 * quad ** 2 + delta * linear))


def fit_piecewise_trajectory(points_df: pd.DataFrame, config: DASBandConfig):
    points_df = points_df.sort_values("time").reset_index(drop=True).copy()
    n = len(points_df)
    if n < max(2 * config.clean_min_segment_points, 4):
        raise ValueError("Not enough candidate points for piecewise trajectory fitting.")

    t = points_df["time"].to_numpy(dtype=np.float64)
    c = points_df["channel"].to_numpy(dtype=np.float64)

    best = None
    min_seg = int(config.clean_min_segment_points)
    stride = max(1, int(config.clean_breakpoint_stride))

    for k in range(min_seg, n - min_seg + 1, stride):
        left_line = _fit_line(t[:k], c[:k])
        right_line = _fit_line(t[k:], c[k:])
        left_res = c[:k] - left_line.predict(t[:k])
        right_res = c[k:] - right_line.predict(t[k:])
        cost = _huber_cost(left_res) + _huber_cost(right_res)
        if best is None or cost < best["cost"]:
            best = {
                "break_index": k,
                "cost": float(cost),
                "line1": left_line,
                "line2": right_line,
            }

    if best is None:
        raise RuntimeError("Failed to fit a piecewise trajectory.")

    k = int(best["break_index"])
    fitted = np.concatenate(
        [
            best["line1"].predict(t[:k]),
            best["line2"].predict(t[k:]),
        ]
    )
    residuals = np.abs(c - fitted)
    keep = residuals <= float(config.clean_outlier_threshold_ch)

    clean = points_df.loc[keep].copy().reset_index(drop=True)
    clean["segment_id"] = np.where(clean.index < np.sum(keep[:k]), 0, 1)

    t_clean = clean["time"].to_numpy(dtype=np.float64)
    c_clean = clean["channel"].to_numpy(dtype=np.float64)
    left_mask = clean["segment_id"].to_numpy(dtype=int) == 0
    right_mask = ~left_mask

    line1 = _fit_line(t_clean[left_mask], c_clean[left_mask]) if np.any(left_mask) else best["line1"]
    line2 = _fit_line(t_clean[right_mask], c_clean[right_mask]) if np.any(right_mask) else best["line2"]

    fitted_clean = np.empty(len(clean), dtype=np.float64)
    if np.any(left_mask):
        fitted_clean[left_mask] = line1.predict(t_clean[left_mask])
    if np.any(right_mask):
        fitted_clean[right_mask] = line2.predict(t_clean[right_mask])
    clean["fitted_channel"] = fitted_clean
    clean["residual"] = np.abs(clean["channel"].to_numpy(dtype=np.float64) - fitted_clean)

    if config.clean_project_to_line:
        clean["channel"] = clean["fitted_channel"]

    summary = {
        "break_index": int(k),
        "n_raw": int(len(points_df)),
        "n_clean": int(len(clean)),
        "n_removed": int(len(points_df) - len(clean)),
        "line1": {"slope": line1.slope, "intercept": line1.intercept},
        "line2": {"slope": line2.slope, "intercept": line2.intercept},
        "break_time": float(t[k]),
    }
    return clean, summary, (line1, line2)
