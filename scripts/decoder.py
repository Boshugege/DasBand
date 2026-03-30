# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from .config import DASBandConfig


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weighted_centroid(mask: np.ndarray, threshold: float = 0.0):
    mask = np.asarray(mask, dtype=np.float64)
    if threshold > 0:
        mask = np.where(mask >= threshold, mask, 0.0)
    channels = np.arange(mask.shape[1], dtype=np.float64)[None, :]
    numer = np.sum(mask * channels, axis=1)
    denom = np.sum(mask, axis=1) + 1e-8
    return (numer / denom).astype(np.float32)


def estimate_measurement_confidence(mask: np.ndarray):
    mask = np.asarray(mask, dtype=np.float64)
    peak = np.max(mask, axis=1)
    mass = np.sum(mask, axis=1)
    concentration = peak / (mass + 1e-6)
    conf = 0.7 * peak + 0.3 * np.clip(20.0 * concentration, 0.0, 1.0)
    return np.clip(conf, 0.05, 0.95).astype(np.float32)


def extract_path_dp(mask: np.ndarray, config: DASBandConfig):
    prob = np.clip(np.asarray(mask, dtype=np.float64), 1e-6, 1.0)
    emission = np.log(prob)
    T, C = emission.shape
    max_jump = max(1, int(config.dp_max_jump_ch))
    lam = float(config.dp_jump_penalty)

    score = np.full((T, C), -np.inf, dtype=np.float64)
    prev = np.full((T, C), -1, dtype=np.int32)
    score[0] = emission[0]

    for t in range(1, T):
        for c in range(C):
            lo = max(0, c - max_jump)
            hi = min(C, c + max_jump + 1)
            prev_cands = np.arange(lo, hi)
            candidate_score = score[t - 1, prev_cands] - lam * np.abs(prev_cands - c)
            if t >= 2 and config.dp_curvature_penalty > 0:
                ref = prev[t - 1, prev_cands]
                valid_ref = ref >= 0
                curvature = np.zeros_like(candidate_score)
                curvature[valid_ref] = float(config.dp_curvature_penalty) * np.abs(c - 2 * prev_cands[valid_ref] + ref[valid_ref])
                candidate_score = candidate_score - curvature
            best_idx = int(np.argmax(candidate_score))
            score[t, c] = emission[t, c] + candidate_score[best_idx]
            prev[t, c] = int(prev_cands[best_idx])

    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = max(0, int(prev[t, path[t]]))
    return path.astype(np.float32)


def kalman_smooth_track(measurements: np.ndarray, frame_times: np.ndarray, measurement_confidence: np.ndarray, config: DASBandConfig):
    z = np.asarray(measurements, dtype=np.float64)
    t = np.asarray(frame_times, dtype=np.float64)
    conf = np.clip(np.asarray(measurement_confidence, dtype=np.float64), 0.05, 0.95)
    n = len(z)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    x_filt = np.zeros((n, 2), dtype=np.float64)
    p_filt = np.zeros((n, 2, 2), dtype=np.float64)
    x_pred = np.zeros((n, 2), dtype=np.float64)
    p_pred = np.zeros((n, 2, 2), dtype=np.float64)

    x_filt[0] = np.array([z[0], 0.0], dtype=np.float64)
    p_filt[0] = np.diag([float(config.kalman_init_pos_var), float(config.kalman_init_vel_var)])

    for i in range(1, n):
        dt = max(1e-6, float(t[i] - t[i - 1]))
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = float(config.kalman_process_var)
        Q = q * np.array(
            [[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]],
            dtype=np.float64,
        )

        x_pred[i] = F @ x_filt[i - 1]
        p_pred[i] = F @ p_filt[i - 1] @ F.T + Q

        H = np.array([[1.0, 0.0]], dtype=np.float64)
        R = float(config.kalman_measurement_var_floor) + float(config.kalman_measurement_var) / (conf[i] ** 2)
        S = H @ p_pred[i] @ H.T + np.array([[R]], dtype=np.float64)
        K = p_pred[i] @ H.T @ np.linalg.inv(S)
        innovation = np.array([z[i] - (H @ x_pred[i])[0]], dtype=np.float64)
        x_filt[i] = x_pred[i] + (K @ innovation).reshape(-1)
        p_filt[i] = (np.eye(2, dtype=np.float64) - K @ H) @ p_pred[i]

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for i in range(n - 2, -1, -1):
        dt = max(1e-6, float(t[i + 1] - t[i]))
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        c = p_filt[i] @ F.T @ np.linalg.inv(p_pred[i + 1] + 1e-9 * np.eye(2))
        x_smooth[i] = x_filt[i] + c @ (x_smooth[i + 1] - x_pred[i + 1])
        p_smooth[i] = p_filt[i] + c @ (p_smooth[i + 1] - p_pred[i + 1]) @ c.T

    return x_smooth[:, 0].astype(np.float32), x_smooth[:, 1].astype(np.float32)


def estimate_uncertainty(mask: np.ndarray, path: np.ndarray, config: DASBandConfig | None = None):
    prob = np.asarray(mask, dtype=np.float64)
    channels = np.arange(prob.shape[1], dtype=np.float64)[None, :]
    path = np.asarray(path, dtype=np.float64)[:, None]
    numer = np.sum(((channels - path) ** 2) * prob, axis=1)
    denom = np.sum(prob, axis=1) + 1e-8
    sigma = np.sqrt(numer / denom).astype(np.float32)
    if config is not None:
        sigma = sigma * float(config.sigma_scale)
        sigma = np.clip(sigma, float(config.sigma_min), float(config.sigma_max))
    return sigma.astype(np.float32)
