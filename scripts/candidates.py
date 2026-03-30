# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import DASBandConfig
from .io import compute_audio_envelope, load_audio_mono, resolve_audio_path
from .signal_utils import moving_average, robust_zscore


def load_candidate_points(candidate_csv: str) -> pd.DataFrame:
    df = pd.read_csv(candidate_csv)
    required = {"time", "channel"}
    if not required.issubset(df.columns):
        raise ValueError(f"Candidate CSV must contain columns {sorted(required)}")
    out = df.copy()
    out["time"] = out["time"].astype(float)
    out["channel"] = out["channel"].astype(float)
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    return out.sort_values("time").reset_index(drop=True)


def detect_audio_step_times(audio_path: str, config: DASBandConfig):
    y, _ = load_audio_mono(audio_path, config.audio_sr)
    start_idx = max(0, int(config.trim_start_s * config.audio_sr))
    end_idx = len(y) if config.trim_end_s is None else min(len(y), int(config.trim_end_s * config.audio_sr))
    y = y[start_idx:end_idx]

    t, env = compute_audio_envelope(y, config)
    log_env = np.log(env + 1e-12)
    z = robust_zscore(log_env)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0 / float(config.audio_sr)
    z = moving_average(z, max(1, int(0.02 / max(1e-6, dt))))
    min_dist = max(1, int(config.step_min_interval / max(dt, 1e-6)))
    peaks, props = find_peaks(
        z,
        distance=min_dist,
        prominence=config.audio_peak_prom,
        height=config.audio_peak_height,
    )
    heights = props.get("peak_heights", z[peaks])
    return t[peaks].astype(np.float32), np.asarray(heights, dtype=np.float32)


def estimate_peak_channel(primary_energy: np.ndarray, frame_times: np.ndarray, event_time: float):
    idx = int(np.argmin(np.abs(frame_times - event_time)))
    channel_energy = primary_energy[idx]
    best_ch = int(np.argmax(channel_energy))
    max_e = float(np.max(channel_energy))
    mean_e = float(np.mean(channel_energy))
    std_e = float(np.std(channel_energy) + 1e-12)
    conf = float(1.0 / (1.0 + np.exp(-(max_e - mean_e) / std_e)))
    return best_ch, conf


def generate_candidates_from_audio(primary_energy: np.ndarray, frame_times: np.ndarray, config: DASBandConfig) -> pd.DataFrame:
    audio_path = resolve_audio_path(config)
    if audio_path is None:
        raise FileNotFoundError("Audio path is not available. Provide --audio_path or --name with Data/Audio.")

    step_times, heights = detect_audio_step_times(str(audio_path), config)
    rows = []
    for t, h in zip(step_times, heights):
        ch, conf = estimate_peak_channel(primary_energy, frame_times, float(t))
        rows.append((float(t), float(ch), float(conf), float(h)))
    if not rows:
        raise ValueError("No candidate points were generated from audio.")
    return pd.DataFrame(rows, columns=["time", "channel", "confidence", "audio_height"])


def resolve_candidate_points(primary_energy: np.ndarray, frame_times: np.ndarray, config: DASBandConfig) -> pd.DataFrame:
    if config.candidate_csv:
        return load_candidate_points(config.candidate_csv)
    return generate_candidates_from_audio(primary_energy, frame_times, config)
