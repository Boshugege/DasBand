# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

from .config import DASBandConfig
from .signal_utils import bandpass_filter, bandpass_filter_2d, moving_average, robust_normalize_map, short_time_rms


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_das_csv(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    return df.values.astype(np.float32), list(df.columns)


def trim_das(das: np.ndarray, fs: int, trim_start_s: float, trim_end_s: Optional[float]):
    start = max(0, int(trim_start_s * fs))
    end = das.shape[0] if trim_end_s is None else min(das.shape[0], int(trim_end_s * fs))
    if end <= start:
        raise ValueError(f"Invalid trim range: start={trim_start_s}, end={trim_end_s}")
    return das[start:end], start / float(fs), end / float(fs)


def compute_frame_geometry(config: DASBandConfig):
    win_samples = max(1, int(config.frame_win_ms * 1e-3 * config.das_fs))
    step_samples = max(1, int(config.frame_step_ms * 1e-3 * config.das_fs))
    return win_samples, step_samples


def _compute_frame_stat_map(signal_tc: np.ndarray, config: DASBandConfig, stat: str) -> Tuple[np.ndarray, np.ndarray]:
    win_samples, step_samples = compute_frame_geometry(config)
    T, C = signal_tc.shape
    frames = []
    frame_times = []
    for start in range(0, T - win_samples + 1, step_samples):
        window = signal_tc[start : start + win_samples]
        if stat == "energy":
            value = np.sum(window ** 2, axis=0)
        elif stat == "envelope":
            value = np.mean(np.abs(window), axis=0)
        elif stat == "coherence":
            x = window - np.mean(window, axis=0, keepdims=True)
            energy = np.sqrt(np.sum(x ** 2, axis=0) + 1e-12)
            corr = np.zeros(C, dtype=np.float64)
            if C > 1:
                prod = np.sum(x[:, 1:] * x[:, :-1], axis=0)
                denom = energy[1:] * energy[:-1] + 1e-12
                edge = prod / denom
                corr[1:] += edge
                corr[:-1] += edge
                corr = corr / np.maximum(1.0, np.array([1] + [2] * max(0, C - 2) + ([1] if C > 1 else []), dtype=np.float64))
            value = corr
        else:
            raise ValueError(f"Unsupported stat: {stat}")
        frames.append(value)
        frame_times.append((start + win_samples // 2) / float(config.das_fs))
    if not frames:
        raise ValueError("No frames available after trimming. Reduce frame window or provide longer data.")
    return np.stack(frames, axis=0), np.asarray(frame_times, dtype=np.float32)


def build_feature_cube(das: np.ndarray, config: DASBandConfig):
    feature_maps: List[np.ndarray] = []
    feature_names: List[str] = []
    frame_times: Optional[np.ndarray] = None
    primary_energy = None

    for low, high in config.das_bp_bands:
        filtered = bandpass_filter_2d(
            das,
            config.das_fs,
            low,
            high,
            order=config.das_filter_order,
            method=config.das_filter_method,
        )
        energy_map, frame_times = _compute_frame_stat_map(filtered, config, "energy")
        env_map, _ = _compute_frame_stat_map(filtered, config, "envelope")
        coh_map, _ = _compute_frame_stat_map(filtered, config, "coherence")

        if primary_energy is None:
            primary_energy = energy_map.astype(np.float32)

        feature_maps.extend(
            [
                robust_normalize_map(np.log1p(energy_map)),
                robust_normalize_map(env_map),
                robust_normalize_map(coh_map),
            ]
        )
        band_tag = f"{low:g}_{high:g}Hz"
        feature_names.extend(
            [f"log_energy_{band_tag}", f"envelope_{band_tag}", f"coherence_{band_tag}"]
        )

    raw_env = np.abs(das.astype(np.float64))
    win_samples, _ = compute_frame_geometry(config)
    smoothed = np.stack([moving_average(raw_env[:, c], win_samples) for c in range(raw_env.shape[1])], axis=1)
    env_map, _ = _compute_frame_stat_map(smoothed, config, "envelope")
    feature_maps.append(robust_normalize_map(env_map))
    feature_names.append("raw_envelope")

    cube = np.stack(feature_maps, axis=0).astype(np.float32)  # [B, Tf, C]
    primary_energy = primary_energy.astype(np.float32)
    return cube, feature_names, frame_times, primary_energy


def load_audio_mono(audio_path: str, target_sr: int):
    data, sr = sf.read(audio_path, always_2d=True)
    y = np.mean(data, axis=1).astype(np.float32)
    from .signal_utils import resample_audio

    return resample_audio(y, int(sr), int(target_sr)), int(target_sr)


def compute_audio_envelope(y: np.ndarray, config: DASBandConfig):
    y_bp = bandpass_filter(
        y,
        config.audio_sr,
        config.audio_bp_low,
        config.audio_bp_high,
        order=config.audio_filter_order,
        method="filtfilt",
    )
    env = short_time_rms(y_bp, int(max(1, config.audio_env_ms * 1e-3 * config.audio_sr)))
    env = moving_average(env, int(max(1, config.audio_smooth_ms * 1e-3 * config.audio_sr)))
    t = np.arange(len(env), dtype=np.float32) / float(config.audio_sr)
    return t, env.astype(np.float32)


def resolve_audio_path(config: DASBandConfig) -> Optional[Path]:
    if config.audio_path:
        return Path(config.audio_path).expanduser().resolve()
    if not config.name:
        return None
    data_root = config.resolve_data_root()
    for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".MP4", ".mp4", ".MOV", ".mov"]:
        cand = data_root / "Audio" / f"{config.name}{ext}"
        if cand.exists():
            return cand.resolve()
    return None


def save_prepare_artifacts(
    prep_dir: Path,
    feature_cube: np.ndarray,
    feature_names: List[str],
    frame_times: np.ndarray,
    primary_energy: np.ndarray,
    prior: np.ndarray,
    pseudo_label: np.ndarray,
    centerline: np.ndarray,
    cleaned_points_df: pd.DataFrame,
    raw_points_df: pd.DataFrame,
    meta: dict,
):
    ensure_dir(prep_dir)
    np.save(prep_dir / "feature_cube.npy", feature_cube)
    np.save(prep_dir / "frame_times.npy", frame_times)
    np.save(prep_dir / "primary_energy.npy", primary_energy)
    np.save(prep_dir / "prior.npy", prior)
    np.save(prep_dir / "pseudo_label.npy", pseudo_label)
    np.save(prep_dir / "centerline.npy", centerline)
    raw_points_df.to_csv(prep_dir / "candidate_points_raw.csv", index=False)
    cleaned_points_df.to_csv(prep_dir / "candidate_points_clean.csv", index=False)
    save_json(prep_dir / "metadata.json", {**meta, "feature_names": feature_names})


def load_prepare_artifacts(prep_dir: str):
    prep = Path(prep_dir).expanduser().resolve()
    meta = json.loads((prep / "metadata.json").read_text(encoding="utf-8"))
    return {
        "prep_dir": prep,
        "feature_cube": np.load(prep / "feature_cube.npy"),
        "frame_times": np.load(prep / "frame_times.npy"),
        "primary_energy": np.load(prep / "primary_energy.npy"),
        "prior": np.load(prep / "prior.npy"),
        "pseudo_label": np.load(prep / "pseudo_label.npy"),
        "centerline": np.load(prep / "centerline.npy"),
        "raw_points": pd.read_csv(prep / "candidate_points_raw.csv"),
        "clean_points": pd.read_csv(prep / "candidate_points_clean.csv"),
        "meta": meta,
    }
