# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DASBandConfig:
    data_root: str = "Data"
    output_root: str = "output"
    csv_utc_offset_hours: float = 8.0
    skip_channels: int = 18

    das_csv: Optional[str] = None
    audio_path: Optional[str] = None
    candidate_csv: Optional[str] = None
    name: Optional[str] = None

    das_fs: int = 2000
    trim_start_s: float = 50.0
    trim_end_s: Optional[float] = None

    das_bp_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(5.0, 10.0), (10.0, 15.0), (15.0, 30.0)]
    )
    das_filter_order: int = 4
    das_filter_method: str = "sosfilt"

    frame_win_ms: float = 100.0
    frame_step_ms: float = 25.0

    audio_sr: int = 48000
    audio_bp_low: float = 4000.0
    audio_bp_high: float = 10000.0
    audio_filter_order: int = 4
    audio_env_ms: float = 15.0
    audio_smooth_ms: float = 30.0
    audio_peak_prom: float = 1.5
    audio_peak_height: float = 0.8
    step_min_interval: float = 0.5
    label_window_s: float = 0.15

    use_signal_prior: bool = True
    gaussian_sigma_ch: float = 3.0
    hard_band_radius_ch: int = 3
    label_mode: str = "gaussian"

    clean_min_segment_points: int = 4
    clean_breakpoint_stride: int = 1
    clean_outlier_threshold_ch: float = 5.0
    clean_project_to_line: bool = False

    patch_frames: int = 256
    patch_stride: int = 128

    model_channels: int = 32
    model_dropout: float = 0.1

    loss_mask_weight: float = 1.0
    loss_center_weight: float = 0.3
    loss_smooth_weight: float = 0.2
    loss_tv_weight: float = 0.1
    loss_area_weight: float = 0.01
    center_huber_delta: float = 2.0

    device: str = "auto"
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    amp: bool = True

    dp_jump_penalty: float = 0.8
    dp_curvature_penalty: float = 0.2
    dp_max_jump_ch: int = 6
    centroid_threshold: float = 0.2
    decode_mode: str = "kalman"
    kalman_process_var: float = 2.0
    kalman_measurement_var: float = 0.8
    kalman_measurement_var_floor: float = 0.15
    kalman_init_pos_var: float = 4.0
    kalman_init_vel_var: float = 1.0
    sigma_scale: float = 0.7
    sigma_min: float = 0.5
    sigma_max: float = 3.0

    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DASBandConfig":
        return cls(**data)

    def resolve_data_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve()

    def resolve_output_root(self) -> Path:
        return Path(self.output_root).expanduser().resolve()

    def resolve_run_root(self, stage: str) -> Path:
        base = self.resolve_output_root()
        tag = self.name or "unnamed"
        return base / tag / "dasband" / stage
