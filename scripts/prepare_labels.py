# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from .config import DASBandConfig
from .pipeline import prepare_training_labels


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare dasband pseudo labels from DAS CSV + candidate points/audio.")
    parser.add_argument("--das_csv", required=True, help="Path to DAS CSV.")
    parser.add_argument("--candidate_csv", default=None, help="Optional candidate point CSV with columns time,channel.")
    parser.add_argument("--audio_path", default=None, help="Optional audio path used when candidate_csv is not provided.")
    parser.add_argument("--name", default=None, help="Sample name for output organization and Data/Audio auto lookup.")
    parser.add_argument("--data_root", default="Data", help="Data root. Used to auto-resolve audio when --name is given.")
    parser.add_argument("--output_root", default="output", help="Output root.")
    parser.add_argument("--trim_start_s", type=float, default=50.0)
    parser.add_argument("--trim_end_s", type=float, default=None)
    parser.add_argument("--das_fs", type=int, default=2000)
    parser.add_argument("--das_filter_method", choices=["filtfilt", "sosfilt"], default="sosfilt")
    parser.add_argument("--gaussian_sigma_ch", type=float, default=3.0)
    parser.add_argument("--label_mode", choices=["gaussian", "hard"], default="gaussian")
    parser.add_argument("--hard_band_radius_ch", type=int, default=3)
    parser.add_argument("--use_signal_prior", action="store_true", default=True)
    parser.add_argument("--disable_signal_prior", action="store_true")
    parser.add_argument("--clean_outlier_threshold_ch", type=float, default=5.0)
    parser.add_argument("--clean_project_to_line", action="store_true")
    parser.add_argument("--clean_min_segment_points", type=int, default=4)
    return parser


def main():
    args = build_parser().parse_args()
    config = DASBandConfig(
        das_csv=args.das_csv,
        candidate_csv=args.candidate_csv,
        audio_path=args.audio_path,
        name=args.name,
        data_root=args.data_root,
        output_root=args.output_root,
        trim_start_s=args.trim_start_s,
        trim_end_s=args.trim_end_s,
        das_fs=args.das_fs,
        das_filter_method=args.das_filter_method,
        gaussian_sigma_ch=args.gaussian_sigma_ch,
        label_mode=args.label_mode,
        hard_band_radius_ch=args.hard_band_radius_ch,
        use_signal_prior=not args.disable_signal_prior,
        clean_outlier_threshold_ch=args.clean_outlier_threshold_ch,
        clean_project_to_line=args.clean_project_to_line,
        clean_min_segment_points=args.clean_min_segment_points,
    )
    result = prepare_training_labels(config)
    print(f"[Prepare] saved to: {Path(result['prep_dir']).resolve()}")


if __name__ == "__main__":
    main()
